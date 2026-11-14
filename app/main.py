import json
import os
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.agent import TriageResult, apply_guardrails, build_prompt, triage_agent  # noqa: E402
from agents import Runner  # noqa: E402
from app.mock_data import SCENARIOS  # noqa: E402
from app.tools import REASONING_TOOLS, TOOL_META, RunCtx  # noqa: E402

BASE_DIR = Path(__file__).parent  # app/
ROOT_DIR = BASE_DIR.parent
FIXTURE_DIR = BASE_DIR / "fixtures"
AUDIT_FILE = ROOT_DIR / "audit_log.jsonl"

MAX_RUN_SECONDS = 150

app = FastAPI(title="ClaimsTriage Agent")

RUNS: dict[str, dict] = {}  # run_id -> {"claim":..., "verdict":...}


PERILS = Literal["Earthquake", "Flood", "Storm", "Fire"]


class CustomPolicy(BaseModel):
    """Policy details typed in by the user, so any scenario can be created without editing code."""

    policy_start: date
    perils_covered: list[PERILS] = Field(min_length=1)
    sum_insured_usd: float = Field(gt=0, lt=1e9)
    deductible_usd: float = Field(ge=0, lt=1e9)
    waiting_period_days: int = Field(ge=0, le=365)

    def as_policy_record(self, location: str) -> dict:
        d = self.model_dump(mode="json")
        d["policy_start"] = str(self.policy_start)
        return {
            "holder": "Custom policy (user-entered)",
            "insured_location": location,
            **d,
            "exclusions": ["Loss caused by an event that occurred before the policy start date"],
        }


class ClaimRequest(BaseModel):
    policy_id: str = Field(min_length=1, max_length=40)
    location: str = Field(min_length=1, max_length=120)
    claim_type: PERILS
    loss_date: date
    claimed_amount_usd: float = Field(gt=0, lt=1e9)
    description: str = Field(max_length=1000)
    data_mode: Literal["live", "recorded"] = "live"
    scenario_id: Optional[str] = None
    custom_policy: Optional[CustomPolicy] = None


class DecisionRequest(BaseModel):
    run_id: str
    action: Literal["accept", "override"]
    route: Optional[Literal["AUTO_APPROVE", "ADJUSTER_REVIEW", "FRAUD_UNIT", "HUMAN_REVIEW"]] = None
    note: str = Field(default="", max_length=500)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def audit(record: dict) -> None:
    record = {"logged_at": datetime.now(timezone.utc).isoformat(), **record}
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_fixtures(scenario_id: Optional[str]) -> Optional[dict]:
    if not scenario_id or not scenario_id.replace("-", "").isalnum():
        return None
    path = FIXTURE_DIR / f"{scenario_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


@app.get("/api/scenarios")
async def scenarios():
    return SCENARIOS


@app.post("/api/investigate")
async def investigate(req: ClaimRequest):
    claim = req.model_dump(mode="json", exclude={"data_mode", "scenario_id", "custom_policy"})
    custom_policy = req.custom_policy.as_policy_record(req.location) if req.custom_policy else None

    async def stream():
        run_id = uuid.uuid4().hex[:10]
        started = time.perf_counter()
        yield sse("status", {"phase": "start", "message": f"Claim received. Run {run_id}."})
        if not os.getenv("OPENAI_API_KEY"):
            yield sse("error", {"message": "OPENAI_API_KEY is not set. Add it to the .env file and restart the server."})
            return

        ctx = RunCtx(mode="live", custom_policy=custom_policy)
        if req.data_mode == "recorded":
            fixtures = load_fixtures(req.scenario_id)
            if fixtures is None:
                yield sse("error", {"message": "No recorded snapshot exists for an edited/custom claim. Switch to Live data."})
                return
            ctx = RunCtx(mode="recorded", fixtures=fixtures, custom_policy=custom_policy)

        names: dict[str, str] = {}  # call_id -> tool name
        followup = False
        evidence_calls = 0
        try:
            result = Runner.run_streamed(triage_agent, input=build_prompt(claim), context=ctx, max_turns=14)
            async for ev in result.stream_events():
                if time.perf_counter() - started > MAX_RUN_SECONDS:
                    result.cancel()
                    raise TimeoutError(f"The agent took longer than {MAX_RUN_SECONDS}s and was stopped.")
                if ev.type != "run_item_stream_event":
                    continue
                item = ev.item
                if item.type == "tool_call_item":
                    raw = item.raw_item
                    name = getattr(raw, "name", "unknown")
                    call_id = getattr(raw, "call_id", None) or uuid.uuid4().hex
                    names[call_id] = name
                    try:
                        args = json.loads(getattr(raw, "arguments", "{}") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if name == "record_plan":
                        yield sse("plan", {
                            "hypotheses": args.get("hypotheses", []),
                            "checks_to_run": args.get("checks_to_run", []),
                            "checks_skipped": args.get("checks_skipped", []),
                        })
                    elif name == "self_check":
                        followup = True
                        yield sse("self_check", {
                            "supports": args.get("supports_claim", []),
                            "contradicts": args.get("contradicts_claim", []),
                            "gaps": args.get("gaps", []),
                            "next_step": args.get("next_step", ""),
                        })
                    else:
                        evidence_calls += 1
                        label, source = TOOL_META.get(name, (name, "Tool"))
                        yield sse("tool_call", {
                            "id": call_id, "name": name, "label": label, "source": source,
                            "followup": followup, "arguments": args,
                        })
                elif item.type == "tool_call_output_item":
                    call_id = getattr(item, "call_id", None)
                    name = names.get(call_id, "")
                    if name in REASONING_TOOLS:
                        continue
                    try:
                        out = json.loads(item.output) if isinstance(item.output, str) else item.output
                    except json.JSONDecodeError:
                        out = {"raw": str(item.output)}
                    out = out if isinstance(out, dict) else {"raw": out}
                    yield sse("tool_output", {
                        "id": call_id, "name": name, "ok": "error" not in out,
                        "headline": out.get("headline") or out.get("error") or "done",
                        "data_mode": out.get("data_mode", "mock"),
                        "output": out,
                    })

            yield sse("status", {"phase": "decide", "message": "Applying guardrails to the agent's recommendation."})
            raw_result: TriageResult = result.final_output
            final, changes = apply_guardrails(raw_result, ctx.tool_log)
            if changes:
                yield sse("guardrail", {"changes": changes})

            metrics = {
                "seconds": round(time.perf_counter() - started, 1),
                "tool_calls": evidence_calls,
                "tool_failures": sum(1 for t in ctx.tool_log if not t["ok"]),
                "data_mode": ctx.mode,
                "model": triage_agent.model,
            }
            verdict = final.model_dump()
            RUNS[run_id] = {"claim": claim, "verdict": verdict}
            audit({
                "type": "agent_recommendation", "run_id": run_id, "claim": claim,
                "custom_policy": custom_policy,
                "agent_raw_route": raw_result.route, "guardrail_changes": changes,
                "verdict": verdict, "tool_log": ctx.tool_log, "metrics": metrics,
            })
            yield sse("final", {"run_id": run_id, "verdict": verdict, "metrics": metrics})
        except Exception as e:  # never leave the UI hanging
            yield sse("error", {"message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/decision")
async def decision(req: DecisionRequest):
    run = RUNS.get(req.run_id)
    if not run:
        raise HTTPException(404, "Unknown run_id")
    if req.action == "override" and not req.route:
        raise HTTPException(422, "Override requires a route")
    final_route = run["verdict"]["route"] if req.action == "accept" else req.route
    audit({
        "type": "human_decision", "run_id": req.run_id, "action": req.action,
        "agent_route": run["verdict"]["route"], "final_route": final_route, "note": req.note,
    })
    return {"ok": True, "final_route": final_route}


@app.get("/api/audit")
async def audit_download():
    if not AUDIT_FILE.exists():
        return PlainTextResponse("", media_type="application/x-ndjson")
    return FileResponse(AUDIT_FILE, media_type="application/x-ndjson", filename="audit_log.jsonl")


@app.get("/", response_class=HTMLResponse)
async def ui():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")
