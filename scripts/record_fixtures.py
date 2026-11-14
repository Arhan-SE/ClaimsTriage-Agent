"""Run each demo scenario against the LIVE data APIs and save what the tools returned.

Usage (from the project root, with API keys in .env):
    ./venv/bin/python scripts/record_fixtures.py [scenario-id ...]

The saved snapshots power the "Recorded snapshot" demo mode, so the demo still works if a
public data API is down or the venue network blocks it (the LLM call itself still needs internet).
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agents import Runner  # noqa: E402

from agent import apply_guardrails, build_prompt, triage_agent  # noqa: E402
from mock_data import SCENARIOS  # noqa: E402
from tools import RunCtx  # noqa: E402


async def record(scenario: dict) -> None:
    ctx = RunCtx(mode="record")
    res = await Runner.run(triage_agent, input=build_prompt(scenario["claim"]), context=ctx, max_turns=14)
    final, changes = apply_guardrails(res.final_output, ctx.tool_log)
    out = ROOT / "fixtures" / f"{scenario['id']}.json"
    out.write_text(json.dumps(ctx.captured, indent=2))
    print(f"\n=== {scenario['id']} ===")
    print(f"tools: {[t['name'] + ('' if t['ok'] else ' (FAILED)') for t in ctx.tool_log]}")
    print(f"severity={final.severity} coverage={final.coverage} fraud={final.fraud_risk} "
          f"route={final.route} confidence={final.confidence}")
    print(f"summary: {final.summary}")
    if changes:
        print(f"guardrails: {changes}")
    print(f"saved -> {out}")


async def main() -> None:
    wanted = set(sys.argv[1:])
    for s in SCENARIOS:
        if not wanted or s["id"] in wanted:
            await record(s)


if __name__ == "__main__":
    asyncio.run(main())
