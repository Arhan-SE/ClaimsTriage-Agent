# ClaimsTriage Agent

**Track:** Claims Triage Agent (Insurance) · Agentic AI Showcase, Aerophilia 2026

An autonomous agent that reads an insurance claim and returns **severity, coverage, fraud risk and the right handler**, with a special focus on **anti-selection**: customers who buy cover (or claim) only *after* a disaster has already happened. It checks the claim against real hazard data and shows every step live.

> Demo with **synthetic claims and policies**. Hazard evidence (earthquakes, rainfall, river flow, news) is real public data. Not a production underwriting system.

## What the agent does

```
PLAN → INVESTIGATE (parallel tools) → SELF-CHECK → (follow-up tools if gaps) → RECOMMEND → GUARDRAILS → HUMAN DECIDES
```

1. **Plan.** The first action is forced to be `record_plan`: competing hypotheses (genuine / anti-selection / fabricated), checks to run, and checks to skip with reasons.
2. **Investigate.** The agent chooses tools by claim type (earthquake claim → USGS; flood claim → rainfall and river data) and batches independent calls in parallel.
3. **Self-check.** It lists evidence for and against the claim and what is missing; if a gap can be closed it makes follow-up calls (shown in the UI as "follow-up after self-check").
4. **Recommend.** Structured output: severity, coverage, fraud risk, route, confidence, timeline, cited evidence and what could not be verified.
5. **Guardrails (plain code, not the LLM).** E.g. unverifiable policy or no hazard evidence → forced to human review; no auto-approve unless low fraud, covered, not major, confidence ≥ 80.
6. **Human in the loop.** The agent can never reject a claim or confirm fraud. A person accepts or overrides; everything is written to an audit log.

### Tools

| Tool | Source |
|---|---|
| `lookup_policy`, `get_claim_history` | Synthetic records (`mock_data.py`) |
| `geocode_location` | OpenStreetMap Nominatim |
| `check_earthquakes` | USGS earthquake catalogue |
| `check_rain_and_flood` | Open-Meteo (archive + flood API) |
| `search_disaster_news` | Tavily (date-windowed) |
| `days_between` | Deterministic calculator (the LLM never does date maths) |
| `record_plan`, `self_check` | The agent's own visible reasoning steps |

Date windows are computed in code (`tools.search_window`), so the agent searches the right period around both the loss date and the policy start.

### Robustness

- Tools never raise: failures return an error the agent can see and report under "Could not verify".
- Unknown policy ID → agent escalates to human review instead of guessing.
- Meaningless signals are suppressed (e.g. river-flow ratios against a near-zero baseline).
- Prompt-injection hygiene: tool output and the claim text are treated as untrusted data.
- **Recorded snapshot mode** replays saved tool results from `fixtures/` if a public data API is down. (The LLM call itself still needs internet.)

## Demo scenarios

| Scenario | Expected behaviour |
|---|---|
| Bought cover after the quake (Turkey, Feb 2023) | Policy starts the day after the M7.8 quake → HIGH fraud risk, not covered, **Fraud unit** |
| Genuine long-time customer (same quake) | Verified disaster, long-standing policy → LOW fraud, covered, **Adjuster review** |
| Flood claim, no flood (Bengaluru, Mar 2023) | No rain/river/news evidence + repeat claims → HIGH fraud risk, **Fraud unit** |

The first two use the *same* real disaster, showing the agent discriminates rather than reacting to keywords. Outputs come from an LLM and can vary slightly between runs.

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add OPENAI_API_KEY and TAVILY_API_KEY
python -m uvicorn main:app --port 8000
# open http://localhost:8000
```

Optional: `AGENT_MODEL=gpt-4.1` (default). Re-record snapshots with `python scripts/record_fixtures.py`. Run tests with `pytest`.

## Files

- `main.py`: FastAPI app, SSE streaming of the agent's steps, human-decision and audit endpoints
- `agent.py`: output schema, instructions, deterministic guardrails
- `tools.py`: all tools, live/recorded/record dispatch
- `mock_data.py`: synthetic policies, claim history, scenarios
- `index.html`: single-file UI (no build step, no CDN)
- `fixtures/`: recorded tool outputs · `tests/`: guardrail and date-window tests

## Known limitations (honest list)

- Mock policy and claim-history systems; no real insurer integration.
- Open-Meteo river data is modelled (GloFAS), coarse for small streams; news search can return noise.
- Single claim at a time. Surge detection across many claims is out of scope.
- The audit log is a local JSONL file, and the in-memory run store resets when the server restarts.

## AI usage disclosure

This project was built with **Claude Code** (Anthropic's AI coding assistant) for design review, code generation and testing. The agent at runtime uses OpenAI models through the OpenAI Agents SDK. The team reviewed and tested the results.
# Aftershock
