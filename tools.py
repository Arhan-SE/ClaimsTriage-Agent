"""Agent tools.

Design rules:
  * Every tool returns JSON with a short human-readable `headline` (shown in the UI trace).
  * Failures never raise: they return {"error": ...} so the agent can adapt and the guardrails
    can see that evidence is missing.
  * Date windows are computed here, in code, so the LLM never does date arithmetic.
  * Network tools run through `_dispatch`, which supports three modes:
      live     -> call the real public API
      recorded -> replay a saved snapshot (demo still works if a data API is down)
      record   -> call live AND capture the result (used by scripts/record_fixtures.py)
"""

import asyncio
import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, List, Optional

import httpx
from agents import RunContextWrapper, function_tool
from tavily import TavilyClient

from mock_data import CLAIM_HISTORY, POLICIES

# Tools that only record the agent's own reasoning (shown in the UI, not "evidence").
REASONING_TOOLS = {"record_plan", "self_check"}
# Tools that gather real-world hazard evidence (used by the guardrails).
HAZARD_TOOLS = {"check_earthquakes", "check_rain_and_flood", "search_disaster_news"}

MIN_MEANINGFUL_DISCHARGE = 5.0  # m3/s; below this a peak/baseline ratio is noise

# Label + source name for the UI.
TOOL_META = {
    "record_plan": ("Plan", "Agent"),
    "self_check": ("Self-check", "Agent"),
    "lookup_policy": ("Policy lookup", "Policy system (mock)"),
    "get_claim_history": ("Claim history", "Claims DB (mock)"),
    "geocode_location": ("Geocode", "OpenStreetMap"),
    "check_earthquakes": ("Earthquake check", "USGS"),
    "check_rain_and_flood": ("Rainfall & river flow", "Open-Meteo"),
    "search_disaster_news": ("News search", "Tavily"),
    "days_between": ("Date calculator", "Calculator"),
}


@dataclass
class RunCtx:
    """Per-run state handed to every tool via the Agents SDK context."""

    mode: str = "live"  # live | recorded | record
    fixtures: dict = field(default_factory=dict)  # tool name -> [{"args":..,"output":..}]
    captured: dict = field(default_factory=dict)  # filled in "record" mode
    tool_log: List[dict] = field(default_factory=list)  # [{"name":..,"ok":..}]
    custom_policy: Optional[dict] = None  # user-entered policy; overrides the mock lookup

    def lookup_recorded(self, name: str, args: dict) -> Optional[dict]:
        entries = self.fixtures.get(name) or []
        for e in entries:
            if e["args"] == args:
                return e["output"]
        return entries[0]["output"] if entries else None


async def _dispatch(
    ctx: RunContextWrapper[RunCtx],
    name: str,
    args: dict,
    live_fn: Callable[..., Awaitable[dict]],
) -> str:
    c = ctx.context
    if c.mode == "recorded":
        out = c.lookup_recorded(name, args)
        if out is None:
            out = {
                "error": f"No recorded snapshot for {name}",
                "headline": f"{name}: no recorded snapshot (switch to live data)",
            }
        else:
            out = {**out, "data_mode": "recorded"}
    else:
        try:
            out = await live_fn(**args)
        except Exception as e:  # tools must never crash the run
            out = {"error": f"{type(e).__name__}: {e}", "headline": f"{name} failed: {e}"}
        out["data_mode"] = "live"
        if c.mode == "record":
            c.captured.setdefault(name, []).append({"args": args, "output": out})
    c.tool_log.append({"name": name, "ok": "error" not in out})
    return json.dumps(out)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _parse(d: str) -> date:
    return date.fromisoformat(d)


def search_window(loss_date: str, policy_start_date: str) -> tuple[date, date]:
    """Window that always covers the loss date and, if the policy is recent, the weeks before
    it started (so a disaster that happened just BEFORE the policy began is not missed)."""
    loss = _parse(loss_date)
    start = loss - timedelta(days=30)
    try:
        ps = _parse(policy_start_date)
        if 0 <= (loss - ps).days <= 90:
            start = min(start, ps - timedelta(days=21))
    except (ValueError, TypeError):
        pass
    return start, loss + timedelta(days=2)


# ----------------------------------------------------------------------------
# live implementations
# ----------------------------------------------------------------------------
async def _geocode(location: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1},
            headers={"User-Agent": "ClaimsTriageDemoAgent/1.0"},
        )
        r.raise_for_status()
        data = r.json()
    if not data:
        return {"error": f"Could not geocode '{location}'", "headline": f"Could not locate '{location}'"}
    hit = data[0]
    return {
        "display_name": hit.get("display_name"),
        "latitude": float(hit["lat"]),
        "longitude": float(hit["lon"]),
        "headline": f"Located at {float(hit['lat']):.2f}, {float(hit['lon']):.2f}",
    }


async def _quakes(lat, lon, loss_date, policy_start_date, radius_km, min_magnitude) -> dict:
    start, end = search_window(loss_date, policy_start_date)
    params = {
        "format": "geojson",
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "starttime": start.isoformat(),
        "endtime": (end + timedelta(days=1)).isoformat(),  # USGS end is exclusive
        "minmagnitude": min_magnitude,
        "orderby": "magnitude",
        "limit": 10,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params=params)
        r.raise_for_status()
        feats = r.json().get("features", [])
    events = []
    for f in feats:
        p = f.get("properties", {})
        ts = p.get("time")
        events.append(
            {
                "magnitude": p.get("mag"),
                "place": p.get("place"),
                "time_utc": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat() if ts else None,
                "url": p.get("url"),
            }
        )
    window = f"{start} to {end}"
    if events:
        top = events[0]
        headline = (
            f"{len(events)} quake(s) M{min_magnitude}+ within {radius_km} km ({window}); "
            f"largest M{top['magnitude']} on {(top['time_utc'] or '')[:10]}"
        )
    else:
        headline = f"No quakes M{min_magnitude}+ within {radius_km} km ({window})"
    return {"window": window, "earthquake_count": len(events), "events": events, "headline": headline}


def _series(daily: dict, key: str) -> list[tuple[str, float]]:
    times = daily.get("time", [])
    vals = daily.get(key, [])
    return [(t, v) for t, v in zip(times, vals) if v is not None]


async def _rain_flood(lat, lon, loss_date, policy_start_date) -> dict:
    start, end = search_window(loss_date, policy_start_date)
    recent = (date.today() - end).days < 7
    weather_url = (
        "https://api.open-meteo.com/v1/forecast" if recent else "https://archive-api.open-meteo.com/v1/archive"
    )
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": min(end, date.today()).isoformat(),
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    flood_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": (start - timedelta(days=45)).isoformat(),  # extra history = baseline flow
        "end_date": end.isoformat(),
        "daily": "river_discharge",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        w, f = await asyncio.gather(
            client.get(weather_url, params=weather_params),
            client.get("https://flood-api.open-meteo.com/v1/flood", params=flood_params),
            return_exceptions=True,
        )
    out: dict[str, Any] = {"window": f"{start} to {end}"}
    parts = []

    if not isinstance(w, Exception) and w.status_code == 200:
        rain = _series(w.json().get("daily", {}), "precipitation_sum")
        if rain:
            peak_day, peak = max(rain, key=lambda x: x[1])
            out["rainfall_mm"] = {
                "total": round(sum(v for _, v in rain), 1),
                "peak_day": peak_day,
                "peak_mm": round(peak, 1),
                "days_over_50mm": [d for d, v in rain if v >= 50],
            }
            parts.append(f"rain peak {peak:.0f} mm on {peak_day} (total {out['rainfall_mm']['total']} mm)")
    if not isinstance(f, Exception) and f.status_code == 200:
        flow = _series(f.json().get("daily", {}), "river_discharge")
        win = [(d, v) for d, v in flow if start.isoformat() <= d <= end.isoformat()]
        if flow and win:
            baseline = statistics.median(v for _, v in flow)
            peak_day, peak = max(win, key=lambda x: x[1])
            # A ratio against a near-zero baseline is meaningless (tiny/dry channel), so only
            # report it when the river is large enough for the model to be informative.
            meaningful = baseline >= MIN_MEANINGFUL_DISCHARGE
            ratio = round(peak / baseline, 2) if meaningful else None
            out["river_discharge_m3s"] = {
                "baseline_median": round(baseline, 2),
                "peak": round(peak, 2),
                "peak_day": peak_day,
                "peak_to_baseline_ratio": ratio,
                "note": None if meaningful else "Baseline flow is negligible; no meaningful flood signal from river data.",
            }
            if meaningful:
                parts.append(f"river flow peak {ratio}x baseline on {peak_day}")
            else:
                parts.append(f"river flow negligible (peak {peak:.1f} m3/s)")
    if not parts:
        out["error"] = "Both rainfall and river-flow APIs returned no usable data."
        out["headline"] = "Rainfall/river data unavailable"
    else:
        out["headline"] = "; ".join(parts)
    return out


async def _news(location, query, loss_date, policy_start_date) -> dict:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"error": "TAVILY_API_KEY is not set", "headline": "News search not configured"}
    start, end = search_window(loss_date, policy_start_date)
    client = TavilyClient(api_key=key)
    resp = await asyncio.to_thread(
        client.search,
        query=f"{location} {query}",
        topic="news",
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        max_results=5,
    )
    arts = [
        {
            "title": r.get("title"),
            "published_date": r.get("published_date"),
            "url": r.get("url"),
            "snippet": (r.get("content") or "")[:300],
        }
        for r in resp.get("results", [])
    ]
    window = f"{start} to {end}"
    headline = (
        f"{len(arts)} article(s) ({window}); top: {arts[0]['title']}" if arts else f"No news found ({window})"
    )
    return {"window": window, "articles_found": len(arts), "articles": arts, "headline": headline}


# ----------------------------------------------------------------------------
# tools exposed to the agent
# ----------------------------------------------------------------------------
@function_tool
async def record_plan(
    ctx: RunContextWrapper[RunCtx],
    hypotheses: List[str],
    checks_to_run: List[str],
    checks_skipped: List[str],
) -> str:
    """Record your investigation plan BEFORE using any other tool.

    Args:
        hypotheses: The competing explanations to test (e.g. genuine loss vs. post-disaster anti-selection vs. fabricated claim).
        checks_to_run: Which tools you will call and what each is meant to prove or disprove.
        checks_skipped: Tools you will NOT call and why (e.g. "earthquake check: claim is a flood").
    """
    return "Plan recorded. Proceed with the investigation."


@function_tool
async def self_check(
    ctx: RunContextWrapper[RunCtx],
    supports_claim: List[str],
    contradicts_claim: List[str],
    gaps: List[str],
    next_step: str,
) -> str:
    """After gathering evidence and BEFORE deciding, critique your own evidence.

    Args:
        supports_claim: Evidence that supports the claim being genuine.
        contradicts_claim: Evidence that contradicts it or points to fraud.
        gaps: What is still unknown or unverified (failed tools, missing data).
        next_step: Either 'decide' or the specific extra tool call you will make to close a gap.
    """
    return "Self-check logged. If you named an extra check, run it now; otherwise produce the final triage."


@function_tool
async def lookup_policy(ctx: RunContextWrapper[RunCtx], policy_id: str) -> str:
    """Fetch the policy: start date, perils covered, sum insured, deductible, waiting period, exclusions.

    Args:
        policy_id: The policy ID on the claim.
    """

    async def live(policy_id: str) -> dict:
        p = ctx.context.custom_policy or POLICIES.get(policy_id.strip().upper())
        if not p:
            return {"found": False, "error": f"Policy '{policy_id}' not found", "headline": f"Policy {policy_id} not found"}
        return {
            "found": True,
            **p,
            "headline": f"Policy active since {p['policy_start']}; covers {', '.join(p['perils_covered'])}; limit ${p['sum_insured_usd']:,.0f}",
        }

    return await _local(ctx, "lookup_policy", {"policy_id": policy_id}, live)


@function_tool
async def get_claim_history(ctx: RunContextWrapper[RunCtx], policy_id: str) -> str:
    """Fetch earlier claims on this policy (to spot repeat or pattern claims).

    Args:
        policy_id: The policy ID on the claim.
    """

    async def live(policy_id: str) -> dict:
        h = CLAIM_HISTORY.get(policy_id.strip().upper())
        if h is None and ctx.context.custom_policy:
            h = []  # a user-created policy starts with a clean history
        if h is None:
            return {"error": f"No history for '{policy_id}'", "headline": "No claim history record"}
        return {"prior_claims": h, "headline": f"{len(h)} prior claim(s) on this policy"}

    return await _local(ctx, "get_claim_history", {"policy_id": policy_id}, live)


async def _local(ctx, name, args, live_fn) -> str:
    """Mock/internal tools are deterministic: always run them, in every mode."""
    c = ctx.context
    try:
        out = await live_fn(**args)
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}", "headline": f"{name} failed"}
    out["data_mode"] = "calc" if name == "days_between" else "synthetic"
    c.tool_log.append({"name": name, "ok": "error" not in out})
    return json.dumps(out)


@function_tool
async def geocode_location(ctx: RunContextWrapper[RunCtx], location: str) -> str:
    """Convert a place name into latitude/longitude.

    Args:
        location: Place name or address from the claim.
    """
    return await _dispatch(ctx, "geocode_location", {"location": location}, _geocode)


@function_tool
async def check_earthquakes(
    ctx: RunContextWrapper[RunCtx],
    lat: float,
    lon: float,
    loss_date: str,
    policy_start_date: str,
    radius_km: int,
    min_magnitude: float,
) -> str:
    """Query the USGS catalogue for earthquakes near the location. The date window is chosen for you:
    it covers the 30 days before the loss date (and the weeks before the policy start if the policy is recent).

    Args:
        lat: Latitude from geocode_location.
        lon: Longitude from geocode_location.
        loss_date: Date of loss, YYYY-MM-DD.
        policy_start_date: Policy start date YYYY-MM-DD from lookup_policy (or "" if unknown).
        radius_km: Search radius in km (use 150-300).
        min_magnitude: Minimum magnitude (use 4.5 unless you have a reason).
    """
    args = {
        "lat": lat, "lon": lon, "loss_date": loss_date, "policy_start_date": policy_start_date,
        "radius_km": radius_km, "min_magnitude": min_magnitude,
    }
    return await _dispatch(ctx, "check_earthquakes", args, _quakes)


@function_tool
async def check_rain_and_flood(
    ctx: RunContextWrapper[RunCtx],
    lat: float,
    lon: float,
    loss_date: str,
    policy_start_date: str,
) -> str:
    """Get daily rainfall and river discharge around the loss date from Open-Meteo, summarised as
    peak rainfall and peak river flow versus its normal baseline. Window is chosen for you.

    Args:
        lat: Latitude from geocode_location.
        lon: Longitude from geocode_location.
        loss_date: Date of loss, YYYY-MM-DD.
        policy_start_date: Policy start date YYYY-MM-DD from lookup_policy (or "" if unknown).
    """
    args = {"lat": lat, "lon": lon, "loss_date": loss_date, "policy_start_date": policy_start_date}
    return await _dispatch(ctx, "check_rain_and_flood", args, _rain_flood)


@function_tool
async def search_disaster_news(
    ctx: RunContextWrapper[RunCtx],
    location: str,
    query: str,
    loss_date: str,
    policy_start_date: str,
) -> str:
    """Search news (Tavily) for disaster reports about the place, restricted to the relevant date window.

    Args:
        location: Place name.
        query: What to look for, e.g. 'flood damage' or 'earthquake damage'.
        loss_date: Date of loss, YYYY-MM-DD.
        policy_start_date: Policy start date YYYY-MM-DD from lookup_policy (or "" if unknown).
    """
    args = {"location": location, "query": query, "loss_date": loss_date, "policy_start_date": policy_start_date}
    return await _dispatch(ctx, "search_disaster_news", args, _news)


@function_tool
async def days_between(ctx: RunContextWrapper[RunCtx], from_date: str, to_date: str) -> str:
    """Exact number of days from one date to another (negative if to_date is earlier). Use this instead of
    doing date arithmetic yourself, e.g. days from disaster to policy start.

    Args:
        from_date: YYYY-MM-DD.
        to_date: YYYY-MM-DD.
    """

    async def live(from_date: str, to_date: str) -> dict:
        n = (_parse(to_date) - _parse(from_date)).days
        return {"days": n, "headline": f"{from_date} to {to_date} = {n} day(s)"}

    return await _local(ctx, "days_between", {"from_date": from_date, "to_date": to_date}, live)


ALL_TOOLS = [
    record_plan,
    self_check,
    lookup_policy,
    get_claim_history,
    geocode_location,
    check_earthquakes,
    check_rain_and_flood,
    search_disaster_news,
    days_between,
]
