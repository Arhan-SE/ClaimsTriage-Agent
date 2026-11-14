import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import TriageResult, apply_guardrails  # noqa: E402
from app.tools import search_window  # noqa: E402


def result(**over):
    base = dict(
        severity="MINOR", severity_reason="r", coverage="COVERED", coverage_reason="r",
        fraud_risk="LOW", fraud_reason="r", confidence=95, route="AUTO_APPROVE", route_reason="r",
        summary="s", timeline=[], evidence=[], limitations=[],
    )
    base.update(over)
    return TriageResult(**base)


OK_LOG = [{"name": "lookup_policy", "ok": True}, {"name": "check_earthquakes", "ok": True}]


def test_clean_auto_approve_is_untouched():
    r, changes = apply_guardrails(result(), OK_LOG)
    assert r.route == "AUTO_APPROVE" and changes == []


def test_failed_policy_lookup_forces_human_review():
    log = [{"name": "lookup_policy", "ok": False}, {"name": "check_earthquakes", "ok": True}]
    r, changes = apply_guardrails(result(), log)
    assert r.route == "HUMAN_REVIEW" and r.coverage == "NEEDS_REVIEW" and changes


def test_no_hazard_evidence_forces_human_review_and_caps_confidence():
    log = [{"name": "lookup_policy", "ok": True}, {"name": "check_earthquakes", "ok": False}]
    r, _ = apply_guardrails(result(), log)
    assert r.route == "HUMAN_REVIEW" and r.confidence <= 40


def test_auto_approve_downgraded_when_risky():
    for over in ({"fraud_risk": "MEDIUM"}, {"coverage": "NEEDS_REVIEW"}, {"severity": "MAJOR"}, {"confidence": 70}):
        r, _ = apply_guardrails(result(**over), OK_LOG)
        assert r.route == "ADJUSTER_REVIEW", over


def test_fraud_unit_with_low_risk_is_incoherent():
    r, _ = apply_guardrails(result(route="FRAUD_UNIT", fraud_risk="LOW"), OK_LOG)
    assert r.route == "HUMAN_REVIEW"


def test_confidence_is_clamped():
    r, _ = apply_guardrails(result(route="ADJUSTER_REVIEW", confidence=250), OK_LOG)
    assert r.confidence == 100


def test_search_window_covers_recent_policy_start():
    start, end = search_window("2023-02-08", "2023-02-07")
    assert str(end) == "2023-02-10" and str(start) <= "2023-01-09"


def test_search_window_old_policy_uses_30_days():
    start, _ = search_window("2023-02-06", "2022-06-01")
    assert str(start) == "2023-01-07"


def test_search_window_bad_policy_date_is_safe():
    start, _ = search_window("2023-02-06", "")
    assert str(start) == "2023-01-07"
