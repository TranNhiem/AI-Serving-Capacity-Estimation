"""Acceptance tests for `ascep.conformance`, the module that grades a report against C1-C8.

Written against the specified contract before the implementation landed, so the tests are a
statement of what the checker must do rather than a description of what it happens to do.

The hardest thing to get right here is not detecting violations — it is *not* detecting ones
that are not there. A checker that cries wolf on an honest partial report trains people to
ignore it, which is worse than having no checker. `test_a_failing_gate_outside_the_envelope_is_
not_a_c7_error` is the guard for the one place that is easy to get wrong.
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest

from ascep.conformance import Finding, Verdict, check

ROOT = pathlib.Path(__file__).parent.parent
REPORT = ROOT / "examples" / "moe-26b-h100-tp2" / "report.json"


@pytest.fixture
def report() -> dict:
    """A fresh copy per test — every test here mutates it."""
    return json.loads(REPORT.read_text())


def _rules(findings, severity=None) -> set:
    return {f.rule for f in findings if severity is None or f.severity == severity}


# --- the published example ------------------------------------------------------------


def test_the_published_example_grades_partial(report):
    v = check(report)
    assert v.level == "partial"
    assert v.claimed == "partial"
    assert not v.overstated


def test_the_published_example_has_no_c1_to_c5_errors(report):
    """It is partial because of what it could not publish, not because it is malformed."""
    hard = {f for f in check(report).errors if f.rule in {"C1", "C2", "C3", "C4", "C5"}}
    assert not hard, [(f.rule, f.path, f.message) for f in hard]


def test_missing_reproduction_material_is_warned_not_failed(report):
    """C8 downgrades to partial; it does not invalidate a report that is otherwise sound."""
    v = check(report)
    assert "C8" in _rules(v.warnings)
    assert "C8" not in _rules(v.errors)


def test_a_failing_gate_outside_the_envelope_is_not_a_c7_error(report):
    """The 8,192-token point fails its TTFT gate, and the sustainable tier still equals the
    measured tier — correctly, because capacity was computed at a 2,000-token context and the
    failing point sits outside that envelope. A blanket "any failing row invalidates
    sustainable" rule would reject an honest report, so the rule is envelope-aware: an error
    only when the failure is at or below the context the capacity claim covers."""
    v = check(report)
    failing = [r for r in report["run"]["results"] if r.get("slo_pass") is False]
    assert failing, "fixture no longer exercises this path"
    tiers = report["capacity_tiers"]
    assert tiers["sustainable"]["max_concurrent_users"] == tiers["measured"]["max_concurrent_users"]
    assert "C7" not in _rules(v.errors)
    # but the reader is still told the envelope is bounded
    assert "C7" in _rules(v.warnings)


def test_a_failing_gate_inside_the_envelope_is_a_c7_error(report):
    """Move the failure to a context the capacity claim covers and it must fail."""
    report["run"]["results"][1]["slo_pass"] = False
    report["run"]["results"][1]["context_tokens"] = 1024
    assert "C7" in _rules(check(report).errors)


# --- the verdict object itself --------------------------------------------------------


def test_findings_are_stable_and_partitioned(report):
    v = check(report)
    assert isinstance(v, Verdict)
    assert all(isinstance(f, Finding) for f in v.findings)
    assert list(v.findings) == sorted(v.findings, key=lambda f: (f.rule, f.path))
    assert set(v.errors) | set(v.warnings) == set(v.findings)
    assert not set(v.errors) & set(v.warnings)


def test_every_finding_names_a_rule_a_path_and_an_action(report):
    for f in check(report).findings:
        assert f.rule in {f"C{i}" for i in range(1, 9)}
        assert f.severity in {"error", "warning"}
        assert f.path, f"{f.rule} finding with no path is not actionable"
        assert len(f.message.split()) >= 4, f"{f.rule}: message too terse to act on"


def test_overstating_conformance_is_detected(report):
    report["conformance"] = "conforming"
    v = check(report)
    assert v.claimed == "conforming"
    assert v.level == "partial"
    assert v.overstated


def test_understating_conformance_is_not_flagged(report):
    """Claiming less than you meet is honest, if pessimistic."""
    report["conformance"] = "non-conforming"
    assert not check(report).overstated


# --- C1: complete declaration ---------------------------------------------------------


def test_unjustified_null_is_a_c1_error(report):
    report["serving"]["batching_mode"] = None
    report["serving"].pop("batching_mode_u_reason", None)
    assert "C1" in _rules(check(report).errors)


def test_stale_justification_is_a_c1_error(report):
    """A reason left behind after the field was measured tells a reviewer to discount a
    number that is in fact solid."""
    report["serving"]["tensor_parallel_u_reason"] = "(U) not recorded"
    assert "C1" in _rules(check(report).errors)


def test_a_c1_error_forces_non_conforming(report):
    report["serving"]["batching_mode"] = None
    report["serving"].pop("batching_mode_u_reason", None)
    assert check(report).level == "non-conforming"


# --- C3: topology binding -------------------------------------------------------------


def test_topology_that_does_not_multiply_out_is_a_c3_error(report):
    report["serving"]["gpu_count"] = 3
    v = check(report)
    assert "C3" in _rules(v.errors)
    assert v.level == "non-conforming"


# --- C5: binding constraint -----------------------------------------------------------


def test_capacity_without_its_binding_constraint_is_a_c5_error(report):
    report["capacity_tiers"]["measured"]["binding_constraint"] = None
    assert "C5" in _rules(check(report).errors)


def test_sizing_result_without_its_binding_constraint_is_a_c5_error(report):
    report["sizing_result"]["binding_constraint"] = None
    assert "C5" in _rules(check(report).errors)


# --- C6: four tiers -------------------------------------------------------------------


def test_a_missing_tier_is_a_c6_error(report):
    del report["capacity_tiers"]["sustainable"]
    assert "C6" in _rules(check(report).errors)


def test_a_declined_tier_is_only_a_warning(report):
    """The example declines `theoretical` with a (U) reason; that is legitimate and better
    than fabricating a roofline nobody can defend."""
    v = check(report)
    assert report["capacity_tiers"]["theoretical"]["max_concurrent_users"] is None
    assert "C6" not in _rules(v.errors)
    assert "C6" in _rules(v.warnings)


def test_tiers_out_of_order_is_a_c6_error(report):
    report["capacity_tiers"]["recommended"]["max_concurrent_users"] = 10_000
    assert "C6" in _rules(check(report).errors)


def test_roofline_efficiency_at_or_above_one_is_a_c6_error(report):
    report["roofline_comparison"]["roofline_efficiency"] = 1.4
    assert "C6" in _rules(check(report).errors)


# --- C7: gates fixed before the run ---------------------------------------------------


def test_gates_chosen_after_the_run_is_a_c7_error(report):
    report["run"]["slo_gates"]["declared_before_run"] = False
    assert "C7" in _rules(check(report).errors)


# --- robustness -----------------------------------------------------------------------


def test_check_does_not_mutate_its_input(report):
    before = copy.deepcopy(report)
    check(report)
    assert report == before


def test_a_wildly_incomplete_report_does_not_crash():
    v = check({"ascep_version": "0.1.0"})
    assert v.level == "non-conforming"
    assert v.findings
