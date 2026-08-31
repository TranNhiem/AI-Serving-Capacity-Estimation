"""Every (I) figure in a published example must recompute from its (M) figures.

This is conformance rule C2 enforced as a test. If someone edits an example's
headline number without editing the measurement it came from, this fails.
"""

import json
import pathlib

import pytest

from ascep.capacity import (
    Tier,
    Workload,
    capacity_at,
    interpolate_throughput,
    scaling_efficiency,
)

EXAMPLES = sorted((pathlib.Path(__file__).parent.parent / "examples").glob("*/run-summary.json"))


@pytest.fixture(params=EXAMPLES, ids=lambda p: p.parent.name)
def example(request):
    return json.loads(request.param.read_text())


def _workload(w):
    return Workload(
        daily_active_users=0,
        concurrent_users=w["concurrent_users"],
        duty_cycle=w["duty_cycle"],
        input_tokens_per_request=w["input_tokens_per_request"],
        output_tokens_per_request=w["output_tokens_per_request"],
        target_tok_s_per_user=w["target_tok_s_per_user"],
    )


def test_declared_workload_derivations(example):
    w = example["workload"]
    work = _workload(w)
    assert work.avg_context_tokens() == pytest.approx(w["avg_context_tokens"])
    assert work.demand_tok_s() == pytest.approx(w["demand_tok_s"])


def test_capacity_result_recomputes(example):
    c = example["capacity_result"]
    work = _workload(example["workload"])
    cap = capacity_at(
        n_gpus=example["serving"]["tensor_parallel"],
        kv_tokens=c["kv_tokens"],
        throughput_tok_s=c["throughput_tok_s"],
        workload=work,
        tier=Tier(c["tier"]),
        headroom=c["headroom"],
    )
    assert round(cap.max_concurrent_users) == c["max_concurrent_users"]
    assert cap.binding_constraint.value == c["binding_constraint"]
    assert round(cap.detail["users_kv_floor"]) == c["kv_floor_users"]
    assert round(cap.detail["users_throughput_floor"]) == c["throughput_floor_users"]


def test_capacity_is_the_minimum_floor(example):
    """C5: the reported capacity must equal the floor it names, not some other floor."""
    c = example["capacity_result"]
    floors = {"kv": c["kv_floor_users"], "throughput": c["throughput_floor_users"]}
    assert c["max_concurrent_users"] == min(floors.values())
    assert floors[c["binding_constraint"]] == c["max_concurrent_users"]


def test_per_gpu_kv_matches_engine_report(example):
    for row in example["measurements"]["kv_by_topology"]:
        assert round(row["engine_reported_kv_tokens"] / row["n_gpus"]) == row["kv_tokens_per_gpu"]


def test_scaling_efficiency_recomputes_and_superlinear_is_flagged(example):
    s = example["measurements"].get("scaling")
    if not s:
        pytest.skip("no scaling sweep in this example")
    eff = scaling_efficiency(
        s["baseline"]["tok_s"], s["scaled"]["tok_s"], s["baseline"]["n_gpus"], s["scaled"]["n_gpus"]
    )
    assert round(eff, 2) == s["efficiency"]
    if eff > 1.0:
        # A report may not quietly present superlinear scaling as a result.
        assert s.get("efficiency_verdict"), "superlinear scaling must carry an explicit verdict"
        assert "INVALID" in s["efficiency_verdict"].upper()


def test_throughput_curve_is_monotonic_in_context(example):
    """Throughput must fall as context grows; a rise means the points aren't comparable."""
    pts = [
        (p["context_tokens"], p["tok_s_per_gpu"])
        for p in example["measurements"]["throughput_vs_context_tp2"]["points"]
    ]
    assert pts == sorted(pts), "curve points must be ordered by context length"
    tputs = [t for _, t in pts]
    assert tputs == sorted(tputs, reverse=True), f"throughput rose with context: {pts}"


def test_interpolation_never_extrapolates(example):
    pts = [
        (p["context_tokens"], p["tok_s_per_gpu"])
        for p in example["measurements"]["throughput_vs_context_tp2"]["points"]
    ]
    lo, hi = pts[0], pts[-1]
    assert interpolate_throughput(pts, lo[0] // 2) == pytest.approx(lo[1])
    assert interpolate_throughput(pts, hi[0] * 4) == pytest.approx(hi[1])


def test_slo_failures_are_excluded_from_sustainable(example):
    """C7/C6: an operating point that failed its gate must not be sold as sustainable."""
    gates = example["measurements"]["latency_gates"]
    assert gates["gates_declared_before_run"] is True
    limit = gates["ttft_p95_s_max"]
    for r in gates["results"]:
        assert r["slo_pass"] == (r["ttft_p95_s"] <= limit)


def test_unmeasured_fields_are_null_with_a_justification(example):
    """C1: unknown values are recorded as null and registered in `unmeasured` — never
    omitted, never guessed. Every entry must state the impact of being wrong, because an
    unmeasured value whose impact nobody assessed is indistinguishable from a guess."""
    registered = {u["field"]: u for u in example.get("unmeasured", [])}
    nulls = {
        f"{section}.{key}"
        for section in ("hardware", "model", "serving")
        for key, value in example[section].items()
        if value is None
    }
    missing = nulls - set(registered)
    assert not missing, f"null fields absent from `unmeasured`: {sorted(missing)}"

    stale = set(registered) - nulls
    assert not stale, f"`unmeasured` names fields that are not null: {sorted(stale)}"

    for field, entry in registered.items():
        assert "(U)" in entry["why_missing"], f"{field}: why_missing must carry the (U) tag"
        assert entry["impact_if_wrong"].strip(), f"{field}: impact_if_wrong must not be empty"
        assert entry["cost_to_measure"].strip(), f"{field}: cost_to_measure must not be empty"


def test_partial_reports_say_why_they_are_partial(example):
    """A report may be partial, but it may not be quietly partial."""
    assert example["conformance"] in {"conforming", "partial", "non-conforming"}
    assert example.get("conformance_note", "").strip(), (
        "every report must say which conformance rules it meets and which it does not — a "
        "level claimed without a reason cannot be reviewed"
    )


def test_conclusion_sensitivity_is_stated(example):
    """SPEC C1/§7: a report whose conclusion flips on a (U) must say so in those words."""
    if example.get("unmeasured"):
        assert example.get("conclusion_sensitivity", "").strip(), (
            "a report with unmeasured assumptions must state which one the conclusion "
            "is most sensitive to"
        )
