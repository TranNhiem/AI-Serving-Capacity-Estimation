"""Layer 5 end to end: a product forecast sized against a real measurement.

`examples/chatbot-10k-dau` answers "10,000 daily users — how many GPUs?" by composing a
workload declaration with per-GPU figures measured in a *different* example. Two things can
rot here and both are silent:

  * the derived (I) fields in `workload.json` drifting away from `ascep.capacity.Workload`;
  * the prose in that example's README quoting numbers the formulas no longer produce.

So this module recomputes the whole chain from the two source artifacts and then asserts the
README literally contains the figures it claims. A worked example whose narrative and
arithmetic disagree is worse than no example — a reader has no way to tell which half is wrong.
"""

import importlib.util
import json
import pathlib
from dataclasses import replace

import pytest

from ascep.capacity import Constraint, Workload, capacity_at, gpus_required, interpolate_throughput
from ascep.validation import validate

ROOT = pathlib.Path(__file__).parent.parent
EXAMPLE = ROOT / "examples" / "chatbot-10k-dau"
MEASURED = ROOT / "examples" / "moe-26b-h100-tp2" / "run-summary.json"

HEADROOM = 1.15
GPUS_PER_REPLICA = 2


@pytest.fixture(scope="module")
def declared() -> dict:
    return json.loads((EXAMPLE / "workload.json").read_text())


@pytest.fixture(scope="module")
def work(declared) -> Workload:
    """The declaration, read back as the dataclass that produced it.

    Deliberately built from the published JSON rather than imported from the builder: this is
    the round trip a third party performs, and it is where a schema/code mismatch would show.
    """
    return Workload(
        daily_active_users=declared["daily_active_users"],
        sessions_per_user_per_day=declared["sessions_per_user_per_day"],
        avg_session_seconds=declared["avg_session_seconds"],
        peak_to_mean=declared["peak_to_mean"],
        duty_cycle=declared["duty_cycle"],
        input_tokens_per_request=declared["input_tokens_per_request"],
        output_tokens_per_request=declared["output_tokens_per_request"],
        requests_per_session=declared["requests_per_session"],
        target_tok_s_per_user=declared["target_tok_s_per_user"] or 0.0,
    )


@pytest.fixture(scope="module")
def measured() -> dict:
    """Per-GPU figures lifted from the example that actually measured them."""
    run = json.loads(MEASURED.read_text())
    kv = next(r for r in run["measurements"]["kv_by_topology"] if r["tensor_parallel"] == 2)
    curve = [
        (p["context_tokens"], p["tok_s_per_gpu"])
        for p in run["measurements"]["throughput_vs_context_tp2"]["points"]
    ]
    return {"kv_tokens_per_gpu": kv["kv_tokens_per_gpu"], "curve": curve}


def _size(w: Workload, measured: dict):
    """The sizing chain exactly as the example's README describes it."""
    thr = interpolate_throughput(measured["curve"], w.avg_context_tokens())
    need = gpus_required(
        w,
        kv_tokens_per_gpu=measured["kv_tokens_per_gpu"],
        throughput_tok_s_per_gpu=thr,
        headroom=HEADROOM,
        gpus_per_replica=GPUS_PER_REPLICA,
    )
    cap = capacity_at(
        n_gpus=need.n_gpus,
        kv_tokens=measured["kv_tokens_per_gpu"] * need.n_gpus,
        throughput_tok_s=thr * need.n_gpus,
        workload=w,
        headroom=HEADROOM,
    )
    return thr, need, cap


# --- the declaration itself -----------------------------------------------------------


def test_workload_validates(declared):
    assert validate("workload", declared) == []


def test_workload_is_regenerable():
    """The committed artifact must equal what the builder emits, or it was hand-edited."""
    spec = importlib.util.spec_from_file_location("_bw", EXAMPLE / "build_workload.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build() == json.loads((EXAMPLE / "workload.json").read_text())


def test_derived_fields_recompute(declared, work):
    """C2: every (I) figure must fall out of the (U) forecast by a named formula."""
    assert declared["peak_concurrent_users"] == pytest.approx(
        work.peak_concurrent_users(), abs=0.01
    )
    assert declared["active_sessions"] == pytest.approx(work.active_sessions(), abs=0.01)
    assert declared["avg_context_tokens"] == pytest.approx(work.avg_context_tokens(), abs=0.01)
    assert declared["demand_tok_s"] == pytest.approx(work.demand_tok_s(), abs=0.01)


def test_active_sessions_is_not_confused_with_concurrency(declared):
    """Chapter 6's MUST: the two differ by exactly the duty cycle, and both must be stated."""
    assert declared["active_sessions"] == pytest.approx(
        declared["peak_concurrent_users"] * declared["duty_cycle"], abs=0.01
    )
    assert declared["active_sessions"] != declared["peak_concurrent_users"]


def test_every_null_is_justified(declared):
    """C1: a null is a recorded unknown, not an omission."""
    for key, value in declared.items():
        if value is None:
            reason = declared.get(f"{key}_u_reason", "")
            assert reason.strip(), f"{key} is null with no {key}_u_reason"
            assert reason.startswith("(U)"), f"{key}_u_reason must carry the (U) tag"


# --- the sizing chain -----------------------------------------------------------------


def test_headline_answer_is_two_gpus_bound_by_throughput(work, measured):
    thr, need, cap = _size(work, measured)
    assert round(thr) == 1_459
    assert need.n_gpus == 2
    assert need.binding_constraint is Constraint.THROUGHPUT
    assert round(cap.detail["users_kv_floor"]) == 2_083
    assert round(cap.detail["users_throughput_floor"]) == 761


def test_the_binding_floor_is_the_one_that_is_short(work, measured):
    """C5 as arithmetic: the named constraint must be the minimum, and it must clear demand."""
    _, need, cap = _size(work, measured)
    floors = {
        Constraint.KV: cap.detail["users_kv_floor"],
        Constraint.THROUGHPUT: cap.detail["users_throughput_floor"],
    }
    assert floors[need.binding_constraint] == pytest.approx(min(floors.values()))
    assert min(floors.values()) >= work.peak_concurrent_users()


def test_illustrative_figures_from_chapter_6_reproduce(work):
    """Chapter 6's worked example (a) answers 4 GPUs on placeholder numbers; the README's
    argument for measuring rather than estimating rests on that gap being real."""
    need = gpus_required(
        work,
        kv_tokens_per_gpu=180_000,
        throughput_tok_s_per_gpu=900,
        headroom=HEADROOM,
        gpus_per_replica=GPUS_PER_REPLICA,
    )
    assert need.n_gpus == 4
    assert need.binding_constraint is Constraint.THROUGHPUT


# --- the sensitivities the README claims ----------------------------------------------


@pytest.mark.parametrize(
    "name,change,gpus,binding",
    [
        ("turns per session double", dict(requests_per_session=10), 4, Constraint.THROUGHPUT),
        ("document-length inputs", dict(input_tokens_per_request=8_000), 4, Constraint.KV),
        ("duty cycle to 1.0 changes nothing", dict(duty_cycle=1.0), 2, Constraint.THROUGHPUT),
        ("sharper peak", dict(peak_to_mean=6.0), 4, Constraint.THROUGHPUT),
        ("sizing to the mean", dict(peak_to_mean=1.0), 2, Constraint.THROUGHPUT),
    ],
)
def test_stated_sensitivities_hold(work, measured, name, change, gpus, binding):
    _, need, _ = _size(replace(work, **change), measured)
    assert (need.n_gpus, need.binding_constraint) == (gpus, binding), name


def test_long_context_flips_the_binding_constraint(work, measured):
    """The crossover, which is the whole reason C5 is normative: at 8k input the KV floor
    drops below the throughput floor and the correct purchase changes character."""
    short = _size(work, measured)[2]
    long = _size(replace(work, input_tokens_per_request=8_000), measured)[2]
    assert short.binding_constraint is Constraint.THROUGHPUT
    assert long.binding_constraint is Constraint.KV
    assert short.detail["users_kv_floor"] > short.detail["users_throughput_floor"]
    assert long.detail["users_kv_floor"] < long.detail["users_throughput_floor"]


# --- prose must agree with arithmetic -------------------------------------------------


def test_readme_quotes_the_numbers_the_formulas_produce(work, measured):
    """The narrative is part of the deliverable. If a figure in the prose stops being what the
    code computes, the example teaches the wrong thing and nothing else would catch it."""
    readme = (EXAMPLE / "README.md").read_text()
    thr, _, cap = _size(work, measured)
    long_cap = _size(replace(work, input_tokens_per_request=8_000), measured)[2]
    flat = replace(work, peak_to_mean=1.0)

    expected = {
        "interpolated throughput": f"{thr:,.0f}",
        "kv floor": f"{cap.detail['users_kv_floor']:,.0f}",
        "throughput floor": f"{cap.detail['users_throughput_floor']:,.0f}",
        "measured kv per gpu": f"{measured['kv_tokens_per_gpu']:,.0f}",
        "long-context kv floor": f"{long_cap.detail['users_kv_floor']:,.0f}",
        "long-context throughput floor": f"{long_cap.detail['users_throughput_floor']:,.0f}",
        "doubled demand": f"{replace(work, requests_per_session=10).demand_tok_s():,.0f}",
        "flat-load concurrency": f"{flat.peak_concurrent_users():,.0f}",
        "flat-load demand": f"{flat.demand_tok_s():,.0f}",
    }
    missing = {k: v for k, v in expected.items() if v not in readme}
    assert not missing, f"README no longer states these computed figures: {missing}"


# --- volume conversions ----------------------------------------------------------------


def test_monthly_requests_is_daily_times_a_day_count_the_caller_chooses(work, measured):
    """Monthly volume is a presentation unit, not a second measurement.

    The method exists because commercial conversations happen in months, and the day count is a
    parameter because "a month" is not a quantity. What must never happen is monthly acquiring
    its own reported field: it would be an (I) derived from an (I), and the two would drift.
    """
    cap = _size(work, measured)[2]
    assert cap.monthly_requests() == pytest.approx(cap.daily_requests() * 30)
    assert cap.monthly_requests(days=31) == pytest.approx(cap.daily_requests() * 31)
    # The spread the docstring warns about: 28 vs 31 days is 10.7% of the answer, several times
    # the 15% headroom factor these deployments are sized on.
    february = cap.monthly_requests(days=28)
    january = cap.monthly_requests(days=31)
    assert (january - february) / february > 0.10


def test_monthly_requests_is_not_a_reportable_field():
    """Guards the design decision, not the arithmetic.

    If someone later adds `monthly_requests` to the report schema, C2 has to tag a number
    derived from another (I) number, and there are then two places for one figure to be wrong.
    The fix, if monthly is ever genuinely needed in a report, is to state the day count next to
    `daily_requests` -- not to add a field.
    """
    schema = json.loads((ROOT / "schemas" / "capacity-report.schema.json").read_text())
    assert "monthly_requests" not in json.dumps(schema)
