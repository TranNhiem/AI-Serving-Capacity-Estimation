"""The load driver, tested as two separable things: boundary arithmetic and orchestration.

Chapter 7 section 6 is the part that has to be exactly right, and none of it needs a server:
which requests are offered demand, which are latency samples, and which the drain deadline
has turned into non-completions is pure arithmetic over timestamps. Those rules are tested
against synthetic records so a failure names the rule rather than the network.

The asynchronous half is tested for structure, not for timing: that warm-up is marked and
kept, that nothing is issued after close, that the deadline is enforced, that every record
carries the operating point it was produced at. Timing assertions here use millisecond
workloads and loose tolerances on purpose -- a driver test that fails on a busy CI runner
teaches contributors to re-run until green, which is worse than not testing it.
"""

import asyncio

import pytest

from ascep.bench import Outcome, RequestRecord
from ascep.bench.adapters.base import Adapter, RequestSpec
from ascep.bench.driver import (
    WindowPolicy,
    apply_boundary_rules,
    no_reset,
    run_window,
)
from ascep.bench.metrics import reduce_window


def _rec(rid, issued, e2e=None, outcome=Outcome.OK, tokens=10):
    """A record that ran ``e2e`` seconds from issue; ``e2e=None`` means it never finished."""
    end = None if e2e is None else issued + e2e
    return RequestRecord(
        request_id=rid,
        issued_ts=issued,
        outcome=outcome,
        first_token_ts=None if end is None else issued + (e2e / 2),
        end_ts=end,
        output_tokens=None if end is None else tokens,
    )


# --- §7.6 boundary arithmetic ---------------------------------------------------------


def test_an_arrival_before_the_window_opened_is_not_offered_demand():
    """Warm-up traffic is the usual source, and it must not be counted as load offered.

    Crediting it inflates both the user count and the speed, which is the pair of errors
    section 7.6 calls out by name.
    """
    early = _rec("w0", issued=-1.0, e2e=0.5)
    inside = _rec("r0", issued=1.0, e2e=0.5)

    apply_boundary_rules([early, inside], t0=0.0, window_s=10.0, drain_deadline_s=2.0)

    assert early.in_window is False
    assert inside.in_window is True


def test_an_arrival_inside_the_window_stays_offered_demand_however_late_it_finishes():
    """Offered demand is decided at issue time and never revised by the outcome.

    A denominator that shrinks when requests go badly is the one shape of error rate that
    can never report a bad number.
    """
    slow = _rec("r0", issued=9.9, e2e=1.5)

    apply_boundary_rules([slow], t0=0.0, window_s=10.0, drain_deadline_s=5.0)

    assert slow.in_window is True


def test_a_straddler_completing_within_the_drain_deadline_is_a_valid_latency_sample():
    straddler = _rec("r0", issued=9.5, e2e=2.0)  # finishes at 11.5, deadline is 12.0

    apply_boundary_rules([straddler], t0=0.0, window_s=10.0, drain_deadline_s=2.0)

    assert straddler.outcome is Outcome.OK
    assert straddler.is_failure is False


def test_a_request_outstanding_at_the_drain_deadline_is_an_error_with_no_latency_sample():
    """The deadline exists to stop one request being both a tail sample and a failure.

    Without it a harness resolves the contradiction whichever way flatters the run, so the
    resolution is written down: past the deadline it is a non-completion, and the reason
    says which deadline, because "0.2% errors" at 5 s and at 120 s are different claims.
    """
    late = _rec("r0", issued=9.5, e2e=8.0)  # finishes at 17.5, deadline is 12.0

    apply_boundary_rules([late], t0=0.0, window_s=10.0, drain_deadline_s=2.0)

    assert late.outcome is Outcome.CANCELLED
    assert late.is_failure is True
    assert "12" in (late.error or ""), "the reason must name the deadline it missed"


def test_a_request_with_no_end_timestamp_is_outstanding_not_missing():
    """A record with no completion is the normal shape of a request the driver cancelled.

    Left as OK it would be a completion with an unmeasurable latency; dropped, it would
    leave the error-rate denominator. Both readings are more flattering than the truth.
    """
    hung = _rec("r0", issued=5.0, e2e=None)

    apply_boundary_rules([hung], t0=0.0, window_s=10.0, drain_deadline_s=2.0)

    assert hung.outcome is Outcome.CANCELLED
    assert hung.is_failure is True


def test_the_drain_deadline_is_absolute_from_window_close_not_per_request():
    """Two requests of identical duration, one issued early and one late; only one survives.

    A per-request grace period would extend the window by the service time of whatever was
    running when it closed, which is longest exactly when the server is worst.
    """
    early = _rec("early", issued=1.0, e2e=8.0)  # ends at 9.0, inside the window
    late = _rec("late", issued=9.5, e2e=8.0)  # ends at 17.5, past the 12.0 deadline

    apply_boundary_rules([early, late], t0=0.0, window_s=10.0, drain_deadline_s=2.0)

    assert early.outcome is Outcome.OK
    assert late.outcome is Outcome.CANCELLED


def test_a_zero_drain_deadline_admits_no_straddler_at_all():
    """Zero is a legal declaration and it means what it says: finish by close or fail."""
    straddler = _rec("r0", issued=9.9, e2e=0.2)  # ends at 10.1

    apply_boundary_rules([straddler], t0=0.0, window_s=10.0, drain_deadline_s=0.0)

    assert straddler.outcome is Outcome.CANCELLED


def test_a_request_that_already_failed_keeps_the_cause_it_failed_for():
    """Refusal and cancellation are both failures and the operator does different things.

    Overwriting a 429 with "outstanding at the drain deadline" hides an admission queue
    that was full, which is the finding, not the accounting.
    """
    refused = RequestRecord(request_id="r0", issued_ts=9.9, outcome=Outcome.REFUSED)

    apply_boundary_rules([refused], t0=0.0, window_s=10.0, drain_deadline_s=2.0)

    assert refused.outcome is Outcome.REFUSED


def test_the_boundary_counts_report_each_cohort_separately():
    """The three cohorts differ only at the edges, so the report has to show the edges."""
    records = [
        _rec("w0", issued=-2.0, e2e=0.5),  # warm-up, finished before the window
        _rec("w1", issued=-0.5, e2e=1.0),  # warm-up, finished inside the window
        _rec("r0", issued=1.0, e2e=1.0),  # wholly inside
        _rec("r1", issued=9.5, e2e=1.0),  # straddles close, inside the deadline
        _rec("r2", issued=9.5, e2e=9.0),  # straddles close, past the deadline
    ]

    b = apply_boundary_rules(records, t0=0.0, window_s=10.0, drain_deadline_s=2.0)

    assert b.offered == 3, "only arrivals inside the window are demand"
    assert b.warmup == 2
    assert b.completed_in_window == 2, "r0 and the warm-up request that landed inside"
    assert b.straddlers == 1, "r1 completed after close but within the deadline"
    assert b.abandoned == 1, "r2 did not"


def test_the_boundary_rules_and_the_reduction_agree_on_the_same_records():
    """The driver and the reduction must read one set of records the same way.

    They are separate modules with separate tests, which is exactly how two consistent
    halves come to disagree: the driver marking by one rule and the reduction counting by
    another produces a report where no set of requests explains all its own figures.
    """
    records = [
        _rec("w0", issued=-0.5, e2e=1.0, tokens=10),  # warm-up finishing inside
        *[_rec(f"r{i}", issued=1.0 + i * 0.1, e2e=1.0, tokens=10) for i in range(20)],
        _rec("late", issued=9.9, e2e=9.0, tokens=10),  # past the deadline
    ]

    b = apply_boundary_rules(records, t0=0.0, window_s=10.0, drain_deadline_s=2.0)
    s = reduce_window(records, window_s=10.0, t0=0.0)

    assert s.n_issued == b.offered == 21
    assert s.excluded_warmup_count == b.warmup == 1
    assert s.n_completed == b.completed_in_window == 21
    assert s.n_latency_samples == 20, "the abandoned request contributes no latency"
    assert s.excluded_error_count == b.abandoned == 1
    assert s.requests_per_s == pytest.approx(21 / 10.0)


# --- the policy is a declaration, so an incomplete one must not build -----------------


def test_a_policy_cannot_be_built_without_declaring_a_drain_deadline():
    """Section 7.6 requires it before timing, which means before the run object exists.

    A default would be a number nobody chose deciding whether a straddler counted.
    """
    with pytest.raises(TypeError):
        WindowPolicy(concurrency=8, window_s=60.0, think_time_s=0.0, warmup_requests=8)


def test_a_policy_cannot_be_built_without_declaring_think_time():
    """Section 7.2 makes think time a required declaration for a closed loop.

    Zero is a legitimate answer and the common one, which is exactly why it must not be the
    default: a caller who declared 1.2 s in the workload and forgot to carry it into the
    policy would run a saturation test the report describes as interactive.
    """
    with pytest.raises(TypeError):
        WindowPolicy(concurrency=8, window_s=60.0, drain_deadline_s=5.0, warmup_requests=8)


def test_a_policy_without_warm_up_is_rejected():
    """Section 7.3: every repetition includes its own warm-up. Zero of both is not one."""
    with pytest.raises(ValueError, match="warm-up"):
        WindowPolicy(concurrency=8, window_s=60.0, drain_deadline_s=5.0, think_time_s=0.0)


@pytest.mark.parametrize(
    "kw",
    [
        dict(concurrency=0),
        dict(window_s=0.0),
        dict(drain_deadline_s=-1.0),
        dict(think_time_s=-0.1),
    ],
)
def test_a_policy_with_an_impossible_operating_point_is_rejected(kw):
    base = dict(
        concurrency=8, window_s=60.0, drain_deadline_s=5.0, think_time_s=0.0, warmup_requests=8
    )
    with pytest.raises(ValueError):
        WindowPolicy(**{**base, **kw})


def test_an_open_loop_policy_says_it_is_not_implemented_rather_than_running_closed():
    """Section 4.4 makes the loop type a declaration, and the two measure different things.

    Silently running a closed loop for an open-loop declaration produces a report whose
    arrival process is not the one it claims, and nothing downstream can detect it.
    """
    with pytest.raises(ValueError, match="open"):
        WindowPolicy(
            concurrency=8,
            window_s=60.0,
            drain_deadline_s=5.0,
            think_time_s=0.0,
            warmup_requests=8,
            loop="open",
        )


# --- orchestration --------------------------------------------------------------------


class _FakeAdapter(Adapter):
    """Serves every request in ``service_s`` seconds of real time, recording honestly."""

    def __init__(self, service_s=0.01, slow_after=None, slow_s=1.0):
        self.service_s = service_s
        self.slow_after = slow_after
        self.slow_s = slow_s
        self.issued = []

    @property
    def name(self):
        return "fake"

    async def issue(self, spec, *, clock, sink=None):
        self.issued.append(spec.request_id)
        n = len(self.issued)
        duration = self.slow_s if self.slow_after and n > self.slow_after else self.service_s
        record = RequestRecord(request_id=spec.request_id, issued_ts=clock())
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            record.outcome = Outcome.CANCELLED
            record.error = "cancelled at window close"
            if sink is not None:
                sink(record)
            raise
        record.first_token_ts = clock()
        record.end_ts = clock()
        record.output_tokens = 10
        return record


def _policy(**kw):
    base = dict(
        concurrency=4,
        window_s=0.20,
        drain_deadline_s=0.20,
        think_time_s=0.0,
        warmup_requests=4,
        repetition=1,
    )
    return WindowPolicy(**{**base, **kw})


def _specs():
    return lambda i: RequestSpec(request_id=f"q{i}", messages=[{"role": "user", "content": "x"}])


def test_warm_up_records_are_kept_in_the_bundle_and_marked_out_of_window():
    """Section 7.3 discards warm-up from the statistics, not from the evidence.

    A bundle missing its warm-up cannot be re-analysed by a reviewer who disagrees with
    where the harness drew the line, which is the whole point of shipping records.
    """
    run = asyncio.run(
        run_window(
            _FakeAdapter(service_s=0.005),
            _specs(),
            policy=_policy(warmup_requests=8),
            reset=no_reset,
        )
    )

    warm = [r for r in run.records if not r.in_window]
    assert len(warm) >= 8, "every warm-up request is still in the bundle"
    assert run.warmup_count >= 8
    assert all(r.issued_ts < run.t0 for r in warm)


def test_nothing_is_issued_after_the_window_closes():
    """A request offered after close is demand the window did not have.

    Counting it makes the denominator depend on how long the last request took to come
    back, which is longest exactly where the error rate matters most.
    """
    run = asyncio.run(
        run_window(_FakeAdapter(service_s=0.005), _specs(), policy=_policy(), reset=no_reset)
    )

    # Asserting it over in-window records only would be circular: in_window IS the test
    # `issued_ts <= t1`, so a driver that kept issuing through the whole drain would file
    # those requests as warm-up and pass. The claim is about every record in the bundle.
    t1 = run.t0 + run.policy.window_s
    last_issue = max(r.issued_ts for r in run.records)
    assert last_issue <= t1 + 0.05, "a request was issued after the window closed"


def test_every_record_carries_the_operating_point_it_was_produced_at():
    """Records outlive the run object; a rung is unreadable if the record does not say so."""
    run = asyncio.run(
        run_window(
            _FakeAdapter(service_s=0.005),
            _specs(),
            policy=_policy(concurrency=4, repetition=2),
            reset=no_reset,
        )
    )

    assert run.records, "the run produced no records at all"
    assert all(r.concurrency == 4 for r in run.records)
    assert all(r.repetition == 2 for r in run.records)


def test_the_run_reports_the_declared_window_not_the_span_its_records_imply():
    """Section 4.2: the rate denominator is the declared duration.

    A run object that reported first-arrival-to-last-completion would hand the reduction a
    denominator that stretches with stragglers, quietly turning a slow window into a
    slightly longer one instead of a slower one.
    """
    policy = _policy()
    run = asyncio.run(
        run_window(_FakeAdapter(service_s=0.005), _specs(), policy=policy, reset=no_reset)
    )

    assert run.window_s == policy.window_s
    span = max(r.end_ts or r.issued_ts for r in run.records) - min(r.issued_ts for r in run.records)
    assert span > policy.window_s, "the fixture must actually have warm-up outside the window"


def test_a_request_still_running_at_the_drain_deadline_is_cancelled_and_recorded():
    """Not dropped: an abandoned request is a failure the user experienced.

    A driver that walks away from its own in-flight requests reports the error rate of the
    requests that came back, which under overload is close to zero by construction.
    """
    adapter = _FakeAdapter(service_s=0.005, slow_after=6, slow_s=30.0)
    run = asyncio.run(
        run_window(adapter, _specs(), policy=_policy(drain_deadline_s=0.05), reset=no_reset)
    )

    cancelled = [r for r in run.records if r.outcome is Outcome.CANCELLED]
    assert cancelled, "the requests that never came back must still be in the bundle"
    assert run.abandoned == len(cancelled)
    assert all(r.in_window for r in cancelled), "they were offered inside the window"


def test_the_declared_initial_state_reset_runs_before_warm_up():
    """Section 7.5: a repetition resets state and then warms up, in that order.

    Warming up first and resetting afterwards throws away the warm-up, so the measured
    window runs cold while the report says it did not.
    """
    order = []

    async def reset():
        order.append(("reset", len(order)))

    class _Recording(_FakeAdapter):
        async def issue(self, spec, *, clock, sink=None):
            order.append(("issue", len(order)))
            return await super().issue(spec, clock=clock, sink=sink)

    asyncio.run(run_window(_Recording(service_s=0.002), _specs(), policy=_policy(), reset=reset))

    assert order[0][0] == "reset"
    assert any(kind == "issue" for kind, _ in order)


def test_think_time_separates_the_requests_of_one_virtual_user():
    """Section 7.2: a closed loop with no think time is a different workload, so it is
    declared rather than assumed. A driver that ignored the declaration would report a
    concurrency of N users who never pause as if they were N users who do."""
    adapter = _FakeAdapter(service_s=0.002)
    run = asyncio.run(
        run_window(
            adapter,
            _specs(),
            policy=_policy(concurrency=1, window_s=0.20, think_time_s=0.05, warmup_requests=1),
            reset=no_reset,
        )
    )

    inside = [r for r in run.records if r.in_window]
    assert len(inside) <= 6, f"0.20 s at 0.05 s think time cannot fit {len(inside)} requests"


def test_the_workload_is_asked_for_a_distinct_request_every_time():
    """Cache control (chapter 7 section 3) is the workload's job, and it needs an index.

    Replaying one prompt makes the second request onward a prefix-cache hit, which is a
    measurement of the cache rather than of the model unless it was declared.
    """
    seen = []

    def next_spec(i):
        seen.append(i)
        return RequestSpec(request_id=f"q{i}", messages=[{"role": "user", "content": f"{i}"}])

    asyncio.run(
        run_window(_FakeAdapter(service_s=0.002), next_spec, policy=_policy(), reset=no_reset)
    )

    assert len(seen) == len(set(seen)), "the same index was handed out twice"
    assert seen == sorted(seen), "indices are monotonic so a replay can be reconstructed"
