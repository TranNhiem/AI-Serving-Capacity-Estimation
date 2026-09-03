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
import hashlib
import random
import statistics

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
    # Lock-step is declared (dephase=False) rather than the expectations changed: this
    # test audits the drain deadline, and with de-phasing the slow requests start
    # inside the de-phasing interval, land out of window, and are cancelled there --
    # breaking both assertions below for reasons that have nothing to do with the drain.
    run = asyncio.run(
        run_window(
            adapter,
            _specs(),
            policy=_policy(drain_deadline_s=0.05, dephase=False),
            reset=no_reset,
        )
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


# --- session replay -------------------------------------------------------------------
#
# The driver reaches a session plan through exactly two methods, so these tests supply a
# fake rather than the shipped ReplaySessionPlan. That is deliberate: what is under test
# here is that the driver walks a session correctly and stamps identity onto every record
# it produces, and a failure should name the driver rather than the prompt builder.


class _FakeStep:
    def __init__(self, turn_index, gap_s=0.0):
        self.turn_index = turn_index
        self.gap_s = gap_s


class _FakeShape:
    def __init__(self, steps):
        self.steps = steps


class _FakePlan:
    """Alternates a three-step session and a one-step session, recording what it was asked."""

    def __init__(self, shapes=None):
        self.shapes = shapes or [
            _FakeShape([_FakeStep(0), _FakeStep(0), _FakeStep(1)]),
            _FakeShape([_FakeStep(0)]),
        ]
        self.asked = []
        self.specced = []

    def shape(self, session_index):
        self.asked.append(session_index)
        return self.shapes[session_index % len(self.shapes)]

    def spec(self, *, session_index, step_index, request_id):
        self.specced.append((session_index, step_index))
        return RequestSpec(request_id=request_id, messages=[{"role": "user", "content": "x"}])


def test_a_window_with_no_source_of_work_at_all_is_refused():
    """Defaulting next_spec to None makes it possible to forget both, so both are checked.

    A window that issues nothing still returns records, a boundary and a duration, and
    reduce_window would divide by it: the run reports zero throughput at full concurrency
    as though the server had stalled, rather than as the empty run it was.
    """
    with pytest.raises(ValueError, match="exactly one source of work"):
        asyncio.run(run_window(_FakeAdapter(), policy=_policy(), reset=no_reset))


def test_a_window_given_both_a_spec_source_and_a_session_plan_is_refused():
    """Two workloads, one report. The driver must not be the one choosing between them.

    Silently preferring either would let a config declare an agent workload and measure
    single-shot requests, and the records would carry the agent declaration's name.
    """
    with pytest.raises(ValueError, match="exactly one source of work"):
        asyncio.run(
            run_window(
                _FakeAdapter(),
                _specs(),
                policy=_policy(),
                reset=no_reset,
                session_plan=_FakePlan(),
            )
        )


def test_a_session_run_refuses_a_think_time_on_top_of_its_captured_gaps():
    """The gaps are already in the capture, so a think time would be added to each of them.

    Left through, a 0.05 s think time on a six-step session adds 0.3 s of undeclared idle
    per session: fewer sessions fit the window, offered load falls, and the run reports a
    server that is less loaded than the declaration says it was.
    """
    with pytest.raises(ValueError, match="think_time_s = 0.0"):
        asyncio.run(
            run_window(
                _FakeAdapter(),
                policy=_policy(think_time_s=0.05),
                reset=no_reset,
                session_plan=_FakePlan(),
            )
        )


def test_every_request_of_a_session_run_carries_its_session_and_turn():
    """Without both, the reduction cannot tell a session's steps from independent requests.

    It would charge each step a cold prefill and compute a duty cycle over the wall clock
    of a loop that spent most of it waiting on tools -- the KV and prefill floors both
    come out too kind, which is the direction that oversells the hardware.
    """
    run = asyncio.run(
        run_window(
            _FakeAdapter(service_s=0.002),
            policy=_policy(concurrency=2, warmup_requests=2),
            reset=no_reset,
            session_plan=_FakePlan(),
        )
    )

    assert run.records, "the run issued nothing"
    for record in run.records:
        assert record.session_id is not None, f"{record.request_id} lost its session"
        assert record.turn_index is not None, f"{record.request_id} lost its turn"


def test_the_steps_of_one_session_are_issued_in_order_and_carry_the_captured_turns():
    """A session is a sequence, and replaying it out of order replays a different workload.

    The three-step shape has turns 0, 0, 1: two requests in one tool-calling turn, then a
    second turn. Shuffled, the record file says three separate turns, and the profiler's
    tool_calls_per_turn -- the whole reason the agent fields exist -- reads 1.0.
    """
    run = asyncio.run(
        run_window(
            _FakeAdapter(service_s=0.002),
            policy=_policy(concurrency=1, window_s=0.30, warmup_requests=1),
            reset=no_reset,
            session_plan=_FakePlan(),
        )
    )

    by_session = {}
    for record in sorted(run.records, key=lambda r: r.issued_ts):
        by_session.setdefault(record.session_id, []).append(record.turn_index)
    three_step = [turns for turns in by_session.values() if len(turns) == 3]
    assert three_step, "no session ran all three of its steps"
    for turns in three_step:
        assert turns == [0, 0, 1], f"steps were issued out of order: {turns}"


def test_two_rungs_of_the_same_ladder_draw_different_session_indices():
    """Otherwise rung eight replays the prompts rung one already left in the prefix cache.

    That is the flattering failure: measured capacity rises with concurrency because the
    later rungs stopped doing prefill, and the report reads as a scaling result.
    """
    first, second = _FakePlan(), _FakePlan()
    for plan, concurrency in ((first, 2), (second, 8)):
        asyncio.run(
            run_window(
                _FakeAdapter(service_s=0.002),
                policy=_policy(concurrency=concurrency, warmup_requests=2),
                reset=no_reset,
                session_plan=plan,
            )
        )

    assert first.asked and second.asked
    assert not set(first.asked) & set(second.asked), "two rungs drew the same session index"


def test_a_session_the_window_cut_short_is_counted_as_started_but_not_completed():
    """A truncated session contributes only its early -- and therefore shortest -- prompts.

    Nothing in the records distinguishes it from a genuinely short session, so a window
    shorter than a few session lengths silently reports a mean context below the capture
    and a prefill floor to match. The two counts are what lets a reader notice.
    """
    stalls = _FakePlan(shapes=[_FakeShape([_FakeStep(0), _FakeStep(0, gap_s=30.0), _FakeStep(1)])])

    # Lock-step is declared (dephase=False) rather than the expectation changed: this
    # test audits truncation counting, and with de-phasing the single user starts its
    # only session inside the de-phasing interval and parks in the 30 s gap, so
    # sessions_started would read 0 -- correct accounting for a run whose window
    # genuinely began no session, which is not what this fixture is about.
    run = asyncio.run(
        run_window(
            _FakeAdapter(service_s=0.002),
            policy=_policy(
                concurrency=1,
                window_s=0.10,
                drain_deadline_s=0.05,
                warmup_requests=1,
                dephase=False,
            ),
            reset=no_reset,
            session_plan=stalls,
        )
    )

    assert run.sessions_started >= 1
    assert run.sessions_completed == 0, "a session parked in a 30 s gap cannot have finished"


def test_a_request_run_reports_no_sessions_rather_than_one_per_request():
    """The counts default to zero, and a workload without sessions must leave them there.

    Reporting one session per request would make every request run look like a perfectly
    truncation-free agent run, which is a claim about a workload it never measured.
    """
    run = asyncio.run(
        run_window(_FakeAdapter(service_s=0.002), _specs(), policy=_policy(), reset=no_reset)
    )

    assert run.sessions_started == 0
    assert run.sessions_completed == 0
    assert all(r.session_id is None for r in run.records)


def test_a_step_cancelled_at_the_deadline_still_names_the_session_it_belonged_to():
    """The adapter hands that record to the sink, not back to the driver's call site.

    Stamped from the returned record alone it would arrive with a null session, and the
    reduction would read the run's one abandoned request as an independent single-shot --
    dropping it from the session whose duty cycle it is evidence for.
    """
    stalls = _FakePlan(shapes=[_FakeShape([_FakeStep(0), _FakeStep(1)])])

    run = asyncio.run(
        run_window(
            _FakeAdapter(service_s=0.002, slow_after=2, slow_s=30.0),
            policy=_policy(concurrency=1, window_s=0.10, drain_deadline_s=0.05, warmup_requests=1),
            reset=no_reset,
            session_plan=stalls,
        )
    )

    cancelled = [r for r in run.records if r.outcome is Outcome.CANCELLED]
    assert cancelled, "the 30 s step should have been cancelled at the deadline"
    for record in cancelled:
        assert record.session_id is not None, "a sunk record lost its session identity"


# --- de-phasing -----------------------------------------------------------------------


def _dephase_offsets(concurrency, repetition, cycle_s):
    """The driver's offset derivation, restated: seeded from the operating point, not the clock."""
    digest = hashlib.blake2b(
        f"dephase:{concurrency}:{repetition}".encode(), digest_size=8
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    return [rng.uniform(0.0, cycle_s) for _ in range(concurrency)]


def test_the_measured_window_does_not_open_with_the_fleet_in_lock_step():
    """A phase-locked fleet reports floor(W / C) completions per window, not N * W / C.

    The steps of that staircase are the false plateaus a capacity report mistakes for
    the saturation knee, so the first in-window issues must spread across a real
    fraction of the cycle: if they still land within a whisker of one another, the
    synchronised waves are back and so is the downward bias.
    """
    concurrency = 8
    run = asyncio.run(
        run_window(
            _FakeAdapter(service_s=0.05),
            _specs(),
            policy=_policy(concurrency=concurrency, warmup_requests=concurrency),
            reset=no_reset,
        )
    )

    assert run.dephase_s is not None and run.dephase_s > 0.0, "the fixture must de-phase"
    first_issues = sorted(r.issued_ts for r in run.records if r.in_window)[:concurrency]
    assert len(first_issues) == concurrency
    spread = first_issues[-1] - first_issues[0]
    assert spread >= 0.2 * run.dephase_s, (
        f"first in-window issues span {spread:.4f} s of a {run.dephase_s:.4f} s cycle; "
        "the users are still starting together"
    )


def test_dephase_false_releases_the_fleet_in_lock_step_just_as_before():
    """The escape hatch must be the old behaviour, or it reproduces nothing.

    A reviewer re-running a suspect ladder with dephase=False asks exactly one
    question: was that plateau floor(W / C)? A hatch that quietly de-phased anyway
    would answer it for a benchmark nobody ran, so this asserts the old release
    directly -- every user's first in-window issue within a fraction of one cycle
    of every other's.
    """
    concurrency = 8
    run = asyncio.run(
        run_window(
            _FakeAdapter(service_s=0.05),
            _specs(),
            policy=_policy(concurrency=concurrency, warmup_requests=concurrency, dephase=False),
            reset=no_reset,
        )
    )

    assert run.dephase_s is None
    first_issues = sorted(r.issued_ts for r in run.records if r.in_window)[:concurrency]
    assert len(first_issues) == concurrency
    spread = first_issues[-1] - first_issues[0]
    assert spread < 0.25 * 0.05, f"dephase=False still spread the starts over {spread:.4f} s"


def test_the_dephasing_offsets_are_deterministic_for_a_given_operating_point():
    """Offsets drawn from the clock would be a benchmark input nobody can replay.

    Two runs of one config have to produce the same offsets, or "the same rung
    measured twice" stops holding: a rate that depends on where the phases happened
    to fall is a rate no reviewer can re-derive, which is precisely the property
    the seeded derivation buys.
    """
    first = _dephase_offsets(16, 0, 35.0)
    second = _dephase_offsets(16, 0, 35.0)
    other_repetition = _dephase_offsets(16, 1, 35.0)

    assert len(first) == 16, "one offset per virtual user, in user order"
    assert all(0.0 <= offset < 35.0 for offset in first)
    assert first == second
    assert first != other_repetition, "the seed must feed the repetition, not relabel it"


def test_dephasing_traffic_is_marked_out_of_window_and_never_counted_as_offered_load():
    """The interval's requests exist to spread the fleet, not to load the window.

    Counting them as offered demand would inflate both the user count and the
    completion rate with traffic that was never part of the declared operating
    point -- the pair of errors section 7.6 names -- so every record stamped
    before t0 must be out of window and absent from the offered count.
    """
    run = asyncio.run(
        run_window(_FakeAdapter(service_s=0.01), _specs(), policy=_policy(), reset=no_reset)
    )

    assert run.dephase_s is not None and run.dephase_s > 0.0, "the fixture must de-phase"
    pre_window = [r for r in run.records if r.issued_ts < run.t0]
    during_interval = [r for r in pre_window if r.issued_ts >= run.t0 - run.dephase_s]
    assert during_interval, "no request was issued inside the de-phasing interval"
    assert all(not r.in_window for r in pre_window)
    assert run.boundary.warmup >= len(pre_window)
    assert run.warmup_count >= len(pre_window), "de-phasing traffic left the warm-up count"


def test_dephase_s_reports_the_interval_used_and_stays_none_when_it_is_skipped():
    """None is a finding, not a missing value: it means the fleet entered in lock-step.

    Folding the two states together (zero instead of None, say) would let a reader
    compare a de-phased ladder against a lock-step one without knowing, and any
    plateau in the second may be floor(W / C) stepping rather than saturation.
    The distinction is the only thing in the bundle that says which was run.
    """
    dephased = asyncio.run(
        run_window(_FakeAdapter(service_s=0.01), _specs(), policy=_policy(), reset=no_reset)
    )
    skipped = asyncio.run(
        run_window(
            _FakeAdapter(service_s=0.01),
            _specs(),
            policy=_policy(dephase=False),
            reset=no_reset,
        )
    )

    assert isinstance(dephased.dephase_s, float)
    assert 0.0 < dephased.dephase_s <= dephased.policy.window_s, "the clamp must bound"
    assert skipped.dephase_s is None


def test_de_phasing_recovers_the_completions_a_lock_step_window_rounds_away():
    """The whole point, stated as a number: N * W / C completions, not N * floor(W / C).

    Everything above checks that the fleet is spread. This checks that the spreading
    changes the published figure, because a de-phasing that left throughput on the same
    staircase would be ceremony. The fixture runs 2.5 cycles per user, so a locked fleet
    completes two per user and throws the half away, while a spread one collects the half
    from the users whose phase happens to land in it. On a GB200 ladder this exact
    rounding made concurrency 256 and 384 report an identical 7,488 tok/s -- a plateau
    the report would have published as the saturation knee, with the true rates 8,554 and
    9,984 tok/s and still climbing.

    Both runs share one fixture and are compared against each other rather than against
    an absolute count, so ordinary scheduling drift moves them together. If the drift is
    bad enough to move W / C onto an integer the regime under test no longer exists, and
    the test says so instead of failing for a reason that is not the code's.
    """
    concurrency = 24
    fixture = dict(concurrency=concurrency, window_s=0.25, drain_deadline_s=0.30)
    service_s = 0.10

    locked = asyncio.run(
        run_window(
            _FakeAdapter(service_s=service_s),
            _specs(),
            policy=_policy(warmup_requests=concurrency, dephase=False, **fixture),
            reset=no_reset,
        )
    )
    spread = asyncio.run(
        run_window(
            _FakeAdapter(service_s=service_s),
            _specs(),
            policy=_policy(warmup_requests=concurrency, **fixture),
            reset=no_reset,
        )
    )

    # Measured, not assumed: asyncio.sleep overshoots, and the cycle this ran at is the
    # only one the assertion may be scaled against.
    served = [r.e2e_s for r in locked.records if r.in_window and r.e2e_s is not None]
    cycle_s = statistics.median(served)
    cycles = fixture["window_s"] / cycle_s
    if abs(cycles - round(cycles)) < 0.15:
        pytest.skip(
            f"timing drift put this machine at {cycles:.2f} cycles per user, too close to "
            "a whole number for the rounding this test exists to observe"
        )

    per_user_locked = locked.boundary.completed_in_window / concurrency
    per_user_spread = spread.boundary.completed_in_window / concurrency
    assert per_user_locked == pytest.approx(float(int(cycles)), abs=0.2), (
        f"the locked fleet completed {per_user_locked:.2f} per user at {cycles:.2f} "
        "cycles: it was not on the staircase, so this fixture proves nothing"
    )
    assert per_user_spread > per_user_locked + 0.15, (
        f"de-phased {per_user_spread:.2f} completions per user against a locked "
        f"{per_user_locked:.2f} at {cycles:.2f} cycles: the fraction is still being lost"
    )


class _NonYieldingAdapter(Adapter):
    """Answers every request without awaiting anything, and reports a latency it never spent.

    Adapters that talk to a server await a socket, so every request hands control back to
    the event loop. An in-process one need not, and several fakes in this repository's own
    suite do not: they fabricate the timestamps and return. That makes this the fixture for
    a driver that assumes it will be rescheduled -- and the fabricated e2e matters as much
    as the missing await, because a zero-latency fake estimates a zero cycle and skips
    de-phasing altogether, which is the path that never broke.

    The runaway is caught here rather than with asyncio.wait_for, and the reason is the
    defect itself: a coroutine that never suspends starves the event loop, so no timer the
    loop owns can fire. The only code a runaway loop keeps executing is this method, so the
    budget is checked in it, on the driver's own clock.
    """

    def __init__(self, reported_e2e_s=0.05, budget_s=1.0):
        self.reported_e2e_s = reported_e2e_s
        self.budget_s = budget_s
        self.started = None
        self.issued = 0

    @property
    def name(self):
        return "non-yielding"

    async def issue(self, spec, *, clock, sink=None):
        now = clock()
        if self.started is None:
            self.started = now
        self.issued += 1
        if now - self.started > self.budget_s:
            raise AssertionError(
                f"still issuing {self.issued} requests {now - self.started:.2f} s into a "
                f"window declared at {self.budget_s:.2f} s of total budget: the loop's end "
                "never became finite, which is the hang this fixture exists to turn into a "
                "failure"
            )
        return RequestRecord(
            request_id=spec.request_id,
            issued_ts=now,
            outcome=Outcome.OK,
            first_token_ts=now + 0.001,
            end_ts=now + self.reported_e2e_s,
            output_tokens=10,
        )


def test_the_window_closes_even_when_the_adapter_never_yields_to_the_event_loop():
    """The window's end must be a number the driver knows before it launches anyone.

    De-phasing first shipped with the close left at infinity until the main coroutine woke
    from the de-phasing wait and assigned it. Every adapter that awaits a socket yields, so
    every run against a real server set it and the ladder on the GB200 measured correctly;
    an adapter that answers in-process never yields, the main coroutine was never
    rescheduled, and the first virtual user looped on an infinite bound appending a record
    per iteration until the process was killed at 32 GB of resident memory. It took out the
    whole suite, not one test, because the runaway held the interpreter.

    The interval is known before it is waited out, so there was never anything to observe:
    t0 and the close are arithmetic on t_launch. This test pins that they are computed
    rather than stamped, and it must stay cheap -- a non-yielding adapter spins hot for the
    whole window, so the window here is milliseconds.
    """
    adapter = _NonYieldingAdapter(reported_e2e_s=0.05, budget_s=1.0)

    run = asyncio.run(
        run_window(
            adapter,
            _specs(),
            policy=_policy(window_s=0.05, drain_deadline_s=0.05, warmup_requests=4),
            reset=no_reset,
        )
    )

    assert run.dephase_s == pytest.approx(0.05, abs=0.02), (
        f"dephase_s was {run.dephase_s!r}: de-phasing has to be engaged for this fixture to "
        "cover the regression, and a skipped interval reproduces the old release order"
    )
    assert any(record.in_window for record in run.records), (
        "the window opened and closed without offering anything, so returning proves nothing"
    )
