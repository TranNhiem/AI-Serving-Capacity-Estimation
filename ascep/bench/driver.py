"""The load driver: reset, warm-up, the measured closed loop, and the drain deadline.

Two separable halves live here on purpose. ``apply_boundary_rules`` is pure arithmetic
over timestamps -- which records are offered demand, which are latency samples, and which
the drain deadline turned into non-completions -- and imports no asyncio, so the
chapter 7 section 6 boundary rules can be falsified with synthetic records alone.
``run_window`` is the orchestration that produces those timestamps honestly.

The driver, not the adapter, owns the clock, the window and the absence of retries. A
harness that lets an adapter decide when a request started, or whether a failed request
may be re-sent, has changed both the workload and the error-rate denominator, and no
reduction over the resulting records can tell that it happened.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ascep.bench.adapters.base import Adapter, RequestSpec
from ascep.bench.records import Outcome, RequestRecord


@dataclass(frozen=True)
class WindowPolicy:
    """The declared operating point of one window, fixed before timing.

    The fields without defaults are the ones section 7.6 forbids defaulting: a number
    nobody chose would be deciding whether a straddler counted as a sample or a failure.
    """

    concurrency: int
    window_s: float
    # No default: this number changes the reported error rate, so it must be chosen.
    drain_deadline_s: float
    # No default either, for the same reason one layer up. Section 7.2 makes think time a
    # required declaration and permits zero only when the product really does submit the
    # next request immediately; a default of zero grants that permission to every caller
    # who forgot to carry the workload's figure across, and the run then saturates the
    # server with fewer users than the report claims -- with nothing in the records saying
    # so, because a request that was never delayed looks exactly like one that had no
    # delay to make.
    think_time_s: float
    warmup_requests: int = 0
    warmup_s: float = 0.0
    loop: str = "closed"
    repetition: int = 0
    poll_interval_s: float = 0.005

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least one virtual user")
        if self.window_s <= 0:
            raise ValueError("window_s must be positive: a zero-length window measures nothing")
        if self.drain_deadline_s < 0:
            raise ValueError("drain_deadline_s cannot be negative")
        if self.think_time_s < 0:
            raise ValueError("think_time_s cannot be negative")
        if self.warmup_requests < 0 or self.warmup_s < 0.0:
            raise ValueError("warm-up bounds cannot be negative")
        if self.warmup_requests == 0 and self.warmup_s == 0.0:
            raise ValueError(
                "a warm-up must be declared via warmup_requests or warmup_s: section 7.3 "
                "makes every repetition carry its own warm-up, and zero of both is not one"
            )
        if self.loop == "open":
            raise ValueError(
                "open-loop arrival is not implemented: silently running a closed loop for "
                "an open-loop declaration measures an arrival process the report denies"
            )
        if self.loop != "closed":
            raise ValueError(f"unknown loop declaration {self.loop!r}; only 'closed' is")


@dataclass(frozen=True)
class Boundary:
    """The resolution of one window's edges, as counts plus the timestamps they came from.

    The cohorts differ only at the edges, so a report that cannot show the edges cannot
    be checked against its own records.
    """

    t0: float
    window_s: float
    drain_deadline_s: float
    deadline: float
    offered: int
    warmup: int
    completed_in_window: int
    straddlers: int
    abandoned: int


def apply_boundary_rules(
    records: list[RequestRecord], *, t0: float, window_s: float, drain_deadline_s: float
) -> Boundary:
    """Mark every record with the cohort section 7.6 assigns it to, then count the cohorts.

    Mutates ``records`` in place because the mark -- ``in_window``, a deadline
    cancellation -- is a field on the record, and the reproduction bundle ships records.
    Had the ruling lived in a parallel structure, a reviewer re-reducing the bundle could
    neither see nor dispute where the harness drew the line.
    """
    close = t0 + window_s
    deadline = close + drain_deadline_s
    for record in records:
        record.in_window = t0 <= record.issued_ts <= close
        if not record.in_window:
            continue
        # Only an OK record may be re-marked. A record that already failed keeps the
        # cause it failed for: overwriting a refusal with "outstanding at the deadline"
        # hides a full admission queue, and refusal and abandonment call for different
        # operator action.
        outstanding = record.end_ts is None or record.end_ts > deadline
        if outstanding and record.outcome is Outcome.OK:
            record.outcome = Outcome.CANCELLED
            # The reason names the numeric deadline: "0.2% errors" at a 5 s drain and at
            # a 120 s drain are different claims, and the record must say which it was.
            record.error = (
                f"outstanding at the drain deadline: no completion by {deadline:.3f} s "
                "on the run clock (window close plus drain_deadline_s)"
            )

    offered = sum(1 for r in records if r.in_window)
    warmup = sum(1 for r in records if not r.in_window)
    completed = sum(
        1
        for r in records
        if r.outcome is Outcome.OK and r.end_ts is not None and t0 <= r.end_ts <= close
    )
    straddlers = sum(
        1
        for r in records
        if r.in_window
        and r.outcome is Outcome.OK
        and r.end_ts is not None
        and close < r.end_ts <= deadline
    )
    abandoned = sum(
        1
        for r in records
        if r.in_window
        and r.outcome is Outcome.CANCELLED
        and (r.end_ts is None or r.end_ts > deadline)
    )
    return Boundary(
        t0=t0,
        window_s=window_s,
        drain_deadline_s=drain_deadline_s,
        deadline=deadline,
        offered=offered,
        warmup=warmup,
        completed_in_window=completed,
        straddlers=straddlers,
        abandoned=abandoned,
    )


async def no_reset() -> None:
    """The explicit no-op state reset.

    Passing this declares that the repetition starts from whatever state the last one
    left, or that reset is handled outside the harness. That is a declaration, not a
    convenience: section 7.5 treats reused allocator, cache or scheduler state as
    evidence masquerading as independent, so choosing ``no_reset`` belongs in the run
    config, in prose -- not in an omitted argument.
    """


@dataclass(frozen=True)
class WindowRun:
    """One executed repetition: the records plus the declarations that give them meaning.

    ``window_s`` is the declared duration, not the span the records imply: a rate
    denominator that stretches with stragglers turns a slow window into a slightly
    longer one instead of a slower one (section 4.2).
    """

    records: list[RequestRecord]
    policy: WindowPolicy
    t0: float
    window_s: float
    drain_deadline_s: float
    warmup_count: int
    warmup_s_actual: float
    boundary: Boundary

    @property
    def offered(self) -> int:
        """Requests issued inside the window -- the error-rate denominator."""
        return self.boundary.offered

    @property
    def straddlers(self) -> int:
        """In-window requests that completed validly after close, before the deadline."""
        return self.boundary.straddlers

    @property
    def abandoned(self) -> int:
        """In-window requests the deadline turned into non-completions."""
        return self.boundary.abandoned


async def run_window(
    adapter: Adapter,
    next_spec: Callable[[int], RequestSpec],
    *,
    policy: WindowPolicy,
    reset: Callable[[], Awaitable[None]],
    clock: Callable[[], float] = time.perf_counter,
) -> WindowRun:
    """Run one repetition: reset, warm up, measure a closed loop, drain, rule the edges.

    ``records`` includes the warm-up traffic, marked out of window -- section 7.3
    discards warm-up from the statistics, not from the evidence. Filter on
    ``record.in_window`` before believing any count you compute yourself.

    One residual is left visible rather than papered over. The driver decides to send
    while its own clock reads before close, and the adapter stamps ``issued_ts`` on a
    second, later clock read; a request decided microseconds before close can therefore
    be stamped just after it and land outside the window. The gap is one clock read wide
    and bounded at one request per virtual user per window. It is left alone because the
    alternative -- refusing to issue during a guard band near the close -- shortens the
    declared offered-load window by an amount no report states, and an undeclared window
    is worse than a boundary effect a reader can bound from the record count.
    """
    records: list[RequestRecord] = []
    sunk: list[RequestRecord] = []
    next_index = 0

    def sink(record: RequestRecord) -> None:
        sunk.append(record)

    def take_index() -> int:
        nonlocal next_index
        index = next_index
        next_index += 1
        return index

    # Reset runs before anything else: warming up first and resetting after throws the
    # warm-up away, so the measured window would run cold while the report says it did
    # not (section 7.5).
    await reset()

    warmup_started = clock()
    warmup_index_base = next_index

    def warmup_done() -> bool:
        enough_requests = policy.warmup_requests == 0 or (
            next_index - warmup_index_base >= policy.warmup_requests
        )
        enough_time = policy.warmup_s == 0.0 or clock() - warmup_started >= policy.warmup_s
        return enough_requests and enough_time

    async def warm_user() -> None:
        while not warmup_done():
            spec = next_spec(take_index())
            records.append(await adapter.issue(spec, clock=clock, sink=sink))
            if policy.think_time_s:
                await asyncio.sleep(policy.think_time_s)

    await asyncio.gather(*(warm_user() for _ in range(policy.concurrency)))
    warmup_count = next_index - warmup_index_base
    warmup_s_actual = clock() - warmup_started

    t0 = clock()
    t1 = t0 + policy.window_s
    deadline = t1 + policy.drain_deadline_s

    async def user() -> None:
        # The loop runs to the declared close and no earlier. A guard band that stopped
        # issuing a millisecond early would shorten the offered-load window by an amount
        # nobody declared, which is the same class of error as stretching the denominator
        # -- see the docstring for the sub-millisecond residual this leaves instead.
        while clock() < t1:
            spec = next_spec(take_index())
            # Deliberately no retry around this call: a retried request vanishes from
            # the error-rate denominator and double-counts throughput, so a failed
            # request is recorded, never re-issued.
            records.append(await adapter.issue(spec, clock=clock, sink=sink))
            if policy.think_time_s:
                await asyncio.sleep(policy.think_time_s)

    tasks = [asyncio.create_task(user()) for _ in range(policy.concurrency)]
    # Poll on clock() rather than asyncio.wait(timeout=...): the deadline is measured on
    # the injected clock, while wait()'s timeout is measured on the event loop's, and a
    # test that injects a fake clock must move the deadline with it.
    while clock() < deadline and not all(task.done() for task in tasks):
        await asyncio.sleep(policy.poll_interval_s)
    for task in tasks:
        if not task.done():
            task.cancel()
    settled = await asyncio.gather(*tasks, return_exceptions=True)
    for result in settled:
        # CancelledError here is the deadline doing its job. Anything else is an adapter
        # breaking the contract that it never raises for a server-side failure, and it
        # voids the window: the records collected so far are missing an unknown number of
        # requests, so publishing them would be a rate over a denominator nobody can
        # reconstruct. A void window is a section 7 outcome; a quietly short one is not.
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result

    # Dedupe by request_id before the sunk records join: an adapter that hands a partial
    # record to the sink at cancellation and also returns it must not double-count into
    # the error-rate denominator.
    seen_ids = {record.request_id for record in records}
    for record in sunk:
        if record.request_id not in seen_ids:
            seen_ids.add(record.request_id)
            records.append(record)

    # Records outlive the run object, so each one carries its own operating point --
    # warm-up included, since those records ship in the bundle too.
    for record in records:
        record.concurrency = policy.concurrency
        record.repetition = policy.repetition

    boundary = apply_boundary_rules(
        records, t0=t0, window_s=policy.window_s, drain_deadline_s=policy.drain_deadline_s
    )
    return WindowRun(
        records=records,
        policy=policy,
        t0=t0,
        window_s=policy.window_s,
        drain_deadline_s=policy.drain_deadline_s,
        warmup_count=warmup_count,
        warmup_s_actual=warmup_s_actual,
        boundary=boundary,
    )
