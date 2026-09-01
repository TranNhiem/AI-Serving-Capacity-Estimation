"""The reduction from raw per-request records to the figures a report publishes.

This module is the whole reason the protocol keeps records raw. Nothing here could be
recomputed from a harness's summary output: percentiles cannot be re-derived from other
percentiles, a window cannot be re-sliced from a mean, and an error-rate denominator
cannot be audited against records that were silently dropped. Chapter 4 sections 4.2-4.3
and chapter 7 section 6 therefore define every published number as a reduction over the
record list, and this file is that reduction. A third party handed the same JSONL must
obtain identical figures without importing -- or trusting -- anything that touched a
network, which is why this file depends on the standard library and the record module
alone.

The rules that look pedantic are the load-bearing ones:

- percentiles use Hyndman-Fan type-7 interpolation and refuse to exist below
  ``n >= 1/(1-p)``: below that floor the interpolated value is the observed sample
  maximum wearing a tail-statistic label, and "p99 = the max" has failed gates a healthy
  deployment would pass and passed gates a sick one should fail.
- between the absolute floor and ``10/(1-p)`` the value is reported but flagged
  low-confidence: one observation sits in the tail, so a single straggler moves the
  figure and it must never ship as a bare point estimate.
- ITL comes only from pooled per-token gaps, never from ``e2e / output_tokens``: a span
  average cannot see a stall inside a request, which is the exact event an ITL gate
  exists to catch.
- the error-rate denominator is issued requests, including refusals: a server that sheds
  a third of its offered load must not score a 0% error rate.
- rates divide by the declared window, never the record-implied span: that span
  stretches with stragglers exactly when the system is saturated, which is exactly where
  the number decides the tier.
- every figure the records cannot support is ``None`` with a recorded reason, because an
  honest gap is conforming telemetry and a substituted statistic is not.

Determinism is part of the contract (section 4.8): any resampling here is seeded by the
caller and driven by ``random.Random``, never the module-level generator, so two sites
handed the same records and seed produce bit-identical bounds.
"""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# Straight from the module, not the package: the package initialiser is where re-exports
# accumulate, and the first one that reaches back into this file turns a working import into
# a cycle that only shows up in whichever module happens to be imported first.
from ascep.bench.records import Outcome, RequestRecord

#: Sub-millisecond disagreement between timestamps is event-loop jitter and unsynchronised
#: clocks, not corruption; beyond it the record's latency arithmetic is garbage and must
#: not enter any aggregate. Slack wider than this would smear the millisecond-scale ITL
#: measurements the validation exists to protect.
_SKEW_SLACK_S = 1e-3

#: Above this measured tokens-per-chunk ratio a window's pooled gaps are inter-chunk, not
#: inter-token, and ITL is taken from the per-request means instead. A tolerance band, not
#: a correction: a stream that really is one token per chunk measures exactly 1.0, and 1.05
#: leaves room for the occasional double-packed delta without flipping the population of a
#: window that is otherwise per-token. The band is safe because a mean chunk of 1.05 tokens
#: cannot inflate a gap by more than five percent, which is below the precision any capacity
#: tier is reported at -- while the coalescing this guards against runs to six and a half
#: tokens a chunk.
_COALESCED_ABOVE_TOKENS_PER_CHUNK = 1.05


def percentile(samples: Sequence[float], p: float) -> float:
    """Hyndman-Fan type-7 quantile: linear interpolation on rank ``p * (n - 1)``.

    One convention is named so that two analysts handed the same records derive the same
    number; nearest-rank would answer 3 where this answers 2.5. The sample-size floor is
    deliberately NOT enforced here -- it depends on what the figure feeds, so it lives in
    :func:`reduce_window`. Raises on an empty sequence: silent defaults are how a
    missing distribution turns into a flattering zero.
    """
    if not samples:
        raise ValueError("percentile of an empty sample set is undefined")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p}")
    xs = sorted(samples)
    n = len(xs)
    rank = p * (n - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return xs[lo] + frac * (xs[hi] - xs[lo])


def absolute_floor(p: float) -> int:
    """Minimum n at which a percentile may be reported at all: ``ceil(1/(1-p))``.

    Below this the interpolated figure is the observed maximum relabelled as a tail
    statistic. The rounding guard exists because 1/(1-0.95) is 19.999... in binary
    floating point, not 20.
    """
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p must lie in [0, 1), got {p}")
    return math.ceil(round(1.0 / (1.0 - p), 9))


def advisory_floor(p: float) -> int:
    """n at which a percentile carries ten tail observations: ``ceil(10/(1-p))``.

    Between this and the absolute floor the figure is publishable but straggler-moved,
    so it is flagged rather than hidden -- the protocol bars substitution, not honesty.
    """
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p must lie in [0, 1), got {p}")
    return math.ceil(round(10.0 / (1.0 - p), 9))


def bootstrap_ci(
    samples: Sequence[float],
    p: float,
    *,
    seed: int,
    resamples: int = 1000,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile-bootstrap interval for the type-7 p-quantile of ``samples``.

    The generator is ``random.Random(seed)`` and the seed is a required argument, never
    drawn from the clock: section 4.8 demands that two sites handed identical records
    and arguments obtain bit-identical bounds across processes and across years. Returns
    ``None`` below the absolute floor -- an interval around an unmeasured figure would
    imply the figure exists.
    """
    xs = list(samples)
    if len(xs) < absolute_floor(p):
        return None
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    rng = random.Random(seed)
    n = len(xs)
    stats: list[float] = []
    for _ in range(resamples):
        draw = [xs[rng.randrange(n)] for _ in range(n)]
        stats.append(percentile(draw, p))
    alpha = (1.0 - level) / 2.0
    return (percentile(stats, alpha), percentile(stats, 1.0 - alpha))


@dataclass(frozen=True)
class SloGates:
    """Window-level SLO gates, fixed before the run.

    Every gate is a percentile or a rate over the whole window, so no individual request
    meets or fails one: ``None`` fields are simply not evaluated, but a configured gate
    whose statistic was unmeasurable counts as FAILED -- "the framework could not expose
    the tail" is an honest gap, never a pass.
    """

    ttft_p95_max_s: float | None = None
    itl_p95_max_s: float | None = None
    e2e_p95_max_s: float | None = None
    error_rate_max_pct: float | None = None


@dataclass
class WindowSummary:
    """Every figure a report may publish for one measurement window.

    ``None`` always means unmeasured, with the reason in :attr:`reasons`; figures in
    :attr:`low_confidence` sit between the absolute and advisory floors and must carry
    an interval when they decide a tier boundary.

    ``n_issued``, ``n_completed`` and ``n_latency_samples`` count three different cohorts
    that section 7.6 deliberately keeps apart -- offered demand, completions attributed to
    this window, and valid latency observations. They are equal below saturation and
    differ under overload, so a reader comparing them is reading the boundary effect, not
    an inconsistency.

    ``tokens_per_stream_chunk`` is the window's measured substitution factor:
    server-reported output tokens divided by streamed content chunks, summed over the
    usable records before dividing. A value above the coalescing threshold is what
    switched :attr:`itl_population` to the per-request mean. ``stream_chunk_gap_p50_s``
    and ``stream_chunk_gap_p95_s`` are what the client actually observed at the transport
    -- one sample per chunk, however many decode steps the server folded into it --
    published under a name that is not ITL because they are not ITL: on the rehearsal this
    change serves, the concurrency-128 rung measured 6.65 tokens per chunk and a "pooled
    ITL p95" of 0.2953 s was this chunk-gap tail while the per-request ITL p95 was
    0.0247 s. Both are computed from the pooled gaps whenever those gaps exist, never only
    on divergence, so a reader can compute the factor and compare across runs; a field
    that appears only when something went wrong is a field nobody builds a comparison on.
    """

    n_issued: int
    n_completed: int
    n_latency_samples: int
    excluded_error_count: int
    excluded_invalid_count: int
    excluded_warmup_count: int
    error_rate_pct: float | None
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    ttft_p99_s: float | None
    itl_p50_s: float | None
    itl_p95_s: float | None
    itl_p99_s: float | None
    itl_population: str | None
    e2e_p50_s: float | None
    e2e_p95_s: float | None
    e2e_p99_s: float | None
    output_tok_s: float | None
    requests_per_s: float | None
    goodput_tok_s: float | None
    slo_pass: bool | None
    # These three default to None ("unmeasured") so summaries built before they existed
    # keep working -- a mandatory field here would raise in every hand-built WindowSummary
    # in tests/test_bench_ladder.py the moment the module imports. A dataclass forbids a
    # defaulted field ahead of the figure fields above, which pins them to the last slots
    # before the defaulted containers rather than beside itl_population, whose disclosure
    # they complete.
    tokens_per_stream_chunk: float | None = None
    stream_chunk_gap_p50_s: float | None = None
    stream_chunk_gap_p95_s: float | None = None
    low_confidence: frozenset[str] = field(default_factory=frozenset)
    reasons: Mapping[str, str] = field(default_factory=dict)


@dataclass
class SliceRow:
    """One row of the steady-state slice table.

    The table exists to expose a bad interval, so accepted and completed are counted on
    different timestamps and concurrency is time-weighted: a burst the server absorbs
    and pays for later must be visible as a shape, not averaged flat. They are also
    counted over different cohorts -- ``accepted`` and ``errors`` over arrivals the window
    offered, ``completed`` and ``achieved_concurrency`` over every request in flight,
    warm-up included -- so the first row of a healthy run may complete more than it
    accepted. That is the boundary being reported, not a miscount.
    """

    index: int
    t_start: float
    t_end: float
    accepted: int
    completed: int
    errors: int
    accepted_req_s: float
    completed_req_s: float
    error_rate_pct: float | None
    achieved_concurrency: float
    output_tok_s: float | None


def _p_label(p: float) -> str:
    return f"p{round(p * 100)}"


def _is_monotonic(record: RequestRecord) -> bool:
    """True unless timestamps regress by more than the clock-skew slack.

    A non-monotonic record is excluded and counted, never clamped: a negative gap folded
    into the pool corrupts every percentile, while clamping to zero silently biases the
    median toward zero.
    """
    stamps = [record.issued_ts]
    for ts in (record.connect_ts, record.first_token_ts):
        if ts is not None:
            stamps.append(ts)
    stamps.extend(record.token_ts)
    if record.end_ts is not None:
        stamps.append(record.end_ts)
    return all(b >= a - _SKEW_SLACK_S for a, b in zip(stamps, stamps[1:]))


def _per_request_itl(record: RequestRecord) -> float | None:
    """Section 4.1's per-request ITL: the decode span over the decode steps it covers.

    The span runs from the first token, not from issue: folding TTFT in conflates prefill
    with decode and yields a flat per-token figure while user-visible latency is not. With
    no per-token stamps the last token's arrival is unknown and ``end_ts`` stands in for it,
    which includes stream teardown and so overstates ITL slightly -- the honest direction,
    and the reason the pooled population is preferred wherever the stamps exist.
    """
    if record.first_token_ts is None or record.output_tokens is None:
        return None
    if record.output_tokens < 2:
        # One token is a TTFT observation with no decode phase behind it; dividing by zero
        # steps, or by one, would put a prefill measurement into the ITL distribution.
        return None
    last = record.token_ts[-1] if record.token_ts else record.end_ts
    if last is None:
        return None
    return (last - record.first_token_ts) / (record.output_tokens - 1)


def _stat(
    samples: Sequence[float],
    p: float,
    name: str,
    reasons: dict[str, str],
    low_confidence: set[str],
) -> float | None:
    """One percentile under the section 4.3 floor discipline, or ``None`` with a reason."""
    n = len(samples)
    floor = absolute_floor(p)
    if n < floor:
        reasons[name] = f"(U) {n} samples; {_p_label(p)} requires {floor} (section 4.3)"
        return None
    if n < advisory_floor(p):
        # One observation in the tail: reportable, but a single straggler moves it.
        low_confidence.add(name)
    return percentile(samples, p)


def reduce_window(
    records: Sequence[RequestRecord],
    *,
    window_s: float,
    t0: float,
    gates: SloGates | None = None,
    seed: int = 0,
) -> WindowSummary:
    """Reduce the records of one declared measurement window to its published figures.

    The window is ``[t0, t0 + window_s]``. The denominator of every rate is ``window_s``,
    the declared post-warmup duration -- never the first-arrival-to-last-completion span
    implied by the records, which stretches with stragglers exactly when the system is
    slowest. ``t0`` is required rather than defaulted because section 7.6 splits requests
    by where they sit relative to the window edges, and a reduction that does not know
    where the window opened cannot apply that split; it can only pick one cohort and hope,
    which is the resolution the drain deadline exists to take away from the harness.
    ``seed`` is accepted so callers pass one deterministic value through the whole
    reduction; intervals computed from these records must reuse it rather than draw
    their own.

    The drain deadline is applied upstream, by ``ascep.bench.driver.apply_boundary_rules``,
    which marks a request outstanding at the deadline as cancelled before the records are
    written. This function therefore reads the ruling rather than re-deriving it, and
    records that never passed through it are an ungraded bundle: a request that finished
    forty seconds after close would arrive here as a valid completion and enter the tail.
    """
    del seed  # reserved for interval computation; a default drawn from the clock is barred
    t1 = t0 + window_s
    # Warm-up is marked rather than deleted so the bundle keeps it, which means the
    # reduction is the thing that has to drop it. Handed a whole bundle, a reduction that
    # trusted its caller would fold cold-cache and cold-scheduler requests into the tail
    # and call the result steady state -- the exact figure section 7.3 discards them for.
    excluded_warmup_count = sum(1 for record in records if not record.in_window)
    offered = [record for record in records if record.in_window]
    n_issued = len(offered)
    excluded_error_count = 0
    excluded_invalid_count = 0
    usable: list[RequestRecord] = []
    for record in offered:
        if record.outcome is not Outcome.OK:
            excluded_error_count += 1
        elif not _is_monotonic(record):
            excluded_invalid_count += 1
        else:
            usable.append(record)
    # Latency counts a request that arrived inside the window and completed validly, even
    # if it completed after close: the driver has already turned anything still outstanding
    # at the drain deadline into a failure, so a late finisher reaching here is a real tail
    # sample and dropping it would lower the tail by deleting the slowest requests.
    n_latency_samples = len(usable)
    # Issued is the normative denominator: a record exists from the moment the driver
    # decided to send, so refusals and cancellations are failures, not absences.
    error_rate_pct = 100.0 * excluded_error_count / n_issued if n_issued else None

    # Rates attribute a completion to the window it finished in, not the one it was
    # offered in. Both halves of that rule matter and they cancel at the edges: a request
    # issued inside the window and finishing after close is a latency sample but not a
    # completion of this window, and a warm-up request finishing inside it is a completion
    # of this window but neither demand nor a latency sample. Applying only the half that
    # excludes traffic biases steady-state throughput low at the opening edge and the half
    # that includes it biases high at the closing edge; counting by completion instant
    # measures across the boundary instead of being moved by it.
    finished_here = [
        record
        for record in records
        if record.outcome is Outcome.OK
        and record.end_ts is not None
        and t0 <= record.end_ts <= t1
        and _is_monotonic(record)
    ]
    n_completed = len(finished_here)

    ttft: list[float] = []
    gaps: list[float] = []
    means: list[float] = []
    e2e: list[float] = []
    for record in usable:
        if record.ttft_s is not None:
            ttft.append(record.ttft_s)
        gaps.extend(record.itls_s)
        mean = _per_request_itl(record)
        if mean is not None:
            means.append(mean)
        if record.e2e_s is not None:
            e2e.append(record.e2e_s)

    # Summed across the window before dividing, never averaged per request: the
    # substitution factor is a property of the window's whole token population, and a mean
    # of per-request ratios would let a handful of very short responses outvote the
    # traffic.
    # The first content chunk is stamped in first_token_ts and every later one in token_ts,
    # so the chunk count is len(token_ts) + 1 wherever a first token arrived. Counting only
    # token_ts would report tokens/(tokens - 1) for a perfectly per-token stream -- 1.02 on
    # a 50-token reply and 1.33 on a four-token one -- and the short answers in a VLM
    # workload would trip the coalescing switch on an artefact of where the harness files
    # the opening timestamp.
    stamped = [record for record in usable if record.token_ts and record.output_tokens]
    n_chunks = sum(
        len(record.token_ts) + (1 if record.first_token_ts is not None else 0)
        for record in stamped
    )
    tokens_per_stream_chunk: float | None
    if n_chunks:
        tokens_per_stream_chunk = sum(record.output_tokens for record in stamped) / n_chunks
    else:
        tokens_per_stream_chunk = None

    # Two independent substitutions hide under the one word "ITL", and itl_population must
    # disclose both.
    #
    # Pooled gaps versus per-request means: chapter 4 section 4.1 defines ITL per request
    # as the mean over the decode phase; the gap population is finer and is what an ITL
    # gate is actually for, since a stall inside one request survives pooling and
    # disappears under a per-request mean.
    #
    # Token gaps versus chunk gaps: token_ts is stamped per streamed chunk, and a server
    # under load coalesces several decode steps into one SSE delta, at which point a
    # pooled "gap" is an inter-chunk latency wearing an inter-token name. On the vLLM
    # rehearsal this reduction serves, the concurrency-128 rung packed 6.65 tokens into
    # the average chunk and the pooled p95 read 0.2953 s against a per-request p95 of
    # 0.0247 s; against a declared 0.05 s gate the choice of population moved sustainable
    # capacity from 1,180 tok/s to 2,503 tok/s. Pooling's finer grain is no defence once
    # the gaps no longer mean per token -- a finer sample of the wrong quantity is still
    # the wrong quantity -- so a window past _COALESCED_ABOVE_TOKENS_PER_CHUNK takes the
    # per-request means, whose denominator (each request's server-reported token count)
    # the transport cannot inflate. Where the factor sits at or below the threshold the
    # two populations agree, 0.0060 s against 0.0060 s at the rehearsal's concurrency-1
    # rung, and that agreement is what validates the per-request estimator for the
    # coalesced rungs rather than merely asserting it.
    #
    # The summary says which population it used either way: two labs reducing the same
    # records and quietly choosing differently is the reproducibility failure the
    # percentile convention exists to prevent.
    if gaps and (
        tokens_per_stream_chunk is None
        or tokens_per_stream_chunk <= _COALESCED_ABOVE_TOKENS_PER_CHUNK
    ):
        itl, itl_population = gaps, "pooled-gaps"
    elif means:
        itl, itl_population = means, "per-request-mean"
    elif gaps:
        # Coalesced, but no request yielded a per-request mean either: the pooled chunk
        # gaps are the only samples left, and publishing them under their population label
        # is honest where dropping them would hide the window's decode phase entirely.
        itl, itl_population = gaps, "pooled-gaps"
    else:
        itl, itl_population = [], None

    reasons: dict[str, str] = {}
    low: set[str] = set()
    if tokens_per_stream_chunk is None:
        reasons["tokens_per_stream_chunk"] = (
            "(U) no usable record carried both per-token stamps and a server token count"
        )
    ttft_p50_s = _stat(ttft, 0.50, "ttft_p50_s", reasons, low)
    ttft_p95_s = _stat(ttft, 0.95, "ttft_p95_s", reasons, low)
    ttft_p99_s = _stat(ttft, 0.99, "ttft_p99_s", reasons, low)
    itl_p50_s = _stat(itl, 0.50, "itl_p50_s", reasons, low)
    itl_p95_s = _stat(itl, 0.95, "itl_p95_s", reasons, low)
    itl_p99_s = _stat(itl, 0.99, "itl_p99_s", reasons, low)
    # Published from the pooled gaps whenever they exist, including windows whose ITL was
    # taken from those same gaps: a field present in every row lets a reader compute the
    # substitution factor and compare across runs, while one that appears only when
    # something went wrong is a field nobody builds a comparison on. They carry a chunk
    # name rather than the ITL name because each sample may span several decode steps. The
    # section 4.3 floor discipline applies unmodified, since a chunk-gap tail estimated
    # from too few stamps is overclaimed exactly as an ITL one is.
    stream_chunk_gap_p50_s = _stat(gaps, 0.50, "stream_chunk_gap_p50_s", reasons, low)
    stream_chunk_gap_p95_s = _stat(gaps, 0.95, "stream_chunk_gap_p95_s", reasons, low)
    e2e_p50_s = _stat(e2e, 0.50, "e2e_p50_s", reasons, low)
    e2e_p95_s = _stat(e2e, 0.95, "e2e_p95_s", reasons, low)
    e2e_p99_s = _stat(e2e, 0.99, "e2e_p99_s", reasons, low)

    requests_per_s: float | None
    output_tok_s: float | None
    if n_issued == 0 or window_s <= 0:
        # Zero figures here would be claims about traffic that was never observed.
        requests_per_s = None
        output_tok_s = None
    else:
        requests_per_s = n_completed / window_s
        counted = [r.output_tokens for r in finished_here if r.output_tokens is not None]
        output_tok_s = sum(counted) / window_s if counted else None
        if not counted:
            reasons["output_tok_s"] = "(U) no completed record reported output_tokens"

    if gates is None:
        slo_pass = None
    else:
        checks: list[bool] = []
        if gates.ttft_p95_max_s is not None:
            checks.append(ttft_p95_s is not None and ttft_p95_s <= gates.ttft_p95_max_s)
        if gates.itl_p95_max_s is not None:
            checks.append(itl_p95_s is not None and itl_p95_s <= gates.itl_p95_max_s)
        if gates.e2e_p95_max_s is not None:
            checks.append(e2e_p95_s is not None and e2e_p95_s <= gates.e2e_p95_max_s)
        if gates.error_rate_max_pct is not None:
            checks.append(error_rate_pct is not None and error_rate_pct <= gates.error_rate_max_pct)
        # A gate whose statistic is None is a failed gate: unmeasurable is not a pass,
        # or engine-ceiling runs get sold as gated user capacity.
        slo_pass = all(checks)
    # Goodput is a property of the window: no request passes a percentile, so a failed
    # or unevaluable window has raw measured-tier throughput and no goodput at all.
    goodput_tok_s = output_tok_s if slo_pass else None

    return WindowSummary(
        n_issued=n_issued,
        n_completed=n_completed,
        n_latency_samples=n_latency_samples,
        excluded_error_count=excluded_error_count,
        excluded_invalid_count=excluded_invalid_count,
        excluded_warmup_count=excluded_warmup_count,
        error_rate_pct=error_rate_pct,
        ttft_p50_s=ttft_p50_s,
        ttft_p95_s=ttft_p95_s,
        ttft_p99_s=ttft_p99_s,
        itl_p50_s=itl_p50_s,
        itl_p95_s=itl_p95_s,
        itl_p99_s=itl_p99_s,
        itl_population=itl_population,
        e2e_p50_s=e2e_p50_s,
        e2e_p95_s=e2e_p95_s,
        e2e_p99_s=e2e_p99_s,
        output_tok_s=output_tok_s,
        requests_per_s=requests_per_s,
        goodput_tok_s=goodput_tok_s,
        slo_pass=slo_pass,
        tokens_per_stream_chunk=tokens_per_stream_chunk,
        stream_chunk_gap_p50_s=stream_chunk_gap_p50_s,
        stream_chunk_gap_p95_s=stream_chunk_gap_p95_s,
        low_confidence=frozenset(low),
        reasons=reasons,
    )


def _slice_index(edges: Sequence[float], ts: float) -> int | None:
    """The slice containing ``ts``, or None outside the declared window.

    Records outside the declared window land in no slice: the table tiles the declared
    span, and a straggler finishing late must not stretch it.
    """
    if ts < edges[0] or ts > edges[-1]:
        return None
    i = bisect.bisect_right(edges, ts) - 1
    return min(i, len(edges) - 2)


def slice_window(
    records: Sequence[RequestRecord],
    *,
    window_s: float,
    n_slices: int,
    t0: float,
    trim_slices: int = 0,
) -> list[SliceRow]:
    """Cut ``[t0, t0 + window_s]`` into ``n_slices`` adjacent equal slices.

    Boundaries are computed as ``t0 + i * window_s / n_slices`` rather than accumulated,
    so floating-point error cannot open a gap between two rows and let a bad interval
    fall out of the table that exists to expose it. Nothing is trimmed unless the caller
    declares it: dropping the ramp is a predeclared, reportable rule, and trimmed rows
    keep their original ``index`` so the removal is visible as a gap in the index
    column, not a silent renumbering.
    """
    if n_slices < 1:
        raise ValueError("n_slices must be at least 1")
    if trim_slices < 0 or 2 * trim_slices >= n_slices:
        # An empty slice table reads as "steady state was not in question" rather than as
        # "the evidence for it was trimmed away", which is the opposite of what section 4
        # asks the table to show.
        raise ValueError(
            f"trim_slices={trim_slices} would leave no rows of {n_slices}; "
            "the slice table is the steady-state evidence and cannot be trimmed to nothing"
        )
    edges = [t0 + i * window_s / n_slices for i in range(n_slices + 1)]
    accepted = [0] * n_slices
    completed = [0] * n_slices
    errors = [0] * n_slices
    tokens = [0] * n_slices
    tokens_seen = [False] * n_slices
    busy = [0.0] * n_slices
    for record in records:
        # Only an arrival the window itself offered is demand on this table. A warm-up
        # request is not accepted here and its failure is not this window's error, but it
        # is still occupying the server -- so it is filtered out of the arrival columns
        # rather than out of the table, and the occupancy and completion columns below see
        # every record. Dropping it wholesale would report a first slice that looks calmer
        # than the machine was, in the table whose only job is to show whether it was.
        if record.in_window:
            ia = _slice_index(edges, record.issued_ts)
            if ia is not None:
                accepted[ia] += 1
                if record.outcome is not Outcome.OK:
                    errors[ia] += 1
        if record.end_ts is not None:
            ic = _slice_index(edges, record.end_ts)
            if ic is not None:
                completed[ic] += 1
                if record.output_tokens is not None:
                    tokens[ic] += record.output_tokens
                    tokens_seen[ic] = True
            # Time-weighted occupancy, not a head count at arrival: two requests each
            # spanning half a slice are a concurrency of one, and arrivals would say two.
            for i in range(n_slices):
                overlap = min(record.end_ts, edges[i + 1]) - max(record.issued_ts, edges[i])
                if overlap > 0.0:
                    busy[i] += overlap
    rows: list[SliceRow] = []
    for i in range(n_slices):
        duration = edges[i + 1] - edges[i]
        rows.append(
            SliceRow(
                index=i,
                t_start=edges[i],
                t_end=edges[i + 1],
                accepted=accepted[i],
                completed=completed[i],
                errors=errors[i],
                accepted_req_s=accepted[i] / duration,
                completed_req_s=completed[i] / duration,
                error_rate_pct=(100.0 * errors[i] / accepted[i]) if accepted[i] else None,
                achieved_concurrency=busy[i] / duration,
                output_tok_s=(tokens[i] / duration) if tokens_seen[i] else None,
            )
        )
    if trim_slices:
        rows = rows[trim_slices : n_slices - trim_slices]
    return rows
