"""Assembling the draft report ``ascep bench`` writes, and nothing else.

Split out of :mod:`ascep.bench.run` because two callers need it and only one of them runs a
benchmark: :func:`ascep.bench.rereduce.rebuild_report` re-derives a published report from its
bundle, and if it reimplemented the assembly the two would drift -- a rebuild that disagrees
with the report it was checking would then be indistinguishable from the bundle having been
tampered with, which is the one question ``ascep reduce --check`` exists to answer. It used to
import ``run._build_report`` through the underscore for that reason. The seam is real, so it is
declared: everything here reads finished measurements and writes JSON, touches no engine, opens
no socket, and decides nothing about what to measure next.

The house rule for every provenance helper below is the same one the protocol states for (M),
(I) and (U): a value and the reason it is absent are mutually exclusive, so setting one clears
the other. A report carrying both says the number was measured *and* was not.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone

from ascep import conformance, init
from ascep.bench import ladder

__all__ = ["build_report"]


def _known(node, key, value):
    """Set ``node[key]`` and drop the ``_u_reason`` companion that the value makes false."""
    node[key] = value
    node.pop(f"{key}_u_reason", None)


#: How far apart two per-rung context means must be before they count as different shapes.
#: Two orders of magnitude above the sampling spread one declared shape shows -- a GB200
#: ladder at a single 1,500-token shape spread its six rung means over 0.14 percent -- and far
#: below the separation a curve worth interpolating over needs. Shapes closer together than
#: this do not make an interpolable curve either, so collapsing them is the honest reading
#: rather than a lost distinction.
_CONTEXT_LENGTH_TOLERANCE = 0.05


def _distinct_context_lengths(rows) -> int:
    """How many genuinely different context lengths the rows cover.

    Counted with a tolerance rather than as a set, because ``context_tokens`` is a per-rung
    MEAN of measured lengths: six rungs of one declared shape land on six distinct floats,
    and ``len(set(...)) < 3`` is then false for every real campaign. That made
    ``run.single_point`` unreachable -- present, documented, and never once taken -- so a
    single-shape ladder published itself as a context curve and silenced the one C4 finding
    written for exactly that campaign.
    """
    distinct = 0
    group_floor = None
    for length in sorted(
        row["context_tokens"]
        for row in rows
        if isinstance(row.get("context_tokens"), (int, float))
        and not isinstance(row["context_tokens"], bool)
    ):
        if group_floor is None or length > group_floor * (1 + _CONTEXT_LENGTH_TOLERANCE):
            distinct += 1
            group_floor = length
    return distinct


def _unknown(node, key, reason):
    """Mark ``node[key]`` unknown, only where the skeleton already emitted a companion.

    The report schemas set ``additionalProperties: false`` on most objects, so inventing a
    ``_u_reason`` the skeleton did not emit fails validation for a cause the operator did
    not create -- exactly the laundering C2 exists to stop.
    """
    companion = f"{key}_u_reason"
    if companion in node:
        node[key] = None
        node[companion] = f"(U) {reason}"


def _measured(node, key, value, reason):
    """Set a measured value, or record why the measurement produced nothing."""
    if value is None:
        _unknown(node, key, reason)
    else:
        _known(node, key, value)


def _median_repetition(repetitions):
    """Pick the median repetition of a rung by output_tok_s, then by ttft_p95_s.

    Averaging percentiles across windows would publish a row no window ever exhibited; the
    figures in one row have to be mutually consistent, so a single real repetition stands
    in for the rung.

    Throughput alone cannot order the windows. Under ``ignore_eos`` with a declared output
    length every completed request emits exactly that many tokens, so a rung's repetitions
    tie on ``output_tok_s`` whenever they complete the same number of requests -- the normal
    case for a saturated rung, not an edge one. A stable sort then leaves submission order
    deciding, and "the median repetition" silently becomes "the second one submitted". On a
    GB200 multi-image ladder that published the second repetition at all nine rungs; at
    concurrency 32 the three windows measured ttft_p95 of 8.8817, 8.1633 and 8.6838 s and
    the report published 8.1633 -- the fastest window, offered to the reader as typical.
    Breaking the tie on ``ttft_p95_s`` makes the choice a median on the axis that moved.

    A repetition whose reduction produced no throughput figure at all is ranked with the
    others only if every repetition is in that state. ``None`` there means no completed
    record reported its output tokens, which is neither a fast window nor a zero one, and
    sorting it either way is a claim: at the top it becomes the median of a half-collapsed
    rung and publishes the best window as typical. A missing ``ttft_p95_s`` sorts last
    inside its throughput group for the same reason -- an unmeasured tail is not a short one.
    """
    ranked = [rep for rep in repetitions if rep.summary.output_tok_s is not None] or list(
        repetitions
    )
    ordered = sorted(
        ranked,
        key=lambda rep: (
            rep.summary.output_tok_s or 0.0,
            rep.summary.ttft_p95_s is None,
            rep.summary.ttft_p95_s or 0.0,
        ),
    )
    return ordered[(len(ordered) - 1) // 2]


def _counted(reps):
    """The repetitions a rung is graded on: the declared three, not the confirmation.

    The section 5 confirmation window is additional evidence about a boundary, never one of
    the repetitions the rung is scored from -- ``grade_rung`` partitions it out, and a row
    whose median came from a window the rung's own throughput median excluded would be a
    row no reader could reconcile with the grade beside it.
    """
    return [rep for rep in reps if not rep.post_search]


#: The four figures the SLO gate reads plus throughput: the figures a reader sizes from and
#: the figures a gate verdict turns on. A block over all twenty row figures would triple the
#: file for statistics nobody grades against.
_DISPERSION_FIGURES = (
    "ttft_p95_s",
    "itl_p95_s",
    "e2e_p95_s",
    "output_tok_s",
    "error_rate_pct",
)


def _figure_dispersion(reps, field):
    """Min, lower median and max of one figure across a rung's counted repetitions.

    A window whose reduction could not compute the figure left it null, and reading that
    null as zero would invent an endpoint no window measured; ``n`` counts the survivors so
    a spread taken over two windows is not published as a spread over three. The median is
    the lower median, index ``(n - 1) // 2`` of the sorted values -- the same convention as
    the row picker, so the two cannot drift into meaning different things.
    """
    values = sorted(value for rep in reps if (value := getattr(rep.summary, field)) is not None)
    if not values:
        return None, (
            f"(U) no counted repetition produced a {field}; every window's reduction left "
            "it null, and a zero here would read as a measured spread of nothing"
        )
    median = values[(len(values) - 1) // 2]
    entry = {
        "min": values[0],
        "median": median,
        "max": values[-1],
        "n": len(values),
    }
    if median == 0:
        # error_rate_pct is the figure where a zero median is the normal case, so this
        # branch is exercised by every healthy ladder, not by an edge one.
        entry["spread_pct"] = None
        entry["spread_pct_u_reason"] = (
            f"(U) the median {field} across the counted repetitions is zero; a relative "
            "spread against a zero median is a division by zero dressed as a statistic"
        )
    else:
        entry["spread_pct"] = round((values[-1] - values[0]) / median * 100.0, 2)
    return entry, None


def _dispersion(reps):
    """Per-figure spread across a rung's counted repetitions, or null with a reason.

    The published row is ONE window: its ttft, itl, e2e and throughput all come from the
    same 120 seconds of the system's life, so a reader can reason about them together. This
    block is per-figure across windows. It follows that ``row["ttft_p95_s"]`` need not equal
    ``row["dispersion"]["ttft_p95_s"]["median"]``, and usually will not: the row's window is
    picked once by throughput and then by tail latency, while each figure's median is taken
    over that figure alone. The alternative is worse -- a row assembled from per-figure
    medians would report a combination of latencies and a throughput that no window ever
    exhibited at the same time, and a reader dividing one by another would be computing a
    property of an imaginary system.

    Fewer than two counted repetitions means nothing was measured twice, so the rung gets an
    explicit null with a reason rather than a block of identical min, median and max --
    publishing ``spread_pct: 0.0`` there would report perfect stability as a finding, and
    omitting the key would make an unmeasured rung indistinguishable from one the harness
    simply never filled, which is the same absence that let the median defect sit unseen.
    The returned reason is untagged; ``_unknown`` adds the ``(U)``.
    """
    if len(reps) < 2:
        return None, (
            f"this rung had {len(reps)} counted repetition(s); a spread needs two windows, "
            "and publishing identical min, median and max would report perfect stability "
            "nothing measured"
        )
    block = {"repetitions_counted": len(reps)}
    for field in _DISPERSION_FIGURES:
        entry, reason = _figure_dispersion(reps, field)
        block[field] = entry
        if reason is not None:
            block[f"{field}_u_reason"] = reason
    return block, None


def _reason_for(summary, field, fallback):
    """Prefer the reducer's own account of a missing figure over a generic one.

    The reducer knows what a generic caller cannot: how many samples survived and which
    section sets the floor they missed. Its reasons already carry the ``(U)`` tag, which
    :func:`_unknown` adds, so it comes off here rather than being published twice.
    """
    stated = summary.reasons.get(field)
    if not stated:
        return fallback
    return stated[len("(U)") :].strip() if stated.startswith("(U)") else stated


def _window_of(runs, concurrency, repetition):
    """Find the WindowRun a graded repetition came from, or None if it is not in the bundle."""
    for run in runs:
        if run.policy.concurrency == concurrency and run.policy.repetition == repetition:
            return run
    return None


def _mean_reported(records, field):
    """Mean of the server's own token count over the records that carry one, else None.

    The adapter asks for ``include_usage`` and writes these straight from the response, so
    they are the only token counts in the run that were counted rather than declared. What
    the operator configured is a request, not a measurement: the synthetic corpus counts
    whitespace, so a declared 512 is 512 words and perhaps 650 tokens, and publishing it in
    a row tagged (M) understates the context every throughput figure in that row was
    produced at.
    """
    values = [
        getattr(record, field)
        for record in records
        if getattr(record, field, None) is not None and record.in_window
    ]
    return sum(values) / len(values) if values else None


def _mean_context(records):
    """Mean of the per-request context lengths the server counted, else None.

    C4 binds every throughput figure in a row to the context length beside it, so that
    length must be one some request actually occupied. mean(input_tokens) +
    mean(output_tokens) fails that: each mean is taken over whichever records happened
    to carry that count, so a record reporting only one side still votes in its half,
    and the sum of the two can be a context no request ever had -- yet it would be
    published in a row tagged (M). A record therefore contributes its input + output
    only when it carries both counts; a half-reported request contributes nothing.
    """
    contexts = [
        record.input_tokens + record.output_tokens
        for record in records
        if record.in_window and record.input_tokens is not None and record.output_tokens is not None
    ]
    mean = sum(contexts) / len(contexts) if contexts else None
    # An empty prompt with an empty completion can report 0 + 0, and a mean of those is
    # 0.0: a measured context of no tokens, which is not a context. The schema floor is
    # exclusive anyway, so it takes the same justified null as no measurement at all.
    return mean if mean else None


def _conformance_note(censor, result):
    """Write the note that stops ``non-conforming`` reading as a verdict on the hardware."""
    # The paragraph is conformance.DRAFT_NOTE rather than a second copy of the same prose:
    # `ascep conformance --raise` finds the text it may replace by an exact prefix match on
    # that constant, so a copy here would drift the first time either was edited, and the
    # raise would then silently leave every published note claiming to be ungraded.
    note = conformance.DRAFT_NOTE
    if censor:
        note += (
            f" The ladder was censored ({censor}), so every concurrency figure in this "
            "report is a lower bound on the hardware, not a finding."
        )
    if result.is_lower_bound:
        # A ladder that ran out of declared rungs before anything failed found the highest
        # concurrency it was *asked* about. Reported without this sentence it understates
        # the hardware, and the next person orders GPUs against it.
        note += (
            " The ladder was exhausted without a failing rung"
            + (f" ({result.censoring_cause})" if result.censoring_cause else "")
            + ", so the concurrency figures are a lower bound -- at least this much, not "
            "this much at most. Extend the declared rungs to find the boundary."
        )
    if result.cache_caveat:
        note += f" {result.cache_caveat}"
    return note


_TIER_FIELDS = (
    "max_concurrent_users",
    "max_tokens_per_s",
    "max_requests_per_s",
    "daily_requests",
)


def _boundary_constraint(result, concurrency):
    """Name the floor the ladder actually hit above ``concurrency``, or say nothing.

    Chapter 5 settles the label: a rung that failed its declared gates has observed which
    floor binds, and ``slo`` overrides the constraint label exactly there. A rung that
    delivered nothing collapsed on throughput, not on a latency gate. ABORTED rungs are
    excluded deliberately: they are failure evidence by cause, not evidence that a floor
    binds. A ladder exhausted without a failing rung has shown no floor at all -- its
    figure is "at least this much", and naming a constraint there would print a lower
    bound as a measured maximum.
    """
    failures = [
        rung
        for rung in result.rungs.values()
        if rung.concurrency > concurrency and rung.outcome is ladder.RungOutcome.FAILED
    ]
    if not failures:
        return None
    boundary = min(failures, key=lambda rung: rung.concurrency)
    return "throughput" if boundary.zero_completions else "slo"


def _observed_constraint(result, concurrency):
    """Name the floor observed AT ``concurrency``, falling back to the one above it.

    The measured tier can now sit on a failed rung, because chapter 5.5 ignores the gates
    when naming the engine ceiling. That rung has already shown which floor binds, so
    looking only above it would leave the tier's constraint null on the top rung of a
    ladder that failed there -- a report claiming "the ceiling is 128 streams" while
    declining to say what stopped it, which is the one thing a reader needs to size against.
    """
    rung = result.rungs.get(concurrency)
    if rung is not None and rung.outcome is ladder.RungOutcome.FAILED:
        return "throughput" if rung.zero_completions else "slo"
    return _boundary_constraint(result, concurrency)


def _fill_tier(tier, concurrency, rung, median):
    """Fill one capacity tier from a graded rung and its median repetition."""
    _known(tier, "max_concurrent_users", concurrency)
    _measured(
        tier,
        "max_tokens_per_s",
        rung.throughput_tok_s,
        "the graded rung carried no sustainable throughput figure",
    )
    _measured(
        tier,
        "max_requests_per_s",
        median.summary.requests_per_s,
        "the median repetition for this rung produced no request rate",
    )
    rps = median.summary.requests_per_s
    _measured(
        tier,
        "daily_requests",
        None if rps is None else rps * 86400.0,
        "the median repetition for this rung produced no request rate to extrapolate",
    )
    tier["provenance"] = "M"


def build_report(config, declarations, runs, repetitions, result, c8, censor):
    """Assemble the draft capacity report from measured rungs and declared documents.

    Bench fills only what it measured or was told; everything else stays null with a reason.
    Emitting keys absent would hand the operator a file that fails validation for causes
    they did not create, and filling them with estimates would launder guesses into
    measurements.
    """
    report = init.skeleton("capacity-report")
    # Bench measures; it does not normalise the operator's declarations on the way past.
    for layer in ("hardware", "model", "serving", "workload"):
        report[layer] = copy.deepcopy(declarations[layer])

    report["report_generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # The schema requires a claim, so bench makes the weakest one the enum offers; a
    # harness that grades its own results is the failure the negative corpus demonstrates.
    report["conformance"] = "non-conforming"
    report["conformance_note"] = _conformance_note(censor, result)

    window = config["window"]
    lad = config["ladder"]
    run_block = report["run"]
    _unknown(
        run_block,
        "engine_version",
        "bench observes an HTTP endpoint, not a build; the engine version was never "
        "visible to the load generator",
    )
    _measured(
        run_block,
        "container_digest",
        config["output"]["container_digest"],
        "the operator did not declare a container digest; bench pulls HTTP responses, "
        "not image registries",
    )
    warmups = [run.warmup_s_actual for run in runs]
    _measured(
        run_block,
        "warmup_seconds",
        sum(warmups) / len(warmups) if warmups else None,
        "no window ran, so there is no warm-up duration to average",
    )
    _known(run_block, "repeats", lad["repetitions"])
    _known(run_block, "sustained_window_seconds", window["window_s"])
    rung_list = [int(c) for c in lad["concurrency"]]
    _known(run_block, "concurrency_ladder", rung_list)
    _known(run_block, "drain_deadline_seconds", window["drain_deadline_s"])
    _known(run_block, "throughput_collapse_ratio", lad["throughput_collapse_ratio"])
    _known(run_block, "monotonic_across_ladder", result.monotone)
    # metrics.percentile implements the Hyndman-Fan type-7 estimator; naming any other
    # method here would make every percentile in this file unreproducible, so this string
    # is a statement about the code, not a guess.
    _known(run_block, "percentile_method", "hyndman-fan-type-7")
    _unknown(
        run_block,
        "tokenizer",
        "bench ships no tokenizer: the local token count required by chapter 4.7.1 was "
        "not taken, so the server's usage numbers are unchecked",
    )
    _known(run_block, "outlier_method", "none")
    _known(run_block, "open_loop", False)
    gates_node = run_block["slo_gates"]
    for gate_key in ("ttft_p95_max_s", "itl_p95_max_s", "e2e_p95_max_s", "error_rate_max_pct"):
        _measured(
            gates_node,
            gate_key,
            config["slo_gates"][gate_key],
            "no gate for this metric was declared in the bench config",
        )
    gates_node["declared_before_run"] = True
    for path_key in ("environment_capture_path", "raw_records_path", "engine_logs_path"):
        _measured(
            run_block,
            path_key,
            c8.get(path_key),
            "the reproduction bundle did not record this path",
        )

    # One row per rung, in ladder order: publishing only the winning rung would throw away
    # the shape of the curve, which is the part that says whether the tier is a plateau or
    # a cliff edge one request wide. A rung with no completed window has no evidence row
    # to publish, and the ladder's censoring cause explains its absence.
    row_template = copy.deepcopy(run_block["results"][0])
    rows = []
    # The chunk-gap fields sit with the ITL figures because they are the transport trace the
    # population label is about: a row whose ITL moved to per-request-mean is only auditable
    # if the factor and the observed chunk gaps ride on the same row.
    summary_fields = (
        "ttft_p50_s",
        "ttft_p95_s",
        "ttft_p99_s",
        "itl_p50_s",
        "itl_p95_s",
        "itl_population",
        "tokens_per_stream_chunk",
        "stream_chunk_gap_p50_s",
        "stream_chunk_gap_p95_s",
        "e2e_p95_s",
        "e2e_p99_s",
        "output_tok_s",
        "prefill_tok_s",
        "measured_input_output_ratio",
        "requests_per_s",
        "error_rate_pct",
        # Not a percentile and not a rate, so the median-by-throughput repetition picker
        # above already does the right thing: the row carries the peak the one chosen
        # window actually exhibited, never an average across windows that no window had.
        "peak_in_flight",
    )
    # These five are optional in the schema, so `ascep init` does not emit them and neither
    # does the row template. _unknown fills only a companion that already exists -- right for
    # a hand-filled report, wrong here: a transport or prefill figure the reduction computed
    # and found empty would vanish from the row, and an absent key reads as "this rung never
    # looked" when the truth is "it looked and the stamps were not there". Seeded so the (U)
    # has somewhere to land; _known pops the companion on every rung that measured a value.
    for name in (
        "tokens_per_stream_chunk",
        "stream_chunk_gap_p50_s",
        "stream_chunk_gap_p95_s",
        "prefill_tok_s",
        "measured_input_output_ratio",
        "dispersion",
    ):
        row_template.setdefault(f"{name}_u_reason", init.TODO)
    thin = []
    for concurrency in rung_list:
        rung = result.rungs.get(concurrency)
        reps = _counted(repetitions.get(concurrency, []))
        if rung is None or not reps:
            continue
        median = _median_repetition(reps)
        summary = median.summary
        row = copy.deepcopy(row_template)
        # The row is tagged (M), so its token counts have to be the server's, not the
        # config's. The declared figures are what was asked for; ignore_eos makes the output
        # side agree, but nothing makes the input side agree, and C4 binds every throughput
        # figure in this row to the context length beside it.
        window_run = _window_of(runs, concurrency, median.repetition)
        records = window_run.records if window_run is not None else []
        no_usage = (
            "the server returned no usage accounting for this rung, so the tokens these "
            "prompts actually cost were never counted; the configured value is a request, "
            "not a measurement, and publishing it in a row tagged (M) would launder it"
        )
        _measured(row, "context_tokens", _mean_context(records), no_usage)
        _measured(row, "input_tokens", _mean_reported(records, "input_tokens"), no_usage)
        _measured(row, "output_tokens", _mean_reported(records, "output_tokens"), no_usage)
        _known(row, "concurrency", concurrency)
        for field in summary_fields:
            _measured(
                row,
                field,
                getattr(summary, field),
                _reason_for(
                    summary,
                    field,
                    f"the window reduction produced no {field} for this rung; too few "
                    "completed samples survived exclusion",
                ),
            )
        # One number per rung is how a GB200 multi-image ladder published 2.4452 s of
        # ttft_p95 at concurrency 7, inside its 2.5 s gate, on a rung whose three windows
        # measured 2.2554, 2.4452 and 3.0255 and whose outcome was therefore failed. The
        # spread rides beside the row so that contradiction is legible, and so a reader
        # never compares two runs across a difference smaller than one run's own noise.
        dispersion, dispersion_reason = _dispersion(reps)
        _measured(row, "dispersion", dispersion, dispersion_reason)
        # A published figure between the two section 4.3 floors is real but one straggler
        # wide. It cannot be flagged in the row -- the schema has no field for it -- so it
        # goes in the assumptions table, which is where a reviewer looks for what the
        # numbers cannot settle.
        thin.extend(
            f"{field} at concurrency {concurrency}"
            for field in sorted(summary.low_confidence & set(summary_fields))
        )
        _measured(
            row,
            "slo_pass",
            summary.slo_pass,
            "the reducer could not grade this rung against the declared gates",
        )
        _known(row, "outcome", rung.outcome.value.lower())
        # A non-COMPLETE row must arrive with the sentence that produced the verdict:
        # without it, a rung can publish "failed" beside a passing slo_pass -- two keys
        # answering different questions -- and read as a harness contradiction, leaving the
        # operator to re-derive from records.jsonl what grading already knew.
        if rung.reasons:
            _known(row, "reasons", list(rung.reasons))
        _unknown(
            row,
            "gpu_util_pct",
            "a load generator cannot see the GPU; this must come from the serving host",
        )
        _unknown(
            row,
            "gpu_mem_util_pct",
            "a load generator cannot see the GPU; this must come from the serving host",
        )
        row["provenance"] = "M"
        rows.append(row)
    run_block["results"] = rows

    # Belt and braces for skeleton drift: any surviving TODO companion in the block bench
    # owns is an unknown with an honest (if generic) reason, never a fabricated value.
    for key in list(run_block):
        companion = f"{key}_u_reason"
        if companion in run_block and "TODO" in str(run_block[companion]):
            _unknown(
                run_block,
                key,
                "bench did not measure this; it is not observable by a load generator over HTTP",
            )

    # Fewer than three measured context lengths is a point, not a scaling curve. Setting the
    # flag does not raise the grade -- it states the limit the grade already reflects, and an
    # unlabelled single point reads as a curve to whoever builds on the draft. single_point is
    # a plain boolean with no _u_reason companion, so it is set directly, after the sweep
    # above so nothing clobbers it.
    #
    # Counted with a tolerance, and that is the whole of it. context_tokens is a per-rung MEAN
    # of measured lengths, so six rungs of one declared shape land on six distinct floats and
    # a set of them always has more than three members. A GB200 ladder at a single declared
    # 1,500-token shape measured 2043.65, 2043.94, 2045.28, 2045.46, 2045.48 and 2046.50 and
    # published single_point false -- claiming a context curve nobody measured, and silencing
    # the one C4 finding written for exactly that campaign. The flag was therefore unreachable
    # for every real bench run, which is the worst kind of escape hatch: present, documented,
    # and never once taken.
    run_block["single_point"] = _distinct_context_lengths(rows) < 3

    tiers = report["capacity_tiers"]
    # n_gpus is the topology the run was bound to in all four tiers, not a per-tier finding.
    n_gpus = declarations["serving"]["gpu_count"]
    for tier in tiers.values():
        _known(tier, "n_gpus", n_gpus)
        _unknown(
            tier,
            "binding_constraint",
            "bench exercises the throughput and latency floors only; the weights and KV "
            "floors are never evaluated, and a rung whose gates held still says nothing "
            "about whether KV would have bound first at another context length",
        )

    roofline_reason = (
        "the roofline needs the hardware's bandwidth and FLOPs and is computed by "
        "`ascep size`; bench measures latency, it does not model it"
    )
    for field in _TIER_FIELDS:
        _unknown(tiers["theoretical"], field, roofline_reason)
    policy_reason = (
        "a recommended tier derates a measurement by a headroom factor, and that factor "
        "is a policy choice, not a measurement; bench does not invent policy"
    )
    for field in _TIER_FIELDS:
        _unknown(tiers["recommended"], field, policy_reason)
    # Null provenance on an empty tier is a C1 error with no lawful answer: the schema has
    # no provenance_u_reason companion for the sibling fix, and "U" is the enum member that
    # means exactly that this row states nothing. The tag is the statement; inventing a
    # reason string beside it would only restate the tag less clearly.
    tiers["theoretical"]["provenance"] = "U"
    tiers["recommended"]["provenance"] = "U"

    # Chapter 5.5 defines the measured tier as "best observed, SLO ignored" -- the engine
    # ceiling. FAILED is a real operating point in the section 7 vocabulary ("a real negative
    # boundary"), so a rung that carried its load and missed a latency gate belongs here;
    # INVALID and ABORTED claim no operating point and stay out. Selecting on COMPLETE alone
    # published the highest PASSING rung as the ceiling, which collapses measured onto
    # sustainable and tells the reader the engine stops where the SLO stops -- erasing the one
    # distinction the two tiers exist to draw, and tripping C7 for saying it.
    observed_rungs = [
        c
        for c in rung_list
        if result.rungs.get(c) is not None
        and result.rungs[c].outcome in (ladder.RungOutcome.COMPLETE, ladder.RungOutcome.FAILED)
    ]
    if observed_rungs:
        top = max(observed_rungs)
        _fill_tier(
            tiers["measured"],
            top,
            result.rungs[top],
            _median_repetition(_counted(repetitions[top])),
        )
        # A ladder that stopped on a failed rung observed which floor binds, and chapter 5
        # settles the label there. Leaving the constraint null beside this number is a C5
        # error the run itself could have answered.
        constraint = _observed_constraint(result, top)
        if constraint is not None:
            _known(tiers["measured"], "binding_constraint", constraint)
    else:
        why = "no rung produced an operating point"
        if censor:
            why += f"; the ladder was censored ({censor})"
        for field in _TIER_FIELDS:
            _unknown(tiers["measured"], field, why)
        tiers["measured"]["provenance"] = "U"

    top_sustainable = result.max_sustainable_concurrency
    sustainable_rung = result.rungs.get(top_sustainable) if top_sustainable is not None else None
    # sustainable_publishable is the ladder's own gate on this figure -- confirmed by a
    # post-search repetition, monotone, and not harness-limited. Publishing the boundary
    # without it turns "the highest rung we happened to try" into "the highest rung that
    # works", which is the most flattering single sentence a capacity report can contain.
    if not result.sustainable_publishable:
        sustainable_rung = None
    if sustainable_rung is not None and _counted(repetitions.get(top_sustainable, [])):
        _fill_tier(
            tiers["sustainable"],
            top_sustainable,
            sustainable_rung,
            _median_repetition(_counted(repetitions[top_sustainable])),
        )
        # Same boundary, same rule as the measured tier: the first failed rung above is the
        # observed floor, and silence there is a C5 error the run could answer.
        constraint = _boundary_constraint(result, top_sustainable)
        if constraint is not None:
            _known(tiers["sustainable"], "binding_constraint", constraint)
    else:
        if censor is not None:
            why = f"the ladder was censored before a sustainable tier emerged ({censor})"
        elif top_sustainable is not None and not result.sustainable_publishable:
            why = (
                f"concurrency {top_sustainable} passed its gates but the ladder does not "
                "permit publishing it as a boundary"
                + ("" if result.confirmed else "; no post-search repetition confirmed it")
                + ("" if result.monotone else "; the ladder was not monotone")
                + (f"; {result.cache_caveat}" if result.cache_caveat else "")
            )
        elif result.terminated_at is not None:
            why = f"the ladder terminated at concurrency {result.terminated_at}"
        else:
            why = "no rung passed its declared SLO gates"
        for field in _TIER_FIELDS:
            _unknown(
                tiers["sustainable"],
                field,
                f"the ladder produced no sustainable tier: {why}",
            )
        tiers["sustainable"]["provenance"] = "U"

    roofline = report["roofline_comparison"]
    _unknown(roofline, "decode_tok_s_theoretical", roofline_reason)
    _unknown(
        roofline,
        "decode_tok_s_measured",
        "bench measures workload throughput, not an isolated decode rate at the "
        "roofline's operating point; the two are not interchangeable",
    )
    _unknown(
        roofline,
        "roofline_efficiency",
        "the ratio of two figures this run does not have: no theoretical roofline and no "
        "isolated decode measurement",
    )
    _unknown(roofline, "prefill_ttft_s_theoretical", roofline_reason)
    _unknown(
        roofline,
        "prefill_ttft_s_measured",
        "the TTFT bench measured mixes queueing with prefill; the roofline needs an "
        "isolated prefill measurement this run did not make",
    )

    sizing = report["sizing_result"]
    _unknown(
        sizing,
        "gpus_required",
        "sizing needs the declared demand targets and a headroom policy; `ascep size` "
        "computes it from those, not from this ladder",
    )
    _unknown(
        sizing,
        "replica_topology",
        "a replica topology is an output of sizing, not of measurement",
    )
    _unknown(
        sizing,
        "binding_constraint",
        "sizing binds only once demand and the roofline are known; this run has neither",
    )
    _unknown(
        sizing,
        "utilization_at_target_pct",
        "a bench run has no declared demand target to compute utilization against",
    )
    sizing["provenance"] = "U"

    # One topology is not a scaling curve, and the row shape requires a numeric
    # scaling_efficiency, so a single row could only be published as a fabricated 1.0.
    report["scaling"] = []

    assumption_template = copy.deepcopy(report["unmeasured_assumptions"][0])

    def _assumption(field, impact, cost):
        entry = copy.deepcopy(assumption_template)
        entry["field"] = field
        entry["impact_if_wrong"] = impact
        entry["cost_to_measure"] = cost
        return entry

    report["unmeasured_assumptions"] = [
        _assumption(
            "roofline_comparison",
            "Without a roofline, a measured throughput far from the hardware's ceiling "
            "is indistinguishable from a healthy one, and the real bottleneck goes "
            "uninvestigated.",
            "Run `ascep size` against the declared hardware block; it needs bandwidth "
            "and FLOPs from the declaration, not another bench run.",
        ),
        _assumption(
            "sizing_result (gpus_required and the headroom policy)",
            "A capacity decision made against the measured tier with no headroom leaves "
            "nothing for demand peaks; one made against an assumed headroom is policy "
            "dressed up as data.",
            "Declare the demand targets and run `ascep size`; the policy input is a "
            "capacity-planning decision, not a measurement.",
        ),
        _assumption(
            "scaling table (one topology only)",
            "A single topology says nothing about scaling efficiency, and extrapolating "
            "it to a node count this run never measured can halve or double a GPU order.",
            "Repeat the ladder at each tensor-parallel and pipeline-parallel degree of "
            "interest and fill one row per topology.",
        ),
        _assumption(
            "run.tokenizer (local token count not taken)",
            "Server-reported usage goes unchecked, so an engine that miscounts tokens "
            "inflates every tokens-per-second figure in this report.",
            "Count prompts and outputs locally with the served model's own tokenizer, "
            "as chapter 4.7.1 requires; bench ships no tokenizer to do it.",
        ),
        _assumption(
            "capacity_tiers.*.binding_constraint (weights and KV floors not evaluated)",
            "The named constraint is the floor observed under the declared gates. A "
            "weights or KV floor lower than it would not have been seen, so a deployment "
            "whose true floor is KV at another context length is still sized against the "
            "wrong resource, and the shortfall appears only in production.",
            "Inspect engine-reported KV cache occupancy and memory headroom on the serving "
            "host during the run, and repeat the ladder at longer context lengths; latency "
            "at one context length cannot decide the weights or KV floors.",
        ),
        _assumption(
            "run.results[].gpu_util_pct and gpu_mem_util_pct (not observed)",
            "Without utilization, a rung that failed its gates because the GPU was "
            "saturated is indistinguishable from one that failed because the client, the "
            "network or another tenant was, and the ladder's boundary gets attributed to "
            "the wrong component.",
            "Sample the serving host during each window -- nvidia-smi, DCGM, or the "
            "engine's own metrics endpoint -- and align the series to the window "
            "timestamps in the bundle; a load generator cannot see it over HTTP.",
        ),
        _assumption(
            "run.engine_version (not observed)",
            "Throughput moves by tens of percent between engine releases, so a report "
            "without the version cannot be compared with another run of the same model on "
            "the same hardware -- which is most of what these reports get used for.",
            "Read it from the serving host or the container image and record it; "
            "`/v1/models` does not carry it, so bench cannot ask.",
        ),
    ]
    if thin:
        report["unmeasured_assumptions"].append(
            _assumption(
                "figures published below the section 4.3 advisory sample floor: " + ", ".join(thin),
                "These percentiles are computed from enough samples to report but not "
                "enough to be stable: a single straggler moves them, so a tier boundary "
                "that turns on one of them can flip between two runs of the same config.",
                "Lengthen the window or raise the repetition count until the tail "
                "percentile clears its advisory floor; the sample count, not the "
                "estimator, is the limit.",
            )
        )

    reproduction = report["reproduction"]
    for key in (
        "run_configs_path",
        "raw_records_path",
        "engine_logs_path",
        "environment_capture_path",
    ):
        _measured(
            reproduction,
            key,
            c8.get(key),
            "the reproduction bundle did not record this path",
        )
    # Not the generic reason above. A missing path means the bundle failed to write an
    # artifact; a missing digest usually means there was no container to take one of -- a
    # bare-metal or conda run -- and bench cannot tell the two apart from a null in the
    # config. So it asserts no cause and states the consequence instead, which is the part
    # a reader needs: with no digest, nothing in the bundle pins the software except the
    # environment capture beside it, and that capture describes the host the harness ran on.
    _measured(
        reproduction,
        "container_digest",
        c8.get("container_digest"),
        "no container digest was declared for this run, so the software these figures were "
        "produced by is pinned only by the environment capture in the same bundle",
    )
    _unknown(
        reproduction,
        "analysis_script_path",
        "bench writes records and a manifest but no analysis script; every figure in "
        "this report is produced by `ascep bench` itself from the bundled records",
    )
    return report
