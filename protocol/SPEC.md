# ASCEP — AI Serving Capacity Estimation Protocol

**Version 0.1 (draft).** Normative specification.

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are to be interpreted as in
[RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## Purpose

ASCEP standardizes how to answer one question:

> Given a hardware configuration, a model, a serving framework and an application workload,
> how much capacity can this infrastructure support, and how much infrastructure does a
> target workload require?

It is a *measurement and reporting* protocol, not a benchmark suite and not a leaderboard.
Two ASCEP-conforming reports for different models on different hardware are comparable
because they declare the same fields, apply the same gates, and tag every number with its
provenance. That comparability — not any particular score — is the deliverable.

## Why a protocol is needed

Published serving benchmarks are routinely non-reproducible for reasons that are cheap to
fix and expensive to discover later:

- **Headline throughput with no context length.** Tokens/s falls 2-4× from chat-length to
  document-length prompts. A single number is not a property of a model.
- **KV capacity quoted as a per-GPU constant.** It is not. It varies with tensor-parallel
  width, sometimes non-monotonically, because KV heads are replicated once TP exceeds the
  model's KV head count. A per-GPU figure without its topology is unusable.
- **`gpu_memory_utilization` unrecorded.** Without it a KV measurement cannot be reproduced,
  and analytic projections silently disagree with it by multiples.
- **Open-loop benchmarks reported as user capacity.** Fixed-token, `ignore_eos` saturation
  runs measure engine ceiling, not what users experience. They are a different quantity.
- **Latency percentiles absent or averaged.** Capacity is set by the tail, not the mean.
- **Superlinear scaling reported as a win.** Efficiency above 1.0 means the baseline was
  degraded, usually KV-starved.

ASCEP makes each of these a declared field, a required gate, or a conformance failure.

## Conformance

A report is **conforming** if all of the following hold.

**C1 — Complete declaration.** Every field in `schemas/*.schema.json` marked required is
present. Unknown or unmeasured values MUST be recorded as `null` with a `(U)` entry in §7 of
the report, never omitted and never guessed.

**C2 — Provenance tagging.** Every numeric claim MUST carry exactly one of **(M)** measured,
**(I)** inferred by a named formula, **(T)** theoretical roofline, **(U)** unmeasured
assumption. An untagged number makes the report non-conforming.

**C3 — Topology binding.** Any capacity, KV or throughput figure MUST be reported together
with the tensor-parallel width, pipeline depth and GPU count it was measured at. Per-GPU
figures MUST NOT be presented as topology-independent.

**C4 — Context binding.** Any throughput figure MUST be reported with the input and output
token counts it was measured at. A throughput curve over at least three context lengths
SHOULD be reported; a single point MUST be labelled as such.

**C5 — Binding constraint.** Every capacity figure MUST name which floor binds it —
`weights`, `kv`, `throughput` or `slo`. A capacity number without its constraint does not
say what to buy and is not actionable.

**C6 — Four tiers.** Capacity MUST be reported as theoretical, measured, sustainable and
recommended (§5). Reporting only one tier is non-conforming, because the four differ by
integer multiples and readers will assume the most favourable.

**C7 — SLO gates declared before measurement, and honoured inside the envelope.** Gate
thresholds MUST be fixed in the run config before the run, not chosen after seeing results.
An operating point whose gates failed MUST be excluded from the sustainable tier when that
point lies at or below the context length the capacity claim covers. A failure *above* that
context does not invalidate the tier — but the report MUST then state the envelope alongside
the figure, because "460 users" and "460 users at ≤2,000 tokens" are different claims and
only one of them is supported (§5.5).

**C8 — Reproduction bundle.** Run configs, per-request raw records, engine version and
container digest, and the environment capture MUST be published with the report.

A report meeting C1-C5 but not C6-C8 MAY be labelled **partial**. Anything less is
**non-conforming** and MUST NOT be compared against conforming reports.

## The five layers

ASCEP separates concerns that are routinely conflated. Each layer has a schema and MUST be
declared independently, because each varies independently.

| layer | schema | question |
|---|---|---|
| 1 · Hardware | `hardware.schema.json` | what silicon, how connected |
| 2 · Model | `model.schema.json` | what weights, what precision, what attention |
| 3 · Serving | `serving.schema.json` | what framework, what parallelism, what batching |
| 4 · Measurement | `run.schema.json` | what was observed, under what gates |
| 5 · Workload | `workload.schema.json` | what the application actually asks for |

Capacity is a *function of all five*. Any statement of the form "model X does N tokens/s" is
under-specified by four layers and ASCEP treats it as meaningless.

## Capacity tiers

Reports MUST distinguish these and MUST NOT interchange them.

| tier | definition | source |
|---|---|---|
| **Theoretical** | hardware roofline: bandwidth-bound decode, FLOP-bound prefill | `ascep.capacity.roofline_*` |
| **Measured** | best observed in benchmark, SLO ignored | measurement |
| **Sustainable** | measured, restricted to operating points where every SLO gate held for the full window | measurement + gates |
| **Recommended** | sustainable ÷ headroom factor | `capacity_at(headroom=...)` |

The ratio measured ÷ theoretical is the **roofline efficiency** and MUST be reported. Real
servers land well below 1.0. A value at or above 1.0 indicates a measurement error — most
often a mis-declared active-parameter count or an untracked cache hit — and MUST be
investigated rather than published.

## The three floors

Capacity is the **minimum** of three independent floors, never the average and never the
most convenient:

1. **Weights floor.** Do the weights plus a usable KV pool fit? Binary. A configuration that
   loads but leaves near-zero KV is not viable — it serializes to batch-size-1.
2. **KV floor.** `sessions = kv_tokens ÷ avg_context_tokens`. Binds at long context.
3. **Throughput floor.** `users = usable_tokens_per_s ÷ per-user demand`. Binds at short
   context, where KV is abundant and compute is not.

Which floor binds **changes with context length**. Reports MUST state the crossover, or state
that it was not determined.

## Formulas

All estimation formulas live in `ascep/capacity.py`, are pure, unit-annotated, and
individually testable. Reports citing an **(I)** value MUST name the function that produced
it. The protocol prefers a measured value over an analytic one wherever both exist: in
particular, when the engine reports its KV cache size, that figure MUST be used in preference
to the analytic memory model, and `calibrate_memory_utilization` SHOULD be used to reconcile
the two before projecting to other context lengths.

## Non-goals

- **Not a quality benchmark.** ASCEP measures capacity, not answer quality. A model that
  serves 10,000 users badly is not a success, and ASCEP will not tell you that. Pair it with
  a task-appropriate quality evaluation.
- **Not a procurement recommendation.** ASCEP produces GPU counts, not vendor choices or
  prices. Cost modelling is left to the reader because pricing is regional and volatile.
- **Not a training-capacity protocol.** Inference serving only.

## Chapters

| # | chapter | covers |
|---|---|---|
| 1 | [Hardware](01-hardware.md) | declaration, interconnect, single vs multi-node |
| 2 | [Model](02-model.md) | dense vs MoE, precision, KV geometry, attention type |
| 3 | [Serving](03-serving.md) | framework, parallelism, batching, KV configuration |
| 4 | [Measurement](04-measurement.md) | TTFT, ITL, throughput, utilization, percentiles |
| 5 | [Capacity model](05-capacity-model.md) | the formulas, the tiers, the floors |
| 6 | [Application sizing](06-application-sizing.md) | users → tokens → GPUs |
| 7 | [Benchmark procedure](07-benchmark-procedure.md) | warm-up, duration, repeats, outliers, failures |
| 8 | [Reporting](08-reporting.md) | the standard report, conformance checking |

## Versioning

ASCEP uses semantic versioning. A change that would alter a conforming report's numbers is a
major bump. Reports MUST cite the protocol version they were produced under.
