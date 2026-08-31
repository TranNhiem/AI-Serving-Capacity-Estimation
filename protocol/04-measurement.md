# Chapter 4 — Performance Measurement

This chapter defines what a `run.schema.json` (layer 4) record **MUST** contain: precise metric definitions, percentile discipline, the visible/hidden token split on reasoning models, the open-loop/closed-loop distinction, utilization reporting, and per-request raw records with their validity and reconciliation rules. Every definition exists because the loosely-defined version is a known way to publish a number that cannot be reproduced. All measured figures in this chapter are tagged **(M)**; derived figures are **(I)** and **MUST** name the formula used.

## 4.1 Latency definitions

Ambiguous timestamps are the cheapest way to inflate a report. The following definitions are normative.

| metric | definition | MUST record |
|---|---|---|
| **TTFT** | time from request receipt to first generated token delivered to the client | per request, seconds |
| **ITL** (per-request) | mean inter-token latency over the request's decode phase | per request, seconds |
| **TPOT** | time per output token; synonym for ITL. A report **MUST** use one term consistently and **MUST NOT** mix them. | per request, seconds |
| **e2e latency** | request receipt to final token delivered | per request, seconds |

**ITL excludes the first token.** ITL **MUST** be computed as `(t_last − t_first) / (output_tokens − 1)` over decode steps only. A run that folds TTFT into per-token time conflates prefill with decode, hides queueing in the prefill stage, and produces a TPOT figure that looks flat while user-visible latency is not. The analytic lower bound on the prefill component is `roofline_prefill_ttft_s`; the gap between it and measured TTFT is queueing, scheduling, and tokenization — report it as such, not as compute time.

**Failure prevented:** a "20 ms token latency" claim that actually averages TTFT in, masking a 2 s prefill backlog at load.

## 4.2 Throughput definitions

| metric | definition |
|---|---|
| **output tokens/s** (`output_tok_s`) | generated (decode) tokens per second, cluster-wide |
| **total tokens/s** (`total_tok_s`) | input + output tokens per second |
| **requests/s** | completed requests per second over the measurement window |
| **goodput** | requests/s (or output tokens/s) measured over a window in which **every** SLO gate held; not defined for a window where any gate failed |

Reports **MUST** state which token count a tokens/s figure covers. Mixing them is a conformance-relevant ambiguity: at long input lengths, total tokens/s is dominated by prefill tokens that users never see, and quoting it as if it were generation speed overstates user capacity by (input + output)/output — often 10×.

Every rate in this section **MUST** be computed over the declared, post-warmup measurement window fixed by the benchmark procedure (chapter 7 §4) — never over the first-arrival-to-last-completion span implied by the records themselves. The record-implied span silently stretches with stragglers exactly when the system is saturated and slowest, so a 60 s window with one request finishing 40 s late reads as a 100 s span and reports req/s 40% low precisely at the load step where the number decides the tier.

Goodput is a property of the window, not a subset of its requests. Every gate in `slo_gates` is a window-level percentile or a window-level rate (§4.3), and no individual request meets or fails a percentile, so there is no per-request pass/fail to filter on; a definition that counts "requests that met the gates" is not computable from what this protocol declares. A report that wants a per-request filter **MUST** declare the per-request thresholds it used as its own stated extension and **MUST NOT** call the result goodput.

**Goodput** is the only throughput figure admissible as a *sustainable* tier input. Raw throughput from a window where p99 ITL breached its gate belongs to the *measured* tier and **MUST NOT** be passed to `capacity_at` with `slo_pass=True`. The same holds when the gate could not be evaluated at all: a gate whose statistic was unmeasurable is a failed gate for sustainable-tier purposes (§4.3), and throughput from such a run is measured-tier only.

### 4.2.1 Reasoning (chain-of-thought) models: visible vs hidden tokens

Reasoning models generate tokens the user never sees. Hidden reasoning tokens occupy decode steps, batch slots, and KV memory exactly like visible tokens — the engine does all the work, the user receives part of it. Any token count used for throughput, KV, or capacity arithmetic on a reasoning model **MUST** therefore split output into user-visible tokens and hidden reasoning tokens, and each downstream figure **MUST** use the side of the split that matches what it claims to describe:

| figure class | token count used |
|---|---|
| **engine-side**: `output_tok_s`, `total_tok_s`, KV growth, batch-slot occupancy | generated = visible + hidden |
| **user-facing**: user-visible latency and demand figures, capacity expressed in served user traffic | visible tokens only |

The engine-side rule exists because the engine is bound by generated tokens, not displayed ones: decode time per step, KV allocation per sequence, and scheduler occupancy all grow with the full stream. The user-facing rule exists because a throughput figure quoted against visible tokens is the demand a product can actually serve, and quoting generated tokens as user capacity overstates it by (visible + hidden)/visible. This sharpens, on reasoning models, the token-count ambiguity rule above and the C4 requirement that throughput claims carry their token counts: an output token count on a reasoning model is not a single number. It also feeds the KV arithmetic of Chapter 5 directly — context length per session in that arithmetic is input + *generated* tokens, and a visible-only output count shrinks the per-session KV footprint by exactly the hidden share.

If the framework cannot report hidden reasoning tokens, the generated-token figure **MUST** be labelled a lower bound and tagged **(U)** for the hidden component; the record then declares the reasoning-token field `null` with a **(U)** reason per C1. Unknown **MUST NOT** be treated as zero — a lower bound silently used as an exact count is how an engine bound by real decode traffic gets sized from the fraction of that traffic the user happens to see. **Failure prevented:** sizing a serving tier from 300-token visible replies when each reply really decodes 400 hidden + 300 visible tokens: real decode traffic and KV growth are 2.33× the figure used, and because per-user demand is the divisor in the throughput floor, the declared sustainable capacity is overstated by that same 2.33× on a thinking model.

## 4.3 Percentiles; the mean is not a capacity metric

Latency **MUST** be reported as **p50, p95, and p99** for TTFT, ITL, and e2e latency, each computed over the full measurement window (see Chapter 7 for window rules). Mean latency **MAY** be reported but **MUST NOT** be used in any gate.

The mean is useless for capacity because capacity is defined by the worst served user, not the average one. A load level can post a healthy mean while its p99 is dominated by occasional long prompts colliding with batch boundaries. Bimodal request mixes (chat + document ingestion) make this routine, not pathological: the mean sits between two modes and describes no real request. **Failure prevented:** sizing to mean latency, then discovering in production that 1 in 20 users exceeds the product's latency budget at your declared capacity.

SLO gates for the *sustainable* tier **MUST** be stated as percentile thresholds (e.g. "p99 ITL ≤ 100 ms") fixed before the run (C7).

A percentile is only comparable across sites if two analysts derive the same number from the same raw records, so the computation **MUST** name its interpolation convention: Hyndman–Fan type-7 linear interpolation on rank `p·(n−1)` (numpy's default `'linear'` method) is the protocol default, and any deviation **MUST** be declared. A percentile **MUST NOT** be reported at all unless the sample supports it — the absolute floor is `n ≥ 1/(1−p)`, so p95 requires ≥ 20 samples and p99 requires ≥ 100. Below that floor the percentile is recorded as unmeasured **(U)**, never substituted by the maximum or the mean: on a small sample the interpolated "p99" is just the observed maximum relabelled as a tail statistic, reading several times higher than the true tail on an unlucky window and flatteringly low on a lucky one. **Failure prevented:** from 23 requests, a report states "p99 ITL = 480 ms" (the worst observed request) and fails a 100 ms gate a healthy deployment would pass — or, on a lucky window, "p99 = 90 ms" passes a gate a sick deployment should fail; both figures are a max wearing a percentile label.

That floor is the point below which the figure is meaningless, not the point above which it is stable, and the two are far apart. At exactly `n = 1/(1−p)` the type-7 estimate interpolates between the top two order statistics — at `n = 100`, rank `0.99 × 99 = 98.01`, the 99th and 100th worst requests — so it is still the max-adjacent quantity the paragraph above condemns, merely one sample further from the edge. A single additional straggler moves it by the whole gap between those two observations. A percentile that decides a **sustainable-tier boundary** therefore **SHOULD** rest on `n ≥ 10/(1−p)` — ≥ 200 for p95, ≥ 1,000 for p99 — which puts ten observations in the tail rather than one, and a boundary figure taken between the two floors **MUST** carry the confidence interval of §4.8 so the width of that noise is visible. **Failure prevented:** two 100-request windows at the same rung report p99 e2e of 2.1 s and 4.8 s, the gate sits at 3 s, and the published sustainable concurrency moves by a factor of two run to run — entirely inside a sample size the protocol called sufficient.

The same discipline governs statistics that were never measurable in the first place. A metric or gate statistic that could not be measured **MUST** be reported as unmeasured **(U)** with its reason recorded per C1, and **MUST NOT** be replaced by a different statistic — not a mean for a missing p99, and not a span-over-count estimate presented as a true ITL distribution. For sustainable-tier (capacity) decisions, a gate whose statistic is unmeasurable **MUST** be counted as FAILED: an unmeasurable gate is never a pass, because treating "the framework could not expose p99 TPOT" as a pass is exactly how engine-ceiling runs get sold as gated user capacity. An honest gap is conforming; a guess is not. **Failure prevented:** a run reports its p99 ITL gate as "met" because the harness emitted no per-token timestamps, so no tail was measured at all, and raw throughput from that run is passed to `capacity_at` with `slo_pass=True` and published as sustainable capacity.

## 4.4 Open-loop vs closed-loop: two different quantities

These methodologies answer different questions, and a figure from one **MUST NOT** be compared with or substituted for a figure from the other.

| | **open-loop** (saturation) | **closed-loop** (virtual users) |
|---|---|---|
| setup | fixed input/output token counts, `ignore_eos`-style forced generation, request rate at or above saturation | realistic length distributions, generation ends freely at EOS, N virtual users with think time between requests |
| measures | **engine ceiling** — maximum tokens/s the scheduler can extract | **user experience** — latency and throughput at a stated offered load |
| maps to tier | **measured** (SLO ignored) | **sustainable** (gated) |
| answers | "how fast can this engine go?" | "how many users does this serve?" |

Open-loop runs suppress the dynamics that determine real capacity: variable output lengths, preemption under memory pressure, and KV turnover as sessions end. Closed-loop runs are load-dependent by construction — doubling virtual users changes every number. A report **MUST** declare its loop type per run, and MUST declare virtual-user count and think time for closed-loop runs. **Failure prevented:** publishing a saturation number produced with `ignore_eos` (which removes EOS-triggered batch shrinkage and often inflates steady-state batch size) as if it were supportable concurrent users.

## 4.5 GPU utilization

Two distinct counters, both **MUST** be recorded per window:

- **SM occupancy / compute utilization** (e.g. `% SM active`). For decode this **misleads**. Autoregressive decode is bandwidth-bound, not FLOP-bound (`roofline_decode_tok_s` models it as streaming weights + KV once per step): a kernel can occupy SMs while stalling on HBM. SM occupancy near 100% during decode does not indicate headroom is exhausted, and SM occupancy near 40% does not indicate 60% waste. It **MUST NOT** be used as evidence of efficiency in either direction; the normative efficiency figure is `measured ÷ theoretical` (roofline efficiency, §5 of the core spec).
- **GPU memory utilization** — two separate numbers, both required: (a) the framework's configured cap (`gpu_memory_utilization` / `mem_fraction_static` or equivalent), without which no KV measurement is reproducible; and (b) the engine-observed KV occupancy during the run. When the engine reports its KV pool size in tokens, that figure **MUST** be used in preference to the analytic memory model, and `calibrate_memory_utilization` **SHOULD** be run to reconcile the two before any projection to other context lengths.

## 4.6 Throughput vs context, and scaling

Output throughput is a function of input context length; a single figure is not. Per **C4**, any throughput claim **MUST** carry its measured input and output token counts, a curve over at least three context lengths **SHOULD** be reported, and a single point **MUST** be labelled as such. Projections between measured points **MUST** use `interpolate_throughput` (which clamps at endpoints); projections beyond the measured range **MUST** be tagged **(U)** — extrapolating a convex-decreasing curve can go negative, and any capacity derived from it via `capacity_at` is then unconstrained garbage.

Multi-GPU and multi-node throughput **MUST** be accompanied by `scaling_efficiency`, computed against the smallest measured configuration of the *same* model and precision. Per the protocol rule, an efficiency **above 1.0 is not superlinear scaling**: it means the baseline was degraded, almost always KV-starved (see `kv_heads_per_rank` for the replication mechanism). Such results **MUST** be flagged and the baseline rerun with adequate KV, not published as a speedup.

## 4.7 Per-request records

For every request, the run record **MUST** contain:

1. arrival timestamp and first-token timestamp
2. final-token timestamp
3. input token count (as tokenized, not characters)
4. output token count
5. per-token timestamps or, at minimum, per-request ITL
6. finish reason (EOS, length cap, abort, error)
7. segment identifier mapping the request into the aggregated window
8. on a reasoning model (§4.2.1), the hidden reasoning token count — with `0` meaning "the engine exposed the count and it was zero" and `null` tagged **(U)** meaning "the engine could not expose it"; the two values are distinct and **MUST NOT** be conflated, because zero is a measurement and unknown is a gap

Percentiles and goodput **MUST** be recomputed from these raw records, not copied from harness summary output: C8 requires the bundle to be independently re-analyzable, and a summary-only record makes every aggregate unverifiable. Requests with error finishes **MUST** be kept in the record and excluded from latency aggregates, with the exclusion count reported — silently dropping them inflates both throughput and tail latency quality. The exclusion rule extends to unmeasurable fields: a request whose harness emitted no per-token timestamps has per-request ITL *unmeasured*, recorded `null` tagged **(U)** per C1 — not estimated from `e2e ÷ output_tokens`, since a span average presented as an ITL sample is a substituted statistic and is barred by §4.3.

Timestamp records **MUST** be validated for monotonicity against the §4.1 definitions before aggregation: start, first-token, or last-token times preceding arrival by more than a small clock-skew slack (order 1 ms), or a last-token time preceding the first-token time, mark an invalid record. Client and server clocks are never perfectly synchronized and event-loop jitter is real, so honest sub-millisecond skew **MUST NOT** be rejected — but slack beyond ~1 ms **MUST NOT** be used either, because it smears the millisecond-scale TTFT and ITL measurements the validation exists to protect. Invalid records are handled like error finishes: kept in the record, excluded from latency aggregates, exclusion count reported; they **MUST NOT** be silently clamped to zero, since a negative TPOT folded into aggregates corrupts every percentile and rate computed from them. **Failure prevented:** clock skew producing ITL samples of −200 ms on some requests — silently clamped they bias p50 ITL toward zero; left in they make one request's implied "throughput" infinite and drag the mean-based summary off the page.

### 4.7.1 Token provenance and reconciliation

Per-request input and output token counts **MUST** be recorded twice: the engine's server-reported usage accounting, and a local count from a declared tokenizer, which the run record **MUST** name. The two counts measure different things, so they answer different questions: the engine-reported count is the normative figure for throughput and KV arithmetic, because it is what actually occupied batch slots and KV blocks, while the tokenizer count is the independent check that the usage accounting is honest. Server and tokenizer counts routinely disagree by a few percent through tokenizer fertility differences and usage-field bugs; any server-vs-tokenizer relative divergence greater than 5% **MUST** be surfaced as an explicit warning in the report, never silently resolved to one side, because a divergence beyond a few percent signals a tokenizer mismatch or a usage-field defect, and choosing a side quietly makes another site's KV floor different from yours.

If either side cannot be obtained — a deployment with no usage fields, or no accessible tokenizer for a proprietary model — the missing field is declared `null` with a **(U)** reason per C1, not copied from the side that exists: a copied count is not an independent measurement and would manufacture a zero divergence. The reconciliation check then simply cannot run, and the report says so. **Failure prevented:** a tokenizer that under-reads prompts by 30% turning a 32k-context workload into a 22k one, so that `sessions = kv_capacity / context_length` overstates concurrent-session capacity by 1.45× — an error invisible in the report because no divergence was recorded.

## 4.8 Statistical discipline: intervals on capacity boundaries

Capacity boundaries are run-to-run noisy, and a bare point estimate presents the full width of that noise as exact knowledge. Published capacity-boundary figures — max sustainable users, max sustainable load, and goodput at the largest passing cell — **SHOULD** therefore carry a confidence interval computed by a deterministic procedure over the raw per-request records: a percentile bootstrap with a stated seed and resample count is the reference form. Determinism is the operative requirement, not the specific method: C8's bit-identical re-analysis rule extends to intervals, so two sites handed identical records must obtain identical bounds, which rules out any unseeded resampling.

This is a **SHOULD**, not a **MUST**: no conformance rule checks for intervals, and honest existing reports remain conforming without them. But readers **SHOULD** treat an interval-free capacity number as carrying unknown noise, and report authors **SHOULD** treat the interval as part of the claim rather than decoration. **Failure prevented:** declaring "sustainable capacity: 240 concurrent users" where the honest statement is "180–310 users" — a buyer provisions at 240 and discovers the true boundary at load is 190, inside the noise band the report never showed.
