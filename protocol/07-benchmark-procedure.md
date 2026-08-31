# Chapter 7 — Standard Benchmark Procedure

This chapter defines the reproducible experiment behind Measurement, the fourth of ASCEP’s five layers. A benchmark MUST be documented so that another laboratory can reconstruct the hardware, model, serving, measurement and workload layers from the reproduction bundle required by **C8**.

The procedure has two audiences:

1. **Engine-capacity runs**, which find the best observed throughput. They MAY ignore SLOs during execution but MUST remain tagged Measured.
2. **Workload-validity runs**, which find operating points where every SLO gate holds for the full steady-state window. Their result is eligible for Sustainable and, after a headroom factor, Recommended.

Open-loop saturation results MUST NOT be reported as user capacity. A closed-loop workload run can only be interpreted against its declared think time, session duration and token distributions.

## 1. Declarations made before GPU time

Gate values, outlier treatment, repetitions and abort conditions MUST be declared before any timed request. This is **C7**.

| Pre-run declaration | Requirement | Failure prevented |
|---|---|---|
| Hardware and serving environment | Record every required field from `hardware.schema.json` and `serving.schema.json`; use `null` and `(U)` where unknown. | A result falsely attributed to one GPU, topology, clock or framework build. |
| Topology | Bind every claim to tensor-parallel width, pipeline depth and GPU count (**C3**). Use `kv_heads_per_rank` and `kv_bytes_per_token` for the analytic estimate. | Treating per-GPU KV capacity as constant when KV-head replication changes it. |
| Engine KV result | When available, record the engine-reported KV capacity and use it in preference to the analytic pool. Use `calibrate_memory_utilization` to reconcile it with the memory model. | Analytic projections disagreeing with engine capacity by multiples. |
| Tokenizer and workload | Record tokenizer identity/version, prompt source, language mix, input and output distributions, and whether request sizes are fixed or sampled. | Character or word targets silently changing after tokenization. |
| Gates | Declare all SLO thresholds, the percentile basis, minimum samples, passed-request restriction and full steady-state duration. | Selecting a threshold after seeing the percentile. |
| Outliers and failures | Declare exclusion tests and immediate-abort conditions. | Removing inconvenient tail samples after the run. |
| Repeat count | Declare repetitions before operation. ASCEP requires at least three complete, independent steady-state repetitions at every reported operating point. | A restart artifact or transient performance mode becoming the headline result. |

Preflight MUST verify required clocks, memory, interconnect visibility, GPU count, framework build, container digest, model revision, context-limit settings and free host resources. It SHOULD also check whether the node, network and accelerator are exclusively available. If exclusivity is not proven, it MUST be recorded; a capacity figure from a contended node retains the Measured tag only if that condition is disclosed.

Preflight SHOULD include a short diagnostic load sufficient to expose node-level problems, but its samples MUST be discarded from every reported appendix.

## 2. Workload construction

A run config MUST specify whether token counts are fixed or sampled.

- **Fixed-token runs** are acceptable for controlled engine characterization. They MUST be labelled as such because they omit variance in prefill and decode.
- **Sampled runs** SHOULD draw from the application workload or an explicitly declared deterministic distribution. Seed, sampler, distribution parameters and sampled sequences MUST be part of the reproduction bundle.

A sampled distribution changes results. Long-prompt mass raises KV traffic and TTFT; mass near the context limit may trigger queueing, preemption or rejection. Short fixed prompts therefore MUST NOT be projected directly to long contexts. Use measured points from at least three context lengths when building an `(I)` estimate with `interpolate_throughput`; otherwise the estimate remains `(U)` outside the measured range (**C4**).

Token quantities MUST be defined after tokenization, not before. Non-English text often changes tokenizer fertility — input tokens per character or word — and a prompt generator targeting “1,000 words” may produce materially different `input_tokens` across languages or tokenizers. Every request record MUST carry the engine or tokenizer-resolved input and output token counts.

For closed-loop runs, think time MUST be declared. Think time is the delay from receipt of a completed response to submission of the next request. Its distribution is part of the workload layer; zero is valid only if the product genuinely submits requests immediately. Omitting think time converts an interactive workload into an artificial arrival process and can saturate the server with fewer users.

## 3. Warm-up and cache control

Warm-up MUST occur before the steady-state window at the same topology, model revision, concurrency setting and token distribution. It has two purposes:

1. initialize kernels, allocator state, communication paths and admission control;
2. populate persistent execution structures such as graph captures or framework warm pools.

Warm-up MUST NOT be treated as evidence that an application cache is warm. Prefix, semantic or page caches MUST either be:

- disabled;
- cleared and controlled;
- exercised with uniquely generated prefixes; or
- explicitly declared as part of the application workload.

Repeated identical prompts can create cache hits unavailable in production and inflate both throughput and roofline efficiency. If cache policy is unknown, record `null` with a `(U)` statement.

The selected number of warm-up requests and duration MUST be declared. All warm-up samples MUST be discarded. The period SHOULD continue at least until engine counters and latency observations leave their initialization regime, but proximity-driven inspection MUST NOT decide where exclusion stops unless that selection rule was written beforehand.

## 4. Steady state

An operating point MAY proceed to steady state only after warm-up is complete and admission has reached the specified concurrency or arrival rate.

Once steady state starts:

- concurrency or arrival configuration MUST NOT change;
- model, engine or serving-runtime settings MUST NOT change;
- periodic diagnostic jobs MUST be identical to those declared for production-like runs;
- the window MUST cover a full sustained interval declared in advance.

To prove steady state, the run MUST divide the declared window into adjacent equal slices and report, for each slice, achieved concurrency, accepted request rate, completed request rate, error rate and aggregate output `tokens/s`. Entry and exit slices MUST be retained or removed only by a predeclared rule.

A run SHOULD demonstrate that completed-request and concurrency series do not have an unexplained monotonic trend incompatible with its arrest criterion. Presentation of slice tables is acceptable; a fabricated “high-confidence stable” label is not. Saturation itself is not a failure if the experimental intent is engine ceiling, but saturation MUST NOT be relabelled Sustainable unless every SLO gate passed across the full window.

## 5. Concurrency ladder

The ladder MUST be fixed before timing begins. It SHOULD be seeded from:

- `capacity_at`, using measured or engine-calibrated KV capacity;
- measured throughput interpolated by `interpolate_throughput`;
- `roofline_decode_tok_s` and `roofline_prefill_ttft_s` as upper or lower bounds, not as measurements;
- `gpus_required` when validating a recommended deployment.

The ladder SHOULD cover at least:

1. one clearly passing operating point;
2. one expected crossover or target operating point;
3. one clearly failing or saturating operating point.

That coarse sweep MAY be followed by a predeclared bisection to locate the largest Sustainable concurrency. Bisection MUST still perform full warm-up, window and repeats; it MUST NOT stop immediately after the first gate failure. At each operating point, throughput MUST be measured under the required context and topology (**C3**, **C4**).

Bisection assumes monotonicity — higher concurrency never improves gate outcomes — but real systems are not monotone: preemption thresholds, cache-state coincidence and batch-boundary effects can make a higher rung pass where a lower rung failed. The report MUST state whether pass/fail outcomes were monotone across the measured ladder, and bisection MUST NOT proceed over rungs whose results already contradict monotonicity: a binary search over a non-monotone function returns an arbitrary point wearing a boundary label.

When two probes at the same or adjacent rung disagree about pass/fail, first-answer-wins MUST NOT decide the rung. The rung MUST be re-measured with the predeclared independent repetitions; if the disagreement persists, the rung MUST be recorded as failing. The conservative reading is normative, not best-of-N or majority vote: capacity is defined by the worst served user, and an intermittently failing rung has gates that do not hold for some fraction of full windows — a majority vote sells exactly those windows as Sustainable.

The reported maximum sustainable concurrency MUST then be confirmed by at least one further independent repetition at that rung, run after the search completes and **in addition to** the three §6 requires at every reported operating point — never instead of them. That confirmation repetition MUST pass; if it does not, the conservative rule above applies and the rung is recorded as failing. The boundary is the one rung the search selected *because* it passed, so it is the rung most exposed to a favourable window, and a repetition taken after the stopping rule is no longer in play is the only evidence that the pass is a property of the system rather than of the search.

If the top rung of the ladder passed every gate, the maximum sustainable concurrency was not found: it lies above the ladder. The result MUST be reported as a lower bound — "sustainable concurrency at least N" — the report MUST state that the ladder was exhausted without failure, and the figure MUST NOT be presented as a measured maximum. A censored observation reported as a discovered boundary understates the system and hides that the experiment, not the engine, set the limit.

The censoring cause MUST be named, because the fixes differ. If the ladder simply ran out of rungs while the server was still healthy, the bound is a genuine server result awaiting a longer ladder, and the fix is more rungs. If instead the ladder was capped because the client harness, generator host or request supply could not offer more load, the server's sustainable concurrency was never probed at all: the result MUST be labelled harness-limited, and the required fix — more generator nodes or raised request supply — MUST be recorded so a reproducing laboratory does not inherit the same invisible ceiling. A harness-limited run MUST NOT enter the Sustainable tier as a boundary figure at any concurrency above the offered load.

## 6. Outliers, repeats and dispersion

Outlier handling MUST be declared before measurement; otherwise high tail samples can be removed after the percentile is known. An exclusion criterion MAY remove malformed client requests or independently proven instrumentation corruption, but MUST NOT remove high TTFT or ITL solely because it is large: the tail is the capacity signal, not noise.

Each statistic MUST name its population before timing, because latency, throughput and error rate are computed from different cohorts and mixing them changes the answer:

| statistic | requests that count | requests that do not |
|---|---|---|
| Latency — TTFT, ITL, e2e | valid completed requests whose arrival is inside the steady-state window, including valid completion after window close; §4.7 error/invalid handling still applies | error finishes, invalid timestamps, unmeasured fields, warm-up or preflight samples, invalidated-window samples |
| Throughput rates | completions or engine-accounted token emissions inside the declared fixed window only, under §4.2 and §4.2.1 | arrivals after the rate stop, warm-up traffic, invalidated-window traffic, extrapolated straggler completions, record-implied span stretching |
| Error rate | every request **issued** during the window, including requests refused at admission, with non-completion by the declared drain deadline, or abort, counted as failure | requests never offered to the system, client-side generation defects declared before timing, records from an invalidated window |

A straddler is assigned by its conservative event, not by convenience. Start-inside/finish-outside belongs to latency only after it completes validly; until then it is an error-rate risk, not a missing latency sample, because deleting the late request lowers the tail. Start-outside/finish-inside counts toward the fixed-window completion rate but not toward window-offered demand or latency, because crediting it as new demand inflates both users and speed. This split is the one that cannot flatter the result: slow finishers remain visible to failure logic, while rates keep the declared window span (§4).

A drain deadline MUST therefore be declared before timing: the wall-clock grace period after window close within which a straddler may still complete and count as a valid latency sample. Without it the two rules above contradict each other — a request that finishes 40 s after close is simultaneously a valid tail sample and a non-completion — and the harness resolves the contradiction whichever way flatters the run. A request completing inside the drain deadline is a latency sample and not an error; one still outstanding at the deadline is an error and contributes no latency sample. The deadline MUST be reported with the error rate, because "0.2% errors" at a 5 s drain and at a 120 s drain are different claims.

Issued, not admitted, is the normative denominator, and the choice only matters where it matters most: the two populations are identical below saturation and diverge exactly under overload, which is the regime the error rate exists to expose. A request refused at admission is a capacity failure the user experiences, so an admitted-only denominator lets a server that rejected a third of its offered load report itself error-free. A report MAY additionally give an admitted-only rate, labelled as such, but MUST NOT substitute it for the issued rate. If the error denominator cannot be determined or reconciled with issue and queue counters, error rate MUST be reported unknown with a `(U)` reason per **C1** and MUST NOT be written as `0.0`: a missing denominator silently becomes a perfect score. An unmeasurable latency statistic follows §4.3; it is failed for sustainable-tier purposes and is not invented here.

At least three independent repetitions MUST be executed at each reported operating point. A repetition MUST reset the declared initial state and MUST include its own warm-up, because reused allocator, cache or scheduler state masquerades as independent evidence. A window marked invalid under §7 is not one of the three; counting it as a repetition manufactures stability from a failed attempt.

A window marked invalid under §7 MUST contribute zero requests, tokens or seconds to any published median, percentile, throughput or error rate, while its raw records remain in the bundle for **C8**. The report MUST state the excluded record count, window bounds, trigger, affected statistics and reason, because silent exclusion and silent inclusion are indistinguishable to a reader and bias the result in opposite directions.

Reported dispersion MUST include minimum, median and maximum. Raw per-request records MUST be preserved; summary-only reporting does not meet **C8** because every aggregate becomes unverifiable.

A report SHOULD also state the relative spread and explain large differences. High dispersion may arise from preemption, KV exhaustion, sampling coincidence, interference, clock transitions or a non-stable arrival process. It MUST NOT be averaged away without diagnosis, because the mean can hide the mechanism that sets the next failure point.

## 7. Abort conditions

A ladder rung MUST end with exactly one outcome from this closed vocabulary, fixed before timing, so a true system limit is not laundered into an instrumentation excuse, nor a broken harness into a capacity boundary.

| Outcome | Use only when | Licenses |
|---|---|---|
| COMPLETE | windows done; telemetry evaluable; no abort. | (M) only; Sustainable needs every gate passed. |
| FAILED | record trusted; declared gate/error/collapse not met. | real negative boundary; never pass-by-omission. |
| INVALID | harness/clock/telemetry/environment defect. | no point claim; diagnostic evidence only. |
| ABORTED | terminate rule fired before required evidence. | failure evidence by cause; not an operating point. |

Observability is a precondition, not a debugging step. Before the first steady-state window, normally during §3 warm-up, the run MUST demonstrate that every SLO gate statistic, slice counter, abort signal and completion counter is emitted, timestamped and stored; entering steady state blind would make FAILED and ABORTED indistinguishable from unobserved. If this cannot be shown, do not start timing and mark the rung INVALID. If required telemetry stops mid-run, terminate, keep the partial record, and mark every affected slice or rung INVALID; a gate that could not be evaluated is not passed under §4.2.

The following MUST terminate the run and set FAILED, INVALID or ABORTED by the table above, because continuing after any of them measures damage control, not one configuration:

| Condition | Required action |
|---|---|
| Accelerator or host OOM, allocator collapse or engine OOM | ABORTED; record the exact counter or symptom and every active-configuration field. |
| Error rate above its predeclared threshold | FAILED, or ABORTED if the run stops; do not recursively retry without record. |
| Engine process exit, automatic restart or replica loss | ABORTED; preserve logs through the event. |
| Confirmed thermal or power throttling outside the declared profile | ABORTED or INVALID; aggregate clock and throttle telemetry MUST be retained. |
| Confirmed shared-node contention | ABORTED or INVALID under the rule declared before timing. |
| Model checksum, serving settings or topology changed mid-run | ABORTED; the run no longer measures one configuration. |
| Raw record loss or clock corruption | ABORTED, or INVALID for the affected slices. |
| Zero completions in a steady-state window under offered load | FAILED, with latency statistics **(U)**; terminate the ladder. |
| Throughput collapse at higher offered load | FAILED; terminate the ladder rather than climbing past queueing. |

Zero completion yields no latency statistic, so it MUST NOT be reported as a slow-but-valid point; the rung is a boundary where service fell over. Throughput collapse MUST use a fixed `throughput_collapse_ratio` declared before timing, never operator taste: mark FAILED when throughput falls below that ratio times the best lower COMPLETE rung at the same accounting and window while offered concurrency rises. The comparison is on throughput and not goodput deliberately, because collapse is a queueing failure that a rung can suffer whether or not its gates held, and goodput is not defined at a rung that failed them (§4.2) — testing collapse on goodput would make every gate failure indistinguishable from a collapse and terminate the ladder before the measured-tier ceiling is found. The ratio MUST NOT be below 0.5; stricter declared values MAY apply. Every rung above a collapsed one measures a queue, not a system.

A failed forced-run MUST NOT be represented as Sustainable. It can be reported as `(M)` evidence only under a failure section, without using it as a Sustainable operating point.

## 8. Required logging

The reproduction bundle MUST include:

- complete schemas or serialized objects for layers 1–5;
- run config, seeds, distributions and ladder;
- engine, core library and tokenizer versions;
- model identifier, revision and checksum where available;
- container digest and orchestration configuration;
- accelerator topology, memory and memory-utilization setting;
- engine-reported KV pool or cache capacity, when exposed;
- concurrency/arrival schedule, warm-up boundaries and steady-state boundaries;
- per-request identifier, issue time, queue time, TTFT, completion time, inter-token records or reconstruction data, input/output token counts, finish reason and error status;
- cache-hit and preemption counters when exposed, otherwise `null` and `(U)`;
- engine process events, restarts, health checks and abort reason;
- repeat index and all slice-level counters.

## 9. Numbered procedure

1. Freeze the workload, SLO gates, outlier policy, repeat count, abort thresholds and concurrency ladder.
2. Capture the complete environment and validate against the layer schemas.
3. Select topology; calculate only `(I)` planning values with named `ascep.capacity` functions.
4. Start the serving process and record engine startup output, KV reporting and topology.
5. Run a discarded preflight diagnostic and verify exclusivity/telemetry.
6. Build or sample prompts; resolve all token counts through the declared tokenizer.
7. Configure memory, task queue and KV stenography. Confirm numerical and reproducibility settings.
8. Clear or deliberately populate caches according to the declared policy.
9. Run the full warm-up; verify initialization completion and discard every sample.
10. Enter steady state at one ladder point without changing configuration.
11. Collect the full steady-state window and required counters.
12. Apply predeclared abort and validity rules immediately.
13. Repeat independently at least three times while retaining independent raw records.
14. Move to the next declared ladder point and repeat from warm-up.
15. For SLO runs, select only points that pass every gate for every required full window as Sustainable.
16. Apply a declared headroom only in the Recommended tier, using `capacity_at`.
17. Publish the report, all raw records, environment capture and validity checklist.

## Run-validity checklist

- [ ] All required layer fields are present, with unknowns recorded as `null` and `(U)` (**C1**).
- [ ] Every numeric estimate carries `(M)`, `(I)`, `(T)` or `(U)` (**C2**).
- [ ] Tensor-parallel width, pipeline depth and GPU count accompany every KV, throughput or capacity figure (**C3**).
- [ ] Input/output token counts and context distribution accompany every throughput figure (**C4**).
- [ ] Warm-up, cache policy, steady-state boundaries and slice evidence are declared.
- [ ] Distributions, tokenizer, language mix, think time and output policy are declared.
- [ ] SLO gates and outlier policy predate the measured percentile (**C7**).
- [ ] At least three independent repetitions and their dispersion are reported.
- [ ] Failure and abort handling was applied exactly as declared.
- [ ] The largest selected Sustainable window passed every SLO gate for its full duration.
- [ ] The bundle contains per-request records and environment material sufficient for reproduction (**C8**).
- [ ] Measured ÷ theoretical roofline efficiency was calculated and any value at or above 1.0 was investigated rather than published as a performance win.
- [ ] Every reported capacity names `weights`, `kv`, `throughput` or `slo` as its binding floor (**C5**), and theoretical, measured, sustainable and recommended values remain distinct (**C6**).
- [ ] Every ladder rung carries exactly one declared outcome — COMPLETE, FAILED, INVALID or ABORTED — and a true capacity limit retains the FAILED label rather than being softened into INVALID.
- [ ] A window with zero completions or a collapsed throughput rung terminated the run per the declared abort conditions, rather than letting the ladder climb past the collapsed rung.
- [ ] Observability of every abort condition and SLO gate was demonstrated during warm-up, before the first steady-state window; mid-run telemetry loss invalidated the affected rung.
- [ ] Latency, throughput and error-rate figures are computed over their fixed declared sample populations under the stated straddler rule; an error rate that could not be determined is reported unknown, never 0.0.
- [ ] A drain deadline was declared before timing and is reported alongside the error rate, so a straddler is either a valid late latency sample or a non-completion — never scored as both or as neither.
- [ ] Samples from every invalidated window are excluded from all published statistics, with the exclusion count and reason stated.
- [ ] Monotonicity across the ladder was declared; contradictory probes were re-measured under one stated resolution rule; a confirmatory probe passed at the reported maximum sustainable concurrency.
- [ ] A ladder exhausted without failure is reported as a lower bound (“at least N”), never as a measured maximum, and names its censoring cause (server not saturated vs client-limited).
- [ ] The percentile convention (§4.3) and the tokenizer used for the independent token count (§4.7.1) are declared in the run config.

