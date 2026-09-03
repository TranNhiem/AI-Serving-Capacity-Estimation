# Chapter 8 — Reporting and Conformance

An ASCEP report is the protocol's only product. This chapter defines the standard report, the conformance rules C1–C12 in operational terms, and the process by which a report earns its label of **full**, **partial**, or **non-conforming**. A conformance checker SHOULD be able to adjudicate every rule in this chapter from the report text and its reproduction bundle alone, without trusting the authors.

The report template lives at `templates/capacity-report.md` and MUST be used as the skeleton. Whether the report is prose or generated, every section of the template MUST appear, in order: the five layers (§1–§3, §4 results, §6 workload), the capacity tiers (§5), the unmeasured-assumptions ledger (§7), and the reproduction bundle (§8). Sections with no data are left in place with their fields set to `null` — deleting a section is a C1 violation, because a reader cannot distinguish "not measured" from "forgotten" if the heading is gone.

## Provenance tagging — the (M)/(I)/(T)/(U) discipline

Every numeric claim in the report MUST carry exactly one tag (C2). "Exactly one" is load-bearing: a number tagged both (M) and (T) lets a reader pick the flattering interpretation square it on the consistency axis. A checker tests C2 by finding untagged numerals in claims — the unit declarations in table headers do not need tags, the values do.

| tag | source of the number | correct use |
|---|---|---|
| **(M)** | observed in this campaign, in the raw records under `runs/` | "peak output throughput at `in=4096/out=512`: 18 400 tok/s (M)" |
| **(I)** | derived from (M) or (U) inputs by a *named* `ascep.capacity` function | "sustainable users = 35 000 (I, via `capacity_at`)") |
| **(T)** | roofline: `roofline_decode_tok_s`, `roofline_prefill_ttft_s` | "decode upper bound: 61 000 tok/s (T)" |
| **(U)** | not measured, not derivable from measured values | "duty cycle assumed 0.3 (U)" |

Tagging rules a checker enforces:

- A number copied from a *different* campaign or a vendor datasheet is (U), not (M), when used as a serving-relevant input. Peak HBM bandwidth from a hardware spec sheet MAY be reported untagged as a declared constant, since it is the input to the roofline, not a capacity claim; the roofline **output** is (T).
- An (I) value MUST name its function. "KV capacity 612 000 tokens (I, via `kv_capacity_tokens`)" is conforming; "KV capacity 612 000 tokens (I)" is not — a reviewer cannot recompute it, and unnamed formulas are how arithmetic errors survive review.
- (T) values MUST NOT appear in the conclusion as capacity. The roofline is an upper bound. A report whose headline says "up to 61 000 tok/s" citing the (T) row is making the exact claim the roofline chapter warns is not an expectation.
- When both exist, the engine-reported KV size (M) MUST displace the analytic estimate (I), with `calibrate_memory_utilization` SHOULD be used to reconcile them. Reporting the analytic number because it is larger is a provenance inversion.

**Failure prevented:** the classic failure is a capacity deck where "1.9 TB/s achieved" turns out to be DRAM bandwidth of the CPU host, the KV pool is computed, not measured, and the throughput comes from a different run. Tagging makes each substitution visible in the same sentence where it happens.

## Conformance rules, operationally

| rule | what a checker tests | typical violation | remedy |
|---|---|---|---|
| C1 — complete declaration | every `required` field in `schemas/*.schema.json` present; unmeasured = `null` + a §7 entry | the `gpu_memory_utilization` cell is blank, or the field is missing | fill it from the run config; it is a config value, so it can always be recovered — there is no legitimate reason a report gets to omit the one number that gates KV reproducibility |
| C2 — provenance tagging | every numeric claim has one tag | the §5 capacity table tagged, the prose summary untagged | tag prose, or strip numbers from prose and refer to the table |
| C3 — topology binding | every capacity/KV/throughput figure carries TP width, pipeline depth, GPU count | "per-GPU KV: 153k tokens" as a standalone line in §3 | restate as "153k tokens/GPU at TP=2 on 8 GPUs (M)"; per-GPU KV changes with TP because of KV-head replication, so an unbound number is silently wrong at any other width |
| C4 — context binding | every throughput figure carries input/output token counts; single points labelled as such; a ≥3-point curve SHOULD be present | a single "output tok/s" column with no shape column | rerun at a third context length, or label "single point at in=512/out=128 (M), context curve not measured" and move the risk to §7 |
| C5 — binding constraint | every capacity figure names its `Constraint` — `weights`, `kv`, `prefill`, `throughput`, `slo` | §5 table shows users but the constraint column says "n/a" | take the constraint from `capacity_at`'s return value; without it the reader cannot tell whether to buy memory or compute |
| C6 — four tiers | all four tiers reported in §5 | only "measured" appears | compute theoretical via `roofline_decode_tok_s`, derive recommended via `capacity_at(headroom=...)`; the four tiers differ by integer multiples, and a reader shown one will assume the most favourable |
| C7 — gates fixed before the run | gate thresholds appear in the published run config *and* in §4, and match | report gate says "TTFT p95 ≤ 2 s" but the committed config has no SLO block | the gates must be committed before measurement; if they were not, do it over — thresholds chosen after seeing p95 turns the sustainable tier into decoration |
| C8 — reproduction bundle | all §8 artefacts resolve and the config reproduces the run | raw records are aggregated CSVs, no per-request rows | publish per-request records; percentiles cannot be re-derived from averages, so post-hoc checking of any percentile claim is impossible without them |
| C9 — archetype corroboration | a declared `archetypes` entry is borne out by the workload's own numbers | an image_grounded archetype declared with every media count at zero or null | supply the media counts, or correct the archetype; a media workload priced with no media tokens has its KV floor computed on the text prefix alone and reports more sessions than the pool holds |
| C10 — agent-loop estimators | under a code_agent archetype, the context estimator and the KV residency are re-derived rather than inherited | `avg_context_tokens` tagged (I) with no note naming an accumulating estimator | cite `requests_per_session` or `context_growth_tokens_per_turn` in the workload notes, or measure the mean context and tag it (M); the chat estimator prices a single request against a transcript that accumulates every turn |
| C11 — mix-carried capacity | a capacity figure read off a benchmark rung has that rung's prefill priced, or its mix matches the declared one | the workload declares 3.5 input tokens per output token, the backing rung measured 1.0, and the rung has no `prefill_tok_s` | measure `prefill_tok_s` on that rung from the same window and records as `output_tok_s`; the throughput floor counts generated tokens only, so the published user count is overstated by exactly the ratio between the two mixes |
| C12 — repetition dispersion | every measured rung of a run declaring `repeats` ≥ 2 carries a `dispersion` block over the gate-bearing statistics, or `dispersion: null` with a `dispersion_u_reason` opening "(U) " | one window per rung published with no spread beside it; on one GB200 multi-image ladder the `ttft_p95_s` spread across the three counted windows ran from 0.78% at concurrency 64 to 31.49% at concurrency 7, and at concurrency 7 the published row read `ttft_p95_s: 2.4452` inside a 2.5 s gate with `slo_pass: true` beside `outcome: failed`, because one of its three windows measured 3.0255 s | publish the per-repetition min/median/max and spread for each gate-bearing statistic, or set `dispersion: null` with a (U) reason naming why fewer than two windows counted; without the spread a reader cannot reconcile a passing figure with a failing verdict, and can compare two reports on a difference smaller than either run's own noise |

A violation in C1–C5 alone downgrades the report to **non-conforming**; meeting C1–C5 but failing any of C6–C12 yields **partial**; meeting all twelve is **full**. C9, C10 and C11 sit outside the non-conforming set deliberately: all three grade the `archetypes` declaration, which is optional in this version of the protocol, and an optional declaration must not be able to sink a report that would otherwise pass. Promoting them is tied to `archetypes` becoming required, not to a judgement that their failures are mild — C11 in particular names an overstatement whose size it computes exactly. C12 sits outside the non-conforming set for a different reason: the `dispersion` block postdates every report the harness has emitted, so a fatal band would retroactively condemn reports written before the field existed, and a run that measured honestly but did not publish its spread is not in the class of one whose capacity arithmetic does not close.

## §7 — Unmeasured assumptions and the flip rule

§7 of the template is mandatory, not decorative. Every (U) in the report MUST recur there with four columns: the value used, the impact if it is wrong, and the cost to measure it. The "cost to measure" column exists to shame expensive assumptions: if a conclusion rests on a (U) that one afternoon of measurement would close, the reviewer's question writes itself.

**The flip rule.** If any report conclusion — the GPU count, the binding constraint, the sustainable/recommended gap — changes when a (U) value moves within its plausible range, the report MUST state this in §7, in the form "conclusion X holds only if assumption Y is at least/most Z." *Illustrative:* if the required replica count doubles when the assumed duty cycle (U) drops from 0.3 to 0.15, the report must say so; sizing commitments will be made on the stated number, not on its footnote. A checker tests this by sampling (U) entries and re-running `ascep.capacity` at the edges of each plausible range — the inputs are pure functions, so this is cheap, and reports SHOULD make §7 machine-readable to allow it.

## The reproduction bundle (C8)

The bundle MUST contain, at minimum: the run configs exactly as executed; per-request raw records (timestamps, prompt/output token counts, TTFT, ITL stream, status) sufficient to recompute every percentile in §4; engine version and the container image *digest* (a mutable tag is not evidence); and the environment capture (driver/runtime versions, node exclusivity). The §8 path table MUST resolve. "Available on request" is not publication and makes the report partial at best. The failure this prevents is the report that cannot be audited: with aggregates only, a reviewer can neither catch a percentile computed over a truncated window nor reproduce the crossover between the KV and throughput floors.

## Labels and comparability

| label | criteria | permitted use |
|---|---|---|
| **full** | C1–C12 all pass | comparable with any full or partial conforming report, declared fields permitting |
| **partial** | C1–C5 pass; at least one of C6–C12 fails | MAY be published, MUST be labelled partial, comparability caveated by the failed rule |
| **non-conforming** | any of C1–C5 fails, or the report distracts from its own tags | MUST NOT be compared against conforming reports, in either direction |

The last row is asymmetric on purpose. Comparing a conforming report to an untagged vendor figure cannot be made fair from one side only: the conforming side disclosed its KV pool and its gates; the other side's number has unknown context length, unknown utilization cap, unknown provenance. Any such pairing manufactures a conclusion from an uncontrolled gap in information. Reports that present non-conforming numbers alongside their own MUST NOT place them in the same table, and SHOULD label them "external, non-conforming, not comparable" wherever cited. A claim like "2× the published throughput of the alternative" built on one conforming and one non-conforming measurement is itself a conformance failure of the report making it.

A conforming report is therefore not a good number. It is a number a stranger can recompute, bound to the topology and context it was measured at, shipped with the constraint that binds it and the assumptions it secretly stands on. That is all ASCEP standardizes, and it is enough.
