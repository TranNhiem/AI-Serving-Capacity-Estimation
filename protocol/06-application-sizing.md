# Chapter 6 — Application-Level Sizing

Layer 5 is the one that turns a product question ("we have 10,000 daily users") into an infrastructure question ("how many GPUs, and which floor binds"). It is declared in `workload.schema.json`, consumed by `ascep.capacity.capacity_at` and `ascep.capacity.gpus_required`, and it is where the most expensive errors happen, because every upstream measurement is multiplied by whatever this layer asserts.

## The sizing chain

Sizing runs forward through the five layers and back again:

```
Hardware → Model → Serving → Performance → Workload → Infrastructure
```

Layers 1–3 are fixed by declaration. Layer 4 (measurement) produces the two per-GPU quantities this chapter consumes: KV tokens and aggregate `tokens/s`, each measured **at the target context length and the intended TP width** — per C3 and C4, both are topology- and context-bound. Layer 5 (workload) produces the demand. The answer is `min` of the KV and throughput floors, rounded up to whole replicas by `gpus_required`.

**Rule (MUST).** Both per-GPU inputs to `gpus_required` MUST come from measurement at the same TP width and the same context length as the target workload. *Failure prevented:* feeding short-context throughput into a document-length workload overstates capacity by 2–4×; feeding TP=1 per-GPU KV into a TP=4 plan ignores the replication penalty in `kv_heads_per_rank` and can halve real capacity.

## From DAU to concurrent users

Product owners speak in daily active users. GPUs serve concurrent sessions. The bridge is Little's law, `L = λ·W`, with a peak-to-mean correction — exactly what `Workload.peak_concurrent_users` implements:

```
arrivals_per_s      = DAU × sessions_per_user_per_day / 86400
peak_concurrent     = arrivals_per_s × avg_session_seconds × peak_to_mean
```

**Rule (MUST).** `peak_to_mean` MUST be declared and MUST NOT default silently. A flat 1.0 load is almost never real; 3–6 is typical for single-timezone consumer traffic. *Failure prevented:* sizing to the daily mean under-provisions the peak hour by the whole ratio — the system fails precisely when it is most visible.

**Rule (MUST).** `peak_concurrent_users` and `active_sessions` MUST both be declared, and MUST NOT be conflated: the second is the first multiplied by `duty_cycle`. Both are `(I)` fields in `workload.schema.json` precisely so that the duty cycle is visible as a number in the artifact rather than buried inside somebody's arithmetic. A user reading a reply is logged in but occupies no KV slot and generates no tokens. *Failure prevented:* omitting the duty cycle inflates both the KV floor and the throughput floor by `1/duty_cycle`. For chat workloads with long pauses between turns this is routinely 2–5× of pure overprovisioning. Conversely — and this is the subtler trap inside `capacity_at` — per-user demand MUST be derived from `demand_tok_s() / peak_concurrent_users()`, not from the raw per-stream target; using `target_tok_s_per_user` directly against concurrent users silently re-inflates demand by the same factor.

## `avg_context_tokens`: the highest-leverage input

`Workload.avg_context_tokens()` approximates a session mid-generation as:

```
avg_context_tokens = input_tokens_per_request + output_tokens_per_request / 2
```

One session's KV footprint moves linearly in this number, so the **entire KV floor moves linearly in it**. No other single workload field shifts the answer as far. A chat workload at 500 tokens of context and a support-agent workload at 8,000 tokens of context can have identical DAU, identical sessions, and a KV floor 16× apart.

**Rule (MUST).** If real conversation-length data exists, `avg_context_tokens` MUST be overridden with it, and the provenance tagged (M). The default formula is an (I) assumption and MUST be tagged as such. *Failure prevented:* a 2× error in context length is a 2× error in the KV floor, invisible until the cluster saturates in production.

## Per-stream rate targets

Where a product needs a minimum per-stream generation speed, declare it in `target_tok_s_per_user`. Common anchors:

- **Reading speed.** Sustained prose reading is on the order of 5–15 tokens/s depending on language and content; a stream slower than the reader stalls attention, faster is invisible.
- **TTS underrun.** A text-to-speech pipeline consuming a stream needs enough tokens/s to never starve its audio buffer; this is often the binding target for voice agents and is set by the TTS engine's consumption rate, not by human preference.
- **Interactive minimum.** Product-derived latency budgets (e.g. first reply rendered within N seconds) translate into a rate floor via `output_tokens_per_request / budget_s`.

**Rule (SHOULD).** `target_tok_s_per_user` SHOULD be justified by one named anchor (reading speed, downstream-consumer rate, or an interaction budget), not picked. *Failure prevented:* an arbitrary target doubles as an un-auditable sizing multiplier.

## Worked example (a): 10,000-DAU chatbot → GPU count

*All numbers below are illustrative; none are measured.*

**Workload.** DAU = 10,000; sessions/user/day = 2; avg session = 600 s; `peak_to_mean` = 4; `duty_cycle` = 0.4; input = 1,000 tokens; output = 400 tokens; requests/session = 5; `target_tok_s_per_user` = 0 (rate derived from session volume).

```
peak_concurrent  = (10,000 × 2 / 86,400) × 600 × 4  ≈ 556 users
active_sessions  = 556 × 0.4                          ≈ 222
avg_context      = 1,000 + 400/2                      = 1,200 tokens
demand_tok_s     = 556 × (400 × 5) / 600              ≈ 1,852 tok/s
```

**Per-GPU inputs (claimed to be (M) at the target TP width and ~1,200-token context).** KV per GPU = 180,000 tokens (I); throughput per GPU = 900 tok/s (I); TP width = 2.

Calling `gpus_required(...)` with `headroom = 1.15`, `gpus_per_replica = 2`:

| replicas | GPUs | KV floor users (÷1.15) | throughput floor users (÷1.15) | meets 556? |
|---|---|---|---|---|
| 1 | 2 | 2×180,000/1.15/1,200/0.4 ≈ 652 | 2×900/1.15/(1,852/556) ≈ 469 | no — throughput |
| 2 | 4 | ≈ 1,304 | ≈ 939 | yes |

**Answer: 4 GPUs (2 replicas of TP=2), RECOMMENDED tier, binding constraint `throughput`.** Note the headroom divides *both* floors inside `capacity_at` before the `min` — it is not applied after.

*Failure this walkthrough prevents:* with `peak_to_mean` dropped to 1.0 and duty cycle dropped, demand collapses to ~139 concurrent and ~463 tok/s, and the same arithmetic answers "1 replica" — a 2× under-buy hidden in two small defaults.

**The same workload against real measurements.** [`examples/chatbot-10k-dau`](../examples/chatbot-10k-dau/) publishes this exact declaration as a validating `workload.json` and re-runs it against H100 figures measured in [`examples/moe-26b-h100-tp2`](../examples/moe-26b-h100-tp2/) — 574,798 KV tokens per GPU and 1,459 tok/s per GPU interpolated to a 1,200-token context. The answer becomes **2 GPUs**, still throughput-bound. The placeholder figures above are slower than the real hardware, and the gap between "4 GPUs" and "2 GPUs" on identical arithmetic is why layers 1–4 are measurements you supply rather than constants this protocol ships.

## Worked example (b): 8 GPUs and a model → concurrent users

*Illustrative.* 8-GPU node, model deployed at TP=4 (two replicas). Engine reports 350,000 KV tokens per replica (M — and per the spec, the engine-reported figure MUST be used over the analytic `kv_pool_bytes` path; `calibrate_memory_utilization` SHOULD reconcile the two). Measured aggregate throughput at the workload's context length: 2,600 tok/s per replica (M).

Workload: input = 4,000 tokens; output = 800 tokens; `duty_cycle` = 0.5; `target_tok_s_per_user` = 12 tok/s (reading-speed anchor).

```
avg_context = 4,000 + 400   = 4,400 tokens
cluster KV  = 2 × 350,000   = 700,000 tokens
cluster thr = 2 × 2,600     = 5,200 tok/s
users_kv    = 700,000/4,400/0.5            ≈ 318
users_thr   = 5,200/(12×0.5)               ≈ 867
```

`capacity_at(n_gpus=8, ...)` returns `max_concurrent_users ≈ 318`, binding constraint **`kv`**, tier MEASURED. Buying more compute would change nothing; the next dollar belongs to KV (wider-but-not-replicating topology, KV quantization, or shorter contexts). The RECOMMENDED tier divides this by the chosen headroom factor.

*Failure prevented:* quoting only `users_thr` — the number a short-context open-loop benchmark would suggest — overstates user capacity by ~2.7× and buys accelerators that sit idle behind a KV wall. C5 exists exactly for this.

## Workload segmentation

Real applications are mixtures, and the mixture changes which floor binds per segment:

| segment | example | typical context | binding floor |
|---|---|---|---|
| short-context | autocomplete, classification, chat turns | < 2k tokens | `throughput` |
| mixed | RAG chat, support agents | 2k–16k | crossover — MUST be reported per the three-floors rule |
| long-context | document QA, code analysis | > 16k | `kv` |

**Head-of-line blocking.** On a shared engine, a long prefill is compute-heavy and occupies the scheduler; short requests queued behind it see their TTFT inflated by seconds. The aggregate-throughput number stays healthy — the open-loop graph looks fine — while the short-context SLO dies in the tail. This is why C7 requires gates fixed per segment before measurement, and why Measurement-layer percentiles MUST be segmented, not pooled.

**Rule (SHOULD).** When a workload mixes segments whose context lengths differ by an order of magnitude or more, capacity SHOULD be evaluated per segment (separate `capacity_at` calls), and segregated serving pools SHOULD be considered against one mixed pool.

**Why segregation is often free.** Segregation does not duplicate compute: it re-partitions it. The long-context pool needs the KV headroom the short pool does not; the short pool needs scheduler responsiveness the long pool cannot guarantee while streaming prefills. Because the KV floor for long-context traffic must be provisioned *somewhere* regardless, moving that traffic onto dedicated replicas typically spends the same GPU total as one mixed pool while removing the head-of-line interference that breaks the short-context SLO gate. The cost of segregation is scheduling rigidity; the cost of mixing is a tail you cannot tune away.

**Rule (MUST NOT).** A single blended capacity number MUST NOT be reported for a segmented workload as if it characterized all segments. *Failure prevented:* the blend lands between the segments and describes neither — short-context capacity overstated, long-context capacity understated, and the crossover invisible.

## Provenance summary for this layer

Every workload field declaring defaults (`peak_to_mean = 4.0`, `duty_cycle = 1.0`, the `avg_context_tokens` formula) is (I) or (U) until backed by product telemetry, at which point it becomes (M). Per C2, the report MUST tag each accordingly; per C1, an unknown MUST be recorded as `null` with a (U) entry, never guessed — because a guessed workload field is a GPU-count error with a serial number filed off.
