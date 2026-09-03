# `gb200-gemma4-31b-tp1` — one GB200, one dense 31B, and the rung where the ladder stopped versus the rung where it passed

> *"How many concurrent chat users can one GB200 carry before time-to-first-token breaks its promise?"*

> **CORRECTION (2026-09-04): the 128 rung on this ladder was never offered.** The bench harness
> built its HTTP client without setting a connection-pool limit, so httpx's default
> `max_connections=100` applied and no more than 100 requests could ever be in flight at once.
> Rungs 4 through 64 are below that cap and stand exactly as measured. The 128 rung offered 100.
> Worse, the requests the pool held back queued *after* their `issued_ts` was stamped, so the
> client's own backlog was billed to the server as time-to-first-token. The measured tier on this
> page therefore reports `max_concurrent_users = 128` for a load this campaign never generated,
> and it does so in the flattering direction: an operator sizing from it would buy capacity on
> the strength of a rung that never ran. The harness defect is fixed (the client now sets an
> unbounded pool, and every window records its peak in-flight count so the failure cannot recur
> silently), and this campaign is queued for re-measurement. Until it is re-run, treat every
> 128-rung figure below as withdrawn and read the ladder as censored at 64.

This example answers that question by running the full ladder on real hardware rather than
composing a forecast, and it is this repository's first campaign on NVIDIA's GB200 — one GPU
of a four-GPU tray, started by hand off-scheduler, serving a dense 31B checkpoint at
tensor_parallel 1. It is also the first campaign this project ran against an engine new
enough that running it surfaced defects in the harness itself, so the README records both the
measurement and what the measurement repaired.

The model is Gemma 4 31B Instruct, a dense bf16 checkpoint with a hybrid 5:1 local:global
attention layout, served by vLLM 0.18.2rc1.dev73+gdb7a17ecc — a release-candidate development
build, declared as such. The workload is a chat_assistant archetype: synthetic text prompts
with a declared 1,500 input tokens and exactly 500 output tokens under `ignore_eos`, with a
distinct prefix on every request so the prefix cache has nothing to return.

| path | what it is |
|---|---|
| `bench.json` | the campaign spec: endpoint, ladder, windows, and the SLO gates declared before the run |
| `hardware.json` | the hardware layer declaration |
| `model.json` | the model layer declaration, read from the checkpoint's own `config.json` and safetensors headers |
| `serving.json` | the serving layer declaration |
| `workload.json` | the workload and demand-model declaration |
| `engine.log` | the serving-engine log, mirrored live from the server's first line to the moment the bundle was sealed, and hashed in the manifest like everything else |
| `report.json`, `bundle/` | the report and the measured bundle it grades |

Every summary number in this README must be recoverable from the report and the bundle.

## What was served

| property | declaration or observation |
|---|---|
| model | gemma-4-31b-it, dense rather than MoE, 31,273,088,876 parameters, bf16 weights and KV |
| model revision | null, with a (U) reason: served from a local checkpoint directory, so the architecture fields are exact for the bytes served but no hub revision pins them |
| architecture | 60 layers, 16 KV heads, head_dim 256; hybrid attention, 5:1 local:global — 50 sliding-window layers at 1,024 tokens, 10 full-attention layers |
| parameter split | language model 30,697,345,340; vision tower 569,550,384; projector 6,193,152 — summed from the safetensors headers, not the model card |
| input modalities | text, image, audio in the checkpoint; **text only exercised in this campaign** |
| serving `framework_version` | vLLM 0.18.2rc1.dev73+gdb7a17ecc |
| container digest | null, with a (U) reason: the engine ran from an enroot container created out of a local squashfs image, which carries no registry digest — this is the one gap that caps the grade |
| GPU | 1 x NVIDIA GB200, tensor_parallel 1, pipeline_parallel 1 |
| VRAM per GPU | 198,674,743,296 bytes (185.0 GiB); HBM bandwidth 8 TB/s; dense bf16 2.5 PFLOP/s |
| interconnect | NVLink 5 intra-node, unused at TP=1; inter-node path null with a (U) reason — nothing about multi-node scaling is claimed |
| host | NVIDIA Grace (Arm Neoverse-V2), 144 cores, 956 GiB system RAM |
| storage and load path | WekaFS parallel filesystem, shared — model load path is the shared filesystem, not node-local NVMe |
| driver / runtime | driver 580.126.09, CUDA 13.0 |
| engine-reported KV pool | 114,544 tokens — a lower bound here, and the KV section below shows why |
| deployment controls | max_model_len 32,768, gpu_memory_utilization 0.9, `--max-num-seqs 256`, continuous batching, chunked prefill on, prefix caching on, KV offload and quantization off, speculative decoding off |
| cold start | 140.3 s from launch command to first HTTP 200 on `/v1/models`, polled every 5 s. Weight load from the shared filesystem plus the 72.3 s the log attributes to engine init. Measured with a warm filesystem cache and no scheduler queueing -- see `serving.json` notes |

The missing revision and missing digest are declared unknowns with consequences written next
to them, not silent gaps. The vision-tower replication field is likewise null at TP=1 — there
is only one rank, so replication is unobservable — and its (U) reason carries the warning: a
replicated 570 M-parameter tower costs about 1.1 GiB on every added rank, so the weights floor
measured here must not be extrapolated to TP>1 without re-measuring.

## What was run

The ladder climbed concurrency 4, 8, 16, 32, 64, 128 — three 120-second windows per rung,
20 warmup requests, a 60-second drain deadline, and a throughput-collapse guard at ratio 0.7
that never triggered. Every request carried a declared 1,500 input tokens and emitted exactly
500 output tokens under `ignore_eos`, so answer length is a controlled input, not an outcome.
Think time between a stream's requests was 1.5 seconds; the workload seed was 20260902.
Load is closed-loop: the stated concurrency is the number of sustained request streams, not a
Poisson arrival rate.

Prefix caching was left **on** in the engine and defeated in the workload, the same deliberate
combination as in the H100 example: `cache_policy` is unique-prefix, so no two requests share
a prefix, and a higher rung cannot answer partly out of a cache the rung below it filled. The
error from a shared prefix would run one way only, toward a ceiling that is not there.

The workload declaration also carries the planning fields this campaign was specified from:
200 peak concurrent users, duty cycle 0.25, 50 active sessions, 15 target tokens per second
per user, and 750 declared demand tokens per second. Those fields describe the demand model
the SLO gates were derived from — the 15 tok/s per user target is the ITL gate, 0.0667 s —
not the ladder itself, and the ladder is what this README grades.

The SLO gates were fixed before the run (`declared_before_run: true`): TTFT p95 no greater
than 2.5 s, ITL p95 no greater than 0.0667 s, end-to-end p95 no greater than 120 s, error
rate no greater than 1%. A rung passes only if all three repetitions satisfy every gate,
because capacity is defined by the worst served user, not by best-of-N.

GPU utilization and GPU memory utilization are unknown, as they always are in a report this
harness emits: a load generator cannot see the GPU, and the protocol's rule is applied to the
harness itself — it must not publish what it cannot observe.

## The result

**16 sustained request streams at 600.0 output tokens per second, bound by the SLO floor.**
That is the `sustainable` tier and it is the number to size against. Concurrency 16 is the
last rung where every window of every repetition met every gate, error rate was 0.0% across
the entire ladder, every request emitted its full 500 tokens, and the section 5 confirmation
repetition at 16 passed again. Concurrency 32 is where the deployment stops keeping its
promise.

The report also publishes a `measured` tier of 128 streams at 1,250.0 output tok/s, and the
two must not be confused. Chapter 5.5 defines the measured tier as *best observed, SLO
ignored* — deliberately gate-blind, because "what did the engine do" and "what may we promise"
are different questions and collapsing them is how a benchmark becomes a sales figure. Rung
128 failed its gates, as did 32 and 64. Throughput climbed monotonically the whole way up and
the server was still answering at 128; it was answering far too slowly to promise. Quote the
sustainable row.

| tier | capacity | throughput | binding constraint | provenance |
|---|---:|---:|---|---|
| `sustainable` — last rung meeting every gate | 16 streams | 600.0 tok/s, 1.2 req/s, 103,680 requests/day | `slo` | M |
| `measured` — best observed, gates ignored | 128 streams | 1,250.0 tok/s, 2.5 req/s, 216,000 requests/day | `slo` | M |
| `theoretical` | null | — | null, with (U) reasons | U |
| `recommended` | null | — | null, with (U) reasons | U |

The first SLO failure is at concurrency 32, on the TTFT p95 gate: 3.025 s against a 2.5 s
bound. The `theoretical` tier is absent because the roofline belongs to `ascep size`, and the
`recommended` tier is absent because derating a measurement needs a headroom factor, which is
a policy choice rather than a measurement — bench does not invent policy.

| concurrency | mean input tokens | prefill tok/s | output tok/s | TTFT p95 (s) | ITL p95 (s) | e2e p95 (s) | rung-level SLO outcome |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 1,543.7 | 617.5 | 200.0 | 0.421 | 0.0160 | 8.04 | pass |
| 8 | 1,543.9 | 1,132.1 | 366.7 | 0.788 | 0.0169 | 8.84 | pass |
| 16 | 1,545.3 | 1,854.3 | 600.0 | 1.566 | 0.0204 | 11.33 | pass |
| 32 | 1,545.5 | 2,884.7 | 933.3 | 3.025 | 0.0260 | 15.43 | fail |
| 64 | 1,545.5 | 3,296.8 | 1,066.7 | 6.022 | 0.0360 | 23.35 | fail |
| 128 | 1,546.5 | 3,866.1 | 1,250.0 | 33.062 | 0.0889 | 64.17 | fail |

The measured input is the declared 1,500 plus about 44 tokens of chat template — a check that
only passes because of a harness repair described below. Output tokens at 500.0 and error
rate 0.0 at every rung are part of the result and must travel with it.

## TTFT is the binding gate, not ITL

Interactive-serving intuition says the per-token rate binds first. It does not here. ITL p95
is 0.0260 s at concurrency 32 — comfortably inside the 0.0667 s gate — and stays inside it
until concurrency 128, by which point TTFT p95 has reached 33.1 seconds. TTFT p95 crosses the
2.5 s bound between rungs 16 and 32, and that is the whole failure story of this ladder. The
reason is the prompt shape: 1,500 in and 500 out is a prefill-heavy request for a dense 31B
model, and every admitted request pays its prefill before any queued request receives a first
token. The consequence for a reader tuning a similar deployment: a monitor that watches
tokens per second per user would have shipped concurrency 32 or 64 and violated a latency
gate it never looked at.

## The KV pool is not what its own token count suggests

The analytic KV model reproduces the engine to 0.011 percent, and in doing so shows that this
vLLM build pages the hybrid model **uniformly**. The closed form for all-60-layers-full is
2 × 2 bytes × 16 KV heads × 256 head_dim × 60 layers = 983,040 bytes per token; times the
engine's reported 114,544 tokens that is 104.868 GiB, against the 104.88 GiB the engine logs
as its available KV cache memory. The hybrid closed form, which charges only the 10 global
layers, gives 163,840 bytes per token and would put the same pool at roughly 687,000 tokens.
The engine allocated the uniform cache: the 50 sliding-window layers hold full-length pages
even though they only ever attend over 1,024 tokens.

The engine's own next log line disagrees with its own token count: it prints a maximum
concurrency of 10.99x for 32,768 tokens per request, where 114,544 / 32,768 is 3.50x — so
the scheduler admits roughly three times what the reported pool implies. The consequences are
declared rather than resolved: `engine_reported_kv_cache_tokens` is a lower bound on what
this deployment will hold, the KV floor derived from it is conservative, and the headroom is
a property of this engine build rather than of the model. A different build may page the
hybrid layout honestly, in which case none of this arithmetic transfers.

## What the ladder found in the harness

Running the campaign against a real engine found three defects in the harness that no unit
test could have found, and they are recorded here because they bear on whether the numbers
can be trusted.

1. **Synthetic prompts were not the size they declared.** The corpus generator padded with
   random hex strings, which this model's tokenizer charges at 7.98 tokens per word, while
   the harness sizes prompts by word count — a config declaring 1,500 input tokens sent about
   12,000. Fixed by padding with common English words; after the fix, 1,500 declared words
   measure as 1,500 tokens, and the 1,543.7 in the table above is that 1,500 plus template
   overhead.
2. **A two-rung ladder failed late for the wrong reason.** It ran to completion and then
   failed draft validation on `run.results` minItems, surfaced as a bench defect rather than
   as the undersized ladder the operator declared. It is now refused before the first
   request.
3. **The single-shape detector could never fire.** `run.single_point` was computed with
   `len(set(...))` over per-rung context means — floats carrying sampling noise — so the flag
   was always false and a single-shape campaign published itself as a context curve.

All three are under Fixed in the changelog. The table in this README is from a ladder that
ran after the prompt fix; the measured input column is the evidence the fix worked.

## Why the report is `partial`

`ascep conformance` grades this report `partial` with no errors and three warnings, and the
three are worth reading individually because they are not the same kind of gap.

**`reproduction.container_digest` (C8)** is the one that caps the grade. The engine ran from
an enroot container started out of a local squashfs image, which carries no registry digest,
so there is nothing to record and a note cannot substitute for one. The
`framework_version` string is the only handle on the build, and a release-candidate
development build is not a stable one. This is the same gap the H100 vision example carries,
for a related reason: a scheduler-managed environment and a locally-built container both
serve real work and neither hands you a digest.

**`capacity_tiers.theoretical` and `capacity_tiers.recommended` (C6)** are absences by
construction, not lost measurements. `ascep bench` is a load generator: it observes latency
and throughput over HTTP and nothing else, so the roofline comparison, the sizing result, the
scaling table and the theoretical tier are left null with (U) reasons rather than estimated.
The recommended tier is missing for a different reason again — derating a measurement needs a
headroom factor, and that is a policy choice nobody has declared for this deployment. Bench
does not invent policy.

What C4 does *not* say is worth noting too. The campaign measured one context length, and
`run.single_point` is true, so C4 is satisfied by declaration rather than by a three-point
curve. Until the release that shipped with this example, that flag could not be set at all,
and this report would have published a context curve it never measured.

Closing the gaps takes three separate things, none of them editable after the fact: build the
engine from an image with a digest and record it; run `ascep size` with the hardware and model
declarations to fill the theoretical tier and the roofline; and declare a headroom policy, or
re-run the ladder at three context lengths if you want C4 satisfied by measurement rather than
by declaration.

## What not to conclude

- **Do not size against 128.** That is where the ladder stopped, not where the deployment is
  usable. The last rung meeting the declared SLO is 16, and 16 is the quoted capacity.
- **Do not tune this deployment by per-token rate.** TTFT binds here, not ITL; a tokens/s
  dashboard would have missed the failing gate.
- **Do not scale from one GPU.** TP=1 on one GB200 of a four-GPU tray measures nothing about
  sharding, NVLink cost, vision-tower replication at TP>1, or multi-node scaling — the
  inter-node interconnect is declared null for exactly that reason.
- **Do not extrapolate the KV pool to another engine build.** The uniform-paging behavior and
  the threefold scheduler headroom are properties of vLLM 0.18.2rc1.dev73+gdb7a17ecc, not of
  Gemma 4 31B. Treat the reported 114,544 tokens as a lower bound, and re-measure on any
  other build.
- **Do not infer image or audio capacity.** The checkpoint accepts text, image and audio;
  this campaign sent text only, and the media fields exist to mark what was left unmeasured.
- **Do not infer thinking-mode capacity.** This is a `non-thinking` run and the output length
  was fixed at 500 tokens by `ignore_eos`; a thinking profile would change decode time and KV
  occupancy without announcing itself through the model name.
- **Do not convert closed-loop concurrency into arrival-rate service.** No Poisson arrival
  process, burst profile or queueing margin was measured.
- **Do not assume these SLO gates are universal.** The thresholds were declared from this
  campaign's demand model before the run.
- **Do not treat 140.3 s of cold start as a floor for your site.** It was measured after the
  same checkpoint had been read from the same filesystem minutes earlier, so the WekaFS client
  cache was warm, and nothing queued for an allocation.
- **Do not quote a tier without its rung and floor.** Capacity moves with context length and
  concurrency, and the named binding constraint is part of the answer.
- **Do not read `partial` as a failed run.** It is a published measurement with declared gaps
  and a published description of what closing them takes. The grade says which inputs are
  absent, not that the numbers are unsound.

## Corrections a later campaign found in this bundle's declarations

The multi-image campaign at [`../gb200-gemma4-31b-multi-image/`](../gb200-gemma4-31b-multi-image/)
served the same checkpoint and, because it actually sent images, exercised three media fields
this campaign only transcribed. Two of them were wrong here. The declarations and the report in
this directory are left exactly as they were run -- editing an input after the fact would break
the correspondence between the bundle and the numbers it produced -- so the corrections are
recorded here instead.

- **`model.image_tokens_fixed: 280` overstates the cost by about 5.6 percent.** 280 is the
  checkpoint's own soft-tokens-per-image setting, transcribed from its config. Measured against
  the running server it is a ceiling the preprocessor approaches but never reaches: over 24
  resolutions the charged cost was 256 to 274 tokens, mean 265, and it depends on aspect ratio
  rather than on pixel count. This campaign sent no images, so nothing here was affected; a
  reader pricing an image workload from this file would have been.
- **`model.video_frame_policy: "n-a"` is wrong, and `input_modalities` is missing `video`.**
  The checkpoint ships a `Gemma4VideoProcessor` with `do_sample_frames: true`, `num_frames: 32`
  and `max_soft_tokens: 70`, and the engine sizes its encoder cache "profiled with 3 video items"
  at startup. The policy is `uniform-count` at a flat 2,240 tokens per clip regardless of
  duration. `n-a` was an assumption from a text-only run, not a finding.
- **`serving` has no `media_preprocessing` object**, which is correct for a text-only run and is
  why C4 stays quiet here. The multi-image bundle fills it, and the number that belongs in it --
  a 645,120-pixel budget, about 0.65 megapixels -- is the one that decides what this deployment
  actually sees of a scanned page.

The general lesson is the one the dead-rule doctrine already states in the other direction: a
field that a campaign does not exercise is a field that campaign cannot check, and transcribing
a config value into it looks exactly like measuring it. The `_tag` mechanism marks a number as
measured, inferred or unknown for exactly this reason, and these three fields carried no tag.
