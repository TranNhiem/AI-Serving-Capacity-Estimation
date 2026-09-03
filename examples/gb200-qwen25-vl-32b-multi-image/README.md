# `gb200-qwen25-vl-32b-multi-image` -- one GB200, 13,644 images, and a ladder that refused to publish a number

> *"How many concurrent multi-image reasoning users can one GB200 carry -- and what does a framework owe you when the measurement cannot honestly say?"*

This example is the companion campaign to
[`../gb200-gemma4-31b-multi-image/`](../gb200-gemma4-31b-multi-image/): same corpus, same
digest, same seed, same ladder, same SLO gates, same tray, run three hours later against a
different model. The model is Qwen2.5-VL-32B-Instruct, a dense bf16 checkpoint that prices
images by dynamic resolution rather than by a fixed grid, so the same 13,644 images cost it
3.78 times the tokens they cost gemma.

The headline is not a number. This is the first published ASCEP bundle in which the harness
refused to publish a sustainable capacity figure at all. Concurrency 4 passed every gate and
was confirmed; concurrency 5 failed its gates on its worst window; concurrency 6, above the
failure, passed. A boundary search rests on the assumption that rungs pass below the boundary
and fail above it, and this ladder violates that assumption, so the harness publishes no
sustainable tier and states why. That refusal is the finding, and most of this README is the
evidence for why refusing is correct. A framework willing to smooth over the gap would report
"6 sustainable users" here, and it would be reporting an arbitrary point wearing a boundary
label.

| path | what it is |
|---|---|
| `bench.json` | the campaign spec: endpoint, ladder, windows, and the SLO gates declared before the run |
| `hardware.json` | the hardware layer declaration |
| `model.json` | the model layer declaration, read from the checkpoint's own `config.json` and safetensors headers |
| `serving.json` | the serving layer declaration, pinned before the launch command was frozen |
| `workload.json` | the workload and demand-model declaration, with the measured media token cost |
| `report.json` | the report, re-derived from the bundle by `ascep reduce` and graded `partial` |
| `bundle/` | the measured bundle the report grades: config, pinned declarations, records, manifest, and `engine.log` inside it |

Every summary number in this README must be recoverable from the report and the bundle.

## What was served

| property | declaration or observation |
|---|---|
| model | qwen2.5-vl-32b-instruct, dense rather than MoE, 33,452,718,336 parameters, all active, bf16 weights and KV |
| model revision | null, with a (U) reason: served from a local checkpoint directory with no upstream commit or content digest recorded |
| architecture | 64 layers, 40 query heads, 8 KV heads, head_dim 128, GQA; `use_sliding_window` false, so every layer is global attention |
| native context | 128,000 tokens |
| weights on disk | 68,283,251,376 bytes, tagged M |
| license | Apache-2.0 |
| input modalities | text, image and video in the checkpoint; **image only exercised in this campaign** |
| vision tower | 688,841,984 parameters, patch 14 px, spatial merge 2 |
| GPU | 1 x NVIDIA GB200, 1 of 4 GPUs on an exclusive tray, tensor_parallel 1 |
| GPU capability | 198,674,743,296 bytes VRAM, HBM bandwidth 8,000,000,000,000 bytes/s, dense bf16 2,500,000,000,000,000 FLOP/s |
| interconnect | NVLink 5 intra-node |
| host | NVIDIA Grace (Arm Neoverse-V2), 144 physical cores across 2 sockets, 1,026,841,051,136 bytes system RAM |
| storage and load path | WekaFS shared parallel filesystem; weights loaded over that filesystem, not from node-local NVMe |
| driver / runtime | driver 580.126.09, CUDA 13.0 |
| media transport | base64 in the request body, not by URL |

Two host facts matter because this workload decodes base64 images on the host. The serving
process was not pinned and not cgroup-limited, and `nproc` inside the session reported 128,
so 16 of the 144 cores sat outside the default affinity mask. And the tray divides those same
cores four ways when fully loaded, so the host-side media path this campaign exercised is not
the host-side media path a fully loaded tray would offer.

**KV bytes per token, verified from geometry**: 2 (K and V) x 2 (bytes per bf16 element) x 8
(KV heads) x 128 (head_dim) x 64 (layers) = **262,144 bytes per token**. This arithmetic is
kept here because it is what makes the KV finding below checkable by the reader: every KV
figure in this README is either this number or a number divided by it.

**Image token policy is `dynamic-resolution`, not fixed.** `image_tokens_fixed` is null with
a (U) reason. The declared range is 6 to 1,282 tokens per image, and the corpus-exact
measured mean is **1,001.98 tokens per image**. Where the gemma campaign could declare a flat
265, this model charges what each image's resolution costs, and the spread across the corpus
is the whole point of finding 2.

## The declaration was pinned before the launch command was frozen

`serving.json` was pinned before the launch command was frozen, and it says so honestly:
`framework_version`, `max_model_len`, `memory_utilization`, `prefix_caching`,
`chunked_prefill`, `kv_cache_offload`, `engine_reported_kv_cache_tokens` and
`cold_start_to_ready_s` are all null with (U) reasons of the form "established by reading the
launch command once it is frozen".

The launch command is now frozen, and the engine log inside the same sealed bundle answers
every one of those (U) reasons. Verbatim from its first lines:

* vLLM `0.18.2rc1.dev73+gdb7a17ecc`
* `tp=1`, `max_model_len=32768`, `gpu_memory_utilization=0.90`, `max_num_seqs=256`
* `--limit-mm-per-prompt {"image":16}`
* `--mm-processor-cache-gb 0` -- the multimodal processor cache is **off**, matching the
  published gemma run, which is what makes the two campaigns comparable
* `--mm-processor-kwargs {"max_pixels":1003520,"min_pixels":3136}`
* chunked prefill enabled, `max_num_batched_tokens=8192`
* asynchronous scheduling enabled
* FLASHINFER attention backend for the language model, FLASH_ATTN for the vision tower,
  HND KV cache layout
* available KV cache memory **101.46 GiB**; **GPU KV cache size 415,584 tokens**; the
  engine's own line "Maximum concurrency for 32,768 tokens per request: 12.68x"
* engine init (profile, create KV cache, warm up) took 47.77 seconds

The declaration was left as pinned rather than back-edited with these values, because a
declaration edited after the run is no longer a declaration. The point is worth stating
plainly: a bundle is not just a receipt, it is a self-correcting document. The declaration
says what was not yet known, the engine log in the same sealed directory says what turned out
to be true, and a reader can close the gap without re-running anything.

One of those closed gaps is the cheapest available check that the model declaration describes
the thing that actually ran: the engine's 415,584 KV tokens times the declared 262,144 bytes
per token is 101.4609 GiB, which is the 101.46 GiB the engine prints. Run the check in that
direction, not the other one: dividing the printed 101.46 GiB by 262,144 gives 415,580.16,
short by about four, because the printed figure is rounded to two decimals and the token count
is not. The pool and the checkpoint geometry agree to within the engine's display rounding,
which is as close as a logged figure can be checked.

The `max_pixels` budget of 1,003,520 is the ceiling on dynamic resolution: at patch 14 and
spatial merge 2 each merged token covers 784 pixels, so 1,003,520 pixels is 1,280 tokens,
which is where the declared `image_tokens_max` of 1,282 comes from. This is a **deployment
choice**, not a property of the model, and a different `max_pixels` would move every context
figure in this report.

## What was run

The workload is multi-image visual reasoning over charts, documents and photographs: corpus
`corpus.jsonl`, digest `742f54d60f7fbd652fab51ebaf6ac2349b015a8f89b8759cb15499064493552a`,
3,256 records, field `conversations`, 13,644 image references, **4.1904 images per record**,
0 images missing dimensions, 11,916 distinct resolutions. Images travel as base64 in the
request body, with 6,124,064,253 bytes of media resident on the tray. The archetype is
`image_grounded`, non-thinking, and `videos_per_request` is 0.

Declared per-request shape: `input_tokens_per_request` 39.5, `output_tokens_per_request` 500,
`media_tokens_per_request` **4,198.71**, `avg_context_tokens` **4,488.21** tagged M. The
per-image mean is 1,001.98 tokens; per-record media runs from 27 to 19,927 tokens. The
declared demand is inherited unchanged from the gemma campaign so the two reports differ only
in what the prompts carry: 200 concurrent users at duty cycle 0.25, so 50.0 active sessions
at a target 15.0 tokens per second per user, **750.0 output tok/s of demand**.

Declared in `bench.json` before the run and not changed during it: endpoint
`http://127.0.0.1:8001`, model `qwen2.5-vl-32b-instruct`, request timeout 600 s; ladder
concurrency `[4, 5, 6, 7, 8, 16, 32, 64, 128]` with 3 repetitions and a
throughput-collapse guard at ratio 0.7 that never triggered; 120 s counted windows, a 60 s
drain deadline, 20 warm-up requests (measured warm-up 30.008 s); `output_tokens` 500 with
`ignore_eos` true, so answer length is a controlled input and not an outcome; `cache_policy`
`unique-prefix`; `think_time_s` 1.5; seed 20260903; sampler uniform-with-replacement. The SLO
gates were fixed before the run (`declared_before_run: true`): TTFT p95 no greater than 2.5
s, ITL p95 no greater than 0.0667 s, end-to-end p95 no greater than 120 s, error rate no
greater than 1.0 percent.

28 windows were recorded: 27 counted (9 rungs x 3) plus one confirmation window at
concurrency 4 taken after the search, which passed.

## The ladder as published

`ttft_p95_s` and `output_tok_s` below are the published row figures; the dispersion columns
are from each row's `dispersion` block.

| concurrency | outcome | slo_pass | ttft_p95_s | ttft min | ttft med | ttft max | ttft spread | output tok/s | itl_p95_s | e2e_p95_s | req/s | error % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | complete | true | 1.6278 | 1.6278 | 1.7515 | 1.7698 | 8.11% | 216.67 | 0.01545 | 9.001 | 0.4333 | 0.0 |
| 5 | **failed** | true | 2.1275 | 2.1099 | 2.1275 | 2.6134 | 23.66% | 250.00 | 0.01656 | 10.150 | 0.5000 | 0.0 |
| 6 | **complete** | true | 2.3651 | 2.1373 | 2.3651 | 2.3958 | 10.93% | 275.00 | 0.01773 | 9.966 | 0.5500 | 0.0 |
| 7 | failed | false | 2.6996 | 2.6193 | 2.6996 | 3.3819 | 28.25% | 320.83 | 0.01928 | 10.586 | 0.6417 | 0.0 |
| 8 | failed | false | 2.5833 | 2.5833 | 2.6461 | 3.1287 | 20.61% | 366.67 | 0.01880 | 10.155 | 0.7333 | 0.0 |
| 16 | failed | false | 4.9384 | 4.5148 | 4.9384 | 5.1546 | 12.95% | 533.33 | 0.02635 | 16.090 | 1.0667 | 0.0 |
| 32 | failed | false | 9.3228 | 8.4440 | 8.8491 | 9.3228 | 9.93% | 666.67 | 0.03628 | 20.963 | 1.3333 | 0.0 |
| 64 | failed | false | 16.7541 | 15.8827 | 16.3748 | 16.7541 | 5.32% | 800.00 | 0.05744 | 33.113 | 1.6000 | 0.0 |
| 128 | failed | false | 55.1756 | 50.4983 | 52.3415 | 55.1756 | 8.94% | 700.00 | 0.07975 | 93.061 | 1.4000 | 0.0 |

Measured context tokens per request trended upward across the ladder, from 4,127.3 at
concurrency 4 to 5,297.0 at concurrency 128, dipping at concurrency 8 and again at 32. It is
not a monotone series and should not be read as one: the sampler draws requests independently
at each rung, so a rung's mean context is a property of the draw as much as of the
concurrency. Measured prefill throughput was 1,629.4 tok/s at concurrency 4, peaked at
**7,135.9 tok/s at concurrency 64**, and fell to 6,581.1 at concurrency 128. The measured
input-to-output ratio at concurrency 4 was 7.52. Do not try to reproduce that from the row's
`input_tokens` of 3,627.3 and `output_tokens` of 500, which divide to 7.25: the two figures are
taken over different populations. The ratio sums prompt and output tokens over the requests
that *completed* inside the window, because it exists to describe the traffic the throughput
figures in the same row were produced by; the mean token counts are taken over every request
in the window, completed or not. `tokens_per_stream_chunk` is 1.2818 and `itl_population` is
`per-request-mean`.

Each row is one real repetition, not an aggregate: the harness ranks the rung's three counted
windows by output tok/s and breaks ties on `ttft_p95_s`, then publishes the median one, so
every figure in a row is mutually consistent because one window exhibited all of them. On the
gemma ladder every rung's three windows tied on throughput, so the tiebreak decided
everywhere and the published TTFT was always the lower median. **Do not carry that shorthand
to this bundle.** Here throughput does not always tie: at concurrency 4 the three windows
measured 200.00, 216.67 and 216.67 tok/s, and at concurrency 8, 32, 64 and 128 one window
differed from the other two. So the published TTFT is not always the middle of the three
TTFTs -- at concurrency 4 and 8 it is the fastest window, and at concurrency 32, 64 and 128
it is the slowest. The `dispersion` block is what lets a reader see this; without it the row
alone is not enough.

Concurrency 5 is the row to read twice: its `slo_pass` is true and its `outcome` is failed,
and finding 1 is the explanation of how both are true.

## Finding 1: no sustainable tier, and why that is correct

`capacity_tiers.sustainable` is null with this (U) reason, quoted verbatim:

> (U) the ladder produced no sustainable tier: concurrency 4 passed its gates but the ladder
> does not permit publishing it as a boundary; the ladder was not monotone

The mechanism, which is the reusable part:

* A rung is COMPLETE only when **all three** counted windows pass **every** gate for the full
  window. Concurrency 4 and 6 are COMPLETE; every other rung is FAILED.
* Concurrency 5 failed on its worst window: its three TTFTs were 2.1099, 2.1275 and 2.6134 s
  against the 2.5 s gate. Two passed, one missed by 4.5 percent. Section 5 grades a rung on
  its worst window, so the rung failed -- while its published row, 2.1275 s, sits comfortably
  inside the gate and reports `slo_pass: true`. This is the same row-versus-outcome gap the
  gemma bundle found at its concurrency 7, arriving here one rung lower.
* The sustainable candidate is the highest COMPLETE rung **below the first non-COMPLETE
  rung**. The first non-COMPLETE rung is 5, so the candidate is 4, and the confirmation
  window at concurrency 4 passed.
* But publishing a sustainable figure also requires the ladder to be monotone, and it is not:
  `monotonic_across_ladder` is false, because concurrency 6 is COMPLETE and sits **above**
  failed concurrency 5. A pass above a failure contradicts the assumption a boundary search
  rests on. The harness therefore publishes no sustainable figure and states why.

Now the argument that matters to a practitioner: **the non-monotonicity is almost certainly
noise, and that is exactly the point.** Concurrency 5's failing window missed the gate by 4.5
percent, and its three windows spread 23.66 percent; concurrency 6's worst window was 2.3958
s, inside the gate, with a 10.93 percent spread. Read as a smooth curve, the true boundary is
somewhere between 5 and 7 and this ladder simply does not resolve it. A framework willing to
smooth would report "6 sustainable users" and be wrong by an unknown amount in an unknown
direction. Refusing is not the framework failing to produce an answer; it is the framework
declining to manufacture one. The fix is more repetitions at 5, 6 and 7, not a different
reduction of the same data.

A reader who needs a number today should treat concurrency 4 as a **lower bound** that was
measured and confirmed, note that it is not published as a boundary, and re-run the 5-to-7
region with more repetitions before quoting anything higher.

## Finding 2: the same images cost 3.78 times the tokens, and at low load it barely shows

This is the cross-model result. Both campaigns ran the same corpus, same digest, same seed,
same ladder, same gates, same tray, same day, both with the multimodal processor cache off.

| | gemma-4-31b-it | Qwen2.5-VL-32B-Instruct |
|---|---|---|
| image token policy | fixed grid | dynamic-resolution, capped by `max_pixels` |
| tokens per image | 265, fixed | 1,001.98 measured mean, measured 18 to 1,282 |
| `media_tokens_per_request` | 1,111.5 | 4,198.71 |
| `avg_context_tokens` | 1,395.0 | 4,488.21 |
| engine KV pool | 114,544 tokens | 415,584 tokens |
| peak GPU KV cache usage in the engine log | 88.3% | 100.0% |
| `ttft_p95_s` at concurrency 4 | 1.6017 | 1.6278 |
| `ttft_p95_s` at concurrency 8 | 2.7197 | 2.5833 |
| output tok/s at concurrency 128 | 833.33 | 700.00 |
| sustainable tier | 6 concurrent users, confirmed | none published, ladder not monotone |
| measured tier | 128 users, 833.33 tok/s | 128 users, 700.00 tok/s |

Qwen prices the identical images at **3.78 times** gemma's token count and carries **3.2
times** the context per request, yet at concurrency 4 its TTFT is within 1.6 percent of
gemma's, and at concurrency 8 it is actually **lower**. At light load the image token count
is not what the user waits for. It becomes decisive at saturation, where the measured ceiling
is 700.00 tok/s against gemma's 833.33 -- 16 percent less throughput for 3.2 times the
context.

Do not draw the obvious wrong conclusion: this is **not** a statement that one model is
better. The two price images differently by design, the token counts are not comparable units
of work, and Qwen's `max_pixels` here is a launch flag that could be set lower. What the
comparison licenses is narrower and more useful: **per-image token count is a poor predictor
of serving capacity, and a capacity estimate built by multiplying a token price by a request
rate would have mis-sized this deployment in both directions.**

## Finding 3: the KV pool fills, and the throughput regression sits on top of it

Concurrency 128 delivers **less** throughput than concurrency 64: 700.00 tok/s against
800.00, while offered load doubled. That is a regression, not a plateau, and the gemma ladder
does not have one -- it climbs monotonically to 833.33.

The evidence in the bundle, stated as correlation and not as a proven mechanism:

* The engine's KV pool is 415,584 tokens. At concurrency 128 the measured context is 5,297.0
  tokens per request plus 500 output tokens, so 128 concurrent streams want roughly 742,000
  tokens of KV -- about 1.8 times the pool.
* The engine log's peak `GPU KV cache usage` on this run is **100.0 percent**. On the gemma
  run, whose ladder does not regress, the peak is **88.3 percent**.
* Peak scheduler state on this run: 100 requests running with up to 33 waiting. The engine
  never ran all 128 offered streams at once.
* Measured prefill throughput peaks at concurrency 64 (7,135.9 tok/s) and falls at 128
  (6,581.1).
* **No preemption is recorded in either engine log.** This is stated explicitly because the
  obvious mechanism for a KV-bound regression is preemption and recompute, and this bundle
  does not contain that evidence. What it contains is a full pool, a queue, and a fall in
  throughput.

The honest claim is therefore narrow: the run whose KV pool saturates is the run whose
throughput regresses, both facts are recorded in the same sealed engine log, and establishing
the mechanism needs instrumentation this campaign did not run.

The reusable lesson: ASCEP's KV floor is not decoration. The declared 262,144 bytes per token
and the engine's own 415,584-token pool line are both in the bundle, so a reader can compute
the concurrency at which KV runs out before spending a GPU-hour -- and this ladder's top rung
is past it.

## Finding 4: dispersion is widest exactly where the boundary is

TTFT spread by rung: 8.11, 23.66, 10.93, 28.25, 20.61, 12.95, 9.93, 5.32, 8.94 percent. The
widest spreads are at concurrency 5 and 7 -- the two rungs that decide where the boundary is
-- and the narrowest at 64, where the engine is saturated and every window measures the same
queue. This is the opposite of convenient: the region that needs resolution is the region the
measurement resolves worst. It is why three repetitions were not enough here and were enough
for gemma, and it generalizes: a campaign that wants a boundary must spend its repetitions
near the boundary, not spread them evenly.

`error_rate_pct` carries `spread_pct: null` with a (U) reason on every rung: a relative
spread against a zero median is a division by zero dressed as a statistic. Error rate was 0.0
at every rung, and that is guaranteed by the run design rather than earned -- `ignore_eos`
with a fixed 500-token output and a 600 s request timeout means nothing in this workload can
fail fast enough to surface as an error.

## Conformance

Graded by `ascep conformance --raise`, which rewrote the report's claim from `non-conforming`
(the ungraded-draft default that `ascep bench` writes) to **`partial`**. Four findings, all
warnings, no errors:

* C6 warning on `capacity_tiers.theoretical.max_concurrent_users` -- a roofline is computed
  by `ascep size`, not observed by a load generator
* C6 warning on `capacity_tiers.sustainable.max_concurrent_users` -- the non-monotone ladder
  of finding 1
* C6 warning on `capacity_tiers.recommended.max_concurrent_users` -- a headroom factor is a
  policy choice, not a measurement
* C8 warning on `reproduction.container_digest` -- the server was launched from a conda
  environment on the tray, not a container, so there is no digest to publish; this caps the
  report at partial, and the cap is correct

There are **no C12 findings**: every rung of this `repeats: 3` run publishes its dispersion
block.

## The bundle

`bundle/` contains, all named in `manifest.json` and all verifying:

* `bench-config.json`, 1,303 bytes
* `declarations/hardware.json`, `declarations/model.json`, `declarations/serving.json`,
  `declarations/workload.json` -- byte-identical to the four copies in this directory
* `engine.log`, 2,703,677 bytes as published -- see the redaction note below
* `environment.json`, 779 bytes
* `manifest.json`, 1,252 bytes
* `records.jsonl`, 29,447,806 bytes -- one line per request
* `run_configs.json`, 23,916 bytes -- 28 window entries with their policies and boundaries

`persist.verify_bundle` reports no problems.

**Redaction.** `engine.log` as written by the engine contains the absolute checkpoint path on
the shared cluster filesystem and one private-range IP address from the engine's distributed
init line. Both were replaced before publication with `tools/redact_bundle.py`, which
re-seals the manifest and records under a top-level `redactions` key the original SHA-256 of
each touched file, the replacement text and the number of occurrences. The manifest's promise
is not "these are the bytes the engine wrote"; it is "these are the published bytes, and here
is exactly how they differ from what the run wrote". A reader who re-runs the campaign will
get the unredacted path back and can check everything else byte for byte.

## Reproducing this

```bash
ascep conformance examples/gb200-qwen25-vl-32b-multi-image/report.json
ascep reduce examples/gb200-qwen25-vl-32b-multi-image/bundle --check
```

`--check` exits 0 when the report rebuilt from the raw records matches the published one,
comparing every figure and excluding two things: `report_generated_utc`, because a rebuild
really is generated at a new time, and the conformance grade, because a rebuild is an
ungraded draft by construction and the grade is recomputed from the figures by the command
above. That is the guarantee that makes the bundle worth its 29 MB: when the reduction code
is fixed, a published figure can be corrected without re-burning the GPU hours.

Note that `corpus.jsonl` (54,209,882 bytes) and the media tree (6,124,064,253 bytes) are
**not** published in this directory. The corpus digest in `run_configs.json` is what pins
them.

## What this bundle does not establish

* **No sustainable capacity figure.** The measured tier, 128 users at 700.00 tok/s, is an
  engine ceiling observed with the two latency gates failing: `ttft_p95_s` is 55.1756 against
  a 2.5 gate and `itl_p95_s` is 0.07975 against 0.0667, while `e2e_p95_s` at 93.061 and an
  error rate of 0.0 are still inside theirs. Nothing broke at 128; it was just far too slow
  to serve, which is why the tier is not a number to size against. Concurrency 4 is a
  measured and confirmed lower bound, not a published boundary.
* **Nothing about tensor_parallel above 1**, and nothing about a second GPU or a second
  replica.
* **Nothing about video**, although the checkpoint supports it: the server was launched with
  `--limit-mm-per-prompt {"image":16}`, so its video budget is zero and a video request is
  refused with HTTP 400. `videos_per_request` is 0 for that reason, not because video was
  tried and found slow.
* **Nothing about long context.** Measured context ran 4,127 to 5,297 tokens against a
  32,768-token `max_model_len` and a 128,000-token native ceiling, so the long-context
  behaviour of this deployment is untouched.
* **Nothing about prefix caching or the multimodal processor cache**, both of which were off
  or undeclared; the gemma bundle's README reports a separate observation on the processor
  cache.
* **No roofline and no sizing result**, both of which are `ascep size` outputs and not
  measurable by a load generator.

## Demand, restated

The declared demand is 750.0 output tok/s. No rung that met the gates came close: the highest
COMPLETE rung by throughput, concurrency 6, delivered 275.00 tok/s, **36.7 percent of
demand**. The rungs that approach demand fail badly -- 800.00 tok/s at concurrency 64 costs
16.7541 s of `ttft_p95_s`, **6.7 times** the 2.5 s gate. One GB200 at tensor_parallel 1 is
undersized for this declared demand by roughly a factor of 3. That conclusion does not depend
on resolving the sustainable rung, which is worth saying because it is the one capacity
statement this bundle can make without qualification.
