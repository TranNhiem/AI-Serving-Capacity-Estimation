# `gb200-gemma4-31b-multi-image` -- one GB200, 13,644 images, and a workload this GPU cannot hold

> *"How many concurrent multi-image reasoning users can one GB200 carry before time-to-first-token breaks its promise -- and what does sending images actually cost?"*

This example answers both by running the full ladder on real hardware against a real corpus
rather than composing a forecast. It is the companion campaign to
[`../gb200-gemma4-31b-tp1/`](../gb200-gemma4-31b-tp1/): same checkpoint, same tray, same
engine build, same KV pool, differing only in what the prompts carry. Where the sibling sent
1,500 synthetic text tokens, this campaign sends questions over a 3,256-record Molmo2
multi-image reasoning corpus -- 13,644 charts, scanned pages and photographs, mean 4.1904
images per record, transported as base64. It is also the campaign that exercised the media
fields the text run could only transcribe, and it found two of them wrong.

The headline is a negative result, and it is stated here in words because the tier machinery
alone does not say it: no tested operating point satisfies both the declared demand and the
declared gates. The workload's demand model asks for 750 output tokens per second -- 50
active sessions at 15 tokens per second per user. The highest ladder rung that passes every
SLO gate is concurrency 6, and it delivers 250.00 output tok/s, which is 33.3 percent of
demand. The rungs that approach demand fail the time-to-first-token gate by a wide margin:
64 users at 800.0 tok/s pays 15.9451 s of TTFT p95, six times the 2.5 s gate, and 128 users
at 833.33 tok/s pays 53.8161 s. One GB200 at tensor_parallel 1 is undersized for this
workload as declared, by a factor of three. The text sibling on the same GPU, whose prompts
carry no media, sustained 16 streams at 600.0 tok/s.

The model is Gemma 4 31B Instruct, a dense bf16 checkpoint with a hybrid 5:1 local:global
attention layout, served by vLLM 0.18.2rc1.dev73+gdb7a17ecc -- a release-candidate
development build, declared as such. The engine was launched with
`--limit-mm-per-prompt {"image":16}` because the corpus reaches 16 images in a single record
and the default of 1 rejects such a request with HTTP 400.

| path | what it is |
|---|---|
| `bench.json` | the campaign spec: endpoint, ladder, windows, and the SLO gates declared before the run |
| `hardware.json` | the hardware layer declaration |
| `model.json` | the model layer declaration, read from the checkpoint's own `config.json` and safetensors headers |
| `serving.json` | the serving layer declaration, including the media preprocessor |
| `workload.json` | the workload and demand-model declaration, with the measured media token cost |
| `report.json` | the report, re-derived from the bundle by `ascep reduce` and graded `partial` |
| `bundle/` | the measured bundle the report grades: config, pinned declarations, records, manifest, and `engine.log` inside it |

Every summary number in this README must be recoverable from the report and the bundle.

## What was served

| property | declaration or observation |
|---|---|
| model | gemma-4-31b-it, dense rather than MoE, 31,273,088,876 parameters, bf16 weights and KV |
| model revision | null, with a (U) reason: served from a local checkpoint directory, so the architecture fields are exact for the bytes served but no hub revision pins them |
| architecture | 60 layers, 16 KV heads, head_dim 256; hybrid attention -- 50 sliding-window layers at 1,024 tokens, 10 full-attention layers |
| parameter split | language model 30,697,345,340; vision tower 569,550,384; projector 6,193,152 |
| input modalities | text, image, video, audio in the checkpoint; **image only exercised in this campaign** |
| serving `framework_version` | vLLM 0.18.2rc1.dev73+gdb7a17ecc |
| container digest | null, with a (U) reason: enroot container from a local squashfs image, which carries no registry digest |
| GPU | 1 x NVIDIA GB200, tensor_parallel 1, pipeline_parallel 1 |
| host | NVIDIA Grace (Arm Neoverse-V2), 144 cores, 956 GiB system RAM |
| storage and load path | WekaFS parallel filesystem, shared |
| driver / runtime | driver 580.126.09, CUDA 13.0 |
| engine-reported KV pool | 114,544 tokens, 104.88 GiB -- byte-identical to the text campaign |
| media transport | base64 in the request body |
| image pixel budget | 645,120 pixels -- the cap that binds silently, below |
| mm processor cache | 0 bytes, deliberately -- the published run is the cache-off one, below |
| multimodal limit | `--limit-mm-per-prompt image=16`, recorded in prose because ASCEP has no field for it |
| deployment controls | max_model_len 32,768, gpu_memory_utilization 0.9, continuous batching, chunked prefill on, prefix caching on, KV offload and quantization off, speculative decoding off |
| cold start | 134.0 s first banner line to last route registration, read off the engine log with 1 s resolution -- a lower bound by well under a second |

The cold-start number wants one caution. The text campaign declared 140.3 s for the same
model on the same tray by polling `/v1/models` every 5 s, which is an upper bound by up to
5 s plus the poll that missed. The two figures are the same start measured two ways, not a
media penalty. What the 134.0 s covers is 62.5 GB of weights off the shared filesystem,
36.43 s of torch.compile, 12 s of CUDA graph capture, and 74.78 s the engine attributes to
init overall; what it does not cover is a genuinely cold filesystem or any scheduler
queueing. GPU utilization and GPU memory utilization are unknown, here as in every report
this harness emits: a load generator cannot see the GPU.

## What was run

The ladder was declared in `bench.json` as concurrency 4, 5, 6, 7, 8, 16, 32, 64, 128. An
earlier run of this campaign doubled from 4 straight to 8 and could therefore only report
that 4 passed and 8 failed; the four rungs between them were added to the declaration and the
campaign re-run, which is why the sustainable answer below is 6 and not 4. The harness does
not insert rungs of its own, so a coarse ladder yields a coarse capacity number and nothing
warns you. Nine rungs in all -- three 120-second windows per rung after 20
warm-up requests, a 60-second drain deadline, the run seeded at 20260903, and a
throughput-collapse guard at ratio 0.7 that never triggered. Every request emitted exactly
500 output tokens under `ignore_eos`, so answer length is a controlled input, not an
outcome; this matters to every decode figure below and is called out again there. Load is
closed-loop: the stated concurrency is sustained request streams, not a Poisson arrival
rate, with 1.5 s of think time between a stream's requests. Prefix caching stayed **on** in
the engine and was defeated in the workload by `cache_policy` unique-prefix, the same
combination as the text campaign; here every record also carries different images, so reuse
is near zero by construction.

The prompt itself is nothing like the text sibling's. The text part measures 33.5 tokens --
questions about images, not passages -- and the media costs a declared 1,111.5 tokens per
request, so the workload is roughly 97 percent media by prompt token. The declared average
context is 1,395.0 tokens: 33.5 text plus 1,111.5 media plus half of the 500 output tokens,
tagged M because the prompt side is server-measured, though the output-half residency term
is a model rather than a measurement. The planning fields (200 peak users, duty cycle 0.25,
50 active sessions, 15 target tokens per second per user, 750 demand tokens per second) are
inherited unchanged from the text campaign so the two reports differ only in what the
prompts contain; they describe a hypothetical deployment, and the result section confronts
them directly.

The SLO gates were fixed before the run: TTFT p95 no greater than 2.5 s, ITL p95 no greater
than 0.0667 s (the 15 tok/s per user target), end-to-end p95 no greater than 120 s, error
rate no greater than 1 percent. A rung passes only if all three repetitions satisfy every
gate -- the rung is graded on its worst window, while the published row shows the lower
median, which is how concurrency 7 can pass its row and fail its rung, two sections on.

## The result

**6 sustained request streams at 250.00 output tokens per second, bound by the SLO floor.**
That is the `sustainable` tier, and it is the number to size against -- with the demand
confrontation from the opening attached: 250.00 tok/s meets 33.3 percent of the declared
750 tok/s, and no rung both approaches demand and keeps its promises. The `measured` tier
is 128 streams at 833.33 output tok/s, but that number is peak observed throughput at an
operating point the benchmark itself rejects: the same rung shows TTFT p95 of 53.8161 s
against the 2.5 s gate. Its 144,000 requests/day is an arithmetic equivalent, not a
supportable operating point. The two tiers differ by a factor of about twenty-one in
concurrency and 3.3 in throughput.

| tier | capacity | throughput | binding constraint | provenance |
|---|---:|---:|---|---|
| `sustainable` -- last rung meeting every gate | 6 streams | 250.00 tok/s, 0.5 req/s, 43,200 requests/day | `slo` | M |
| `measured` -- peak observed, SLO-failing | 128 streams | 833.33 tok/s, 1.6667 req/s, 144,000 requests/day equivalent -- not a supportable operating point | `slo` | M |
| `theoretical` | null | -- | null, with (U) reasons | U |
| `recommended` | null | -- | null, with (U) reasons | U |

Both daily figures apply a full 86,400-second day with no haircut for the declared duty
cycle of 0.25; a duty-cycled figure would be a quarter of these.

| concurrency | TTFT p95 (s) | TTFT p95 spread | output tok/s | req/s | error rate | rung SLO | outcome |
|---:|---:|---:|---:|---:|---:|:--|:--|
| 4 | 1.6017 | 17.38% | 183.33 | 0.367 | 0.0 | pass | complete |
| 5 | 1.8629 | 7.77% | 229.17 | 0.458 | 0.0 | pass | complete |
| 6 | 2.2458 | 15.00% | 250.00 | 0.500 | 0.0 | pass | complete |
| 7 | 2.4452 | 31.49% | 291.67 | 0.583 | 0.0 | pass | failed |
| 8 | 2.7197 | 11.70% | 333.33 | 0.667 | 0.0 | fail | failed |
| 16 | 4.6955 | 1.10% | 466.67 | 0.933 | 0.0 | fail | failed |
| 32 | 8.6838 | 8.27% | 666.67 | 1.333 | 0.0 | fail | failed |
| 64 | 15.9451 | 0.78% | 800.00 | 1.600 | 0.0 | fail | failed |
| 128 | 53.8161 | 5.72% | 833.33 | 1.667 | 0.0 | fail | failed |

Each row is one real repetition, not an aggregate: the harness ranks the rung's three
counted windows by output tok/s and breaks ties on `ttft_p95_s`, then publishes the median
one, so every figure in a row is mutually consistent because one window exhibited all of
them. On this ladder every rung's three windows tie on throughput -- `ignore_eos` with a
declared 500-token output quantizes it -- so the tiebreak decides at every rung and the
published TTFT is the lower median of the three. Do not carry that shorthand to another
bundle: where throughput does move between windows, the row's TTFT is whatever the
throughput-median window measured, which can be the fastest or the slowest of the three.
The spread column is the relative dispersion of the three windows, published in the row's
`dispersion` block. Concurrency 7 is the row to read twice: its `slo_pass` is true and its `outcome` is `failed`, and the dispersion section
below is the explanation of how both are true.

The failure story is the TTFT gate, and it binds early. Every rung from concurrency 8
upward fails it on its published row -- 2.7197 s at concurrency 8 is a 9 percent miss, and
the miss only widens from there -- and concurrency 7 fails it in one window of three, the
boundary case the dispersion section gives its own paragraph. The sustainable rung is
therefore 6, and it is confirmed: the harness's own confirmation window at concurrency 6,
run after the declared ladder finished, measured 1.9738 s, inside the gate. That window is
the one rung the harness adds on its own, and it is additional evidence about the boundary
and is not counted in the rung's spread.

Two properties of this table carry no signal about serving health, and the README says so
because a reader could over-read them. Throughput climbs monotonically and the error rate
is 0.0 at every rung's median, but both are expected by construction: every completion was
truncated at the 500-token cap under `ignore_eos`, and req/s times 500 reproduces output
tok/s exactly, so more concurrency means more fixed-length completions, and nothing fails.
The SLO and outcome columns are the only pass/fail evidence in the table. By the same
token, decode throughput describes forced 500-token generations over multi-image questions,
not natural-answer behavior.

A reader must not size against 128: it is where the ladder stopped, and where the
deployment had long since stopped keeping its promise. The text campaign's warning carries
over with more force -- a monitor watching tokens per second per user would have shipped
concurrency 7 through 32, where per-user throughput still clears the 15 tok/s target, and
violated the TTFT gate at every one: in one window of three at concurrency 7, and on the
published row itself from 8 upward.

## Every rung now publishes its dispersion -- and concurrency 7 is why

Every rung row now carries a `dispersion` block: for each of `ttft_p95_s`, `itl_p95_s`,
`e2e_p95_s`, `output_tok_s` and `error_rate_pct`, the min, the lower median, the max, the
count of counted windows, and the relative spread against the median. One field refuses
the statistic: `error_rate_pct` carries `spread_pct: null` with a (U) reason on every rung
of this run, because a relative spread against a zero median is a division by zero dressed
as a statistic.

Concurrency 7 is the case worth a paragraph of its own. Its three windows measured 2.2554,
2.4452 and 3.0255 s against the 2.5 s gate, so two passed and one missed by 21 percent.
All three tie on throughput, so the row publishes the middle window by TTFT, 2.4452 s --
inside the gate -- and reports `slo_pass: true`. Its `outcome` is `failed`. Section 5 grades a rung on its worst window
and is right to fail it, but a reader holding only the row sees a figure inside the gate
and a verdict outside it with nothing on the page between them. The dispersion block is
that missing sentence. Before it existed, the only way to recover the third window was to
re-reduce the bundle.

Note also the shape of the spreads: they are widest at the low rungs, where the load is
light and per-window noise dominates -- 17.38 percent at concurrency 4, 31.49 percent at
7 -- and narrowest at the high rungs, where the engine is saturated and every window
measures the same queue: 0.78 percent at concurrency 64, 1.10 percent at 16. A reader
comparing this campaign against another one must not read a difference of a few percent at
the low rungs as a result.

## The media processor cache bought latency and no throughput -- reported as an observation

An earlier configuration of this same campaign ran with vLLM's multimodal processor cache
enabled; the published run has it disabled (`--mm-processor-cache-gb 0`). On the six rungs
the two runs share, the cache bought latency and no throughput at all:

| concurrency | TTFT p95 (s), cached | TTFT p95 (s), published | output tok/s, both |
|---:|---:|---:|---:|
| 4 | 1.5424 | 1.6017 | 183.33 |
| 8 | 2.5639 | 2.7197 | 333.33 |
| 16 | 4.6703 | 4.6955 | 466.67 |
| 32 | 7.9891 | 8.6838 | 666.67 |
| 64 | 15.1995 | 15.9451 | 800.00 |
| 128 | 52.9464 | 53.8161 | 833.33 |

Throughput is identical to the token at every shared rung, and the identity is structural
rather than informative: the ladder is driven at a fixed 500 output tokens under
`ignore_eos` inside a 120 s window, so output tok/s is quantized to whole requests and both
runs land on the same step. The cache moves time-to-first-token by 0.5 to 8 percent and
nothing else.

Note what this does and does not mean. The multimodal processor cache stores decoded and
preprocessed images, not KV, so it saves host-side work when the same image is drawn
twice, and this corpus draws with replacement. It does not make the GPU faster, and it
does not change the capacity answer -- the cached run's concurrency 8 still fails the
gate. The published run has the cache off because a capacity number should not depend on
how often the load generator happens to repeat an image.

The evidence status of this comparison wants stating plainly. The cached run's report and
engine log are not published in this directory: they have no reproduction bundle, and this
framework's own rule is that a report without its bundle is not evidence. The cached
configuration is therefore reported here as an observation, with the numbers above, rather
than published as a report.

## The image-token finding: 280 is a ceiling, not a price

The checkpoint's `config.json` declares `vision_soft_tokens_per_image 280`, and the
text-only campaign transcribed that number into `model.image_tokens_fixed` without
exercising it. Serving says otherwise. Sending one image and subtracting the same prompt
without it, the charged cost is 256 to 274 tokens across 24 distinct resolutions, mean
265.1, and 280 is never charged. A second measurement, server-side over 300 draws weighted
by the corpus's actual image mix, gives 265.23. The declarations reconcile these two ways:
the derived workload fields are computed from the 300-draw corpus-weighted value, so
4.1904 x 265.23 = 1,111.42 is declared as `media_tokens_per_request 1,111.5` after
rounding, and `image_tokens_fixed` carries the flat value 265. Declaring 280 overstates
every image by about 5.6 percent.

The mechanism is legible in the numbers, and it explains why cost depends on aspect ratio
and not at all on pixel count. The preprocessor lays the image on the integer patch grid
R x C whose ratio best matches the image and whose product does not exceed the 280-token
budget, then adds 4 wrapper tokens. A square takes 16 x 16 because 17 x 17 is 289 and
overruns the budget, giving 256 + 4 = 260; a 4:3 image takes 14 x 19 = 266 + 4 = 270; a 2:1
image takes 11 x 23 = 253 + 4 = 257. The independence from size is direct: one picture
resized from 32 x 32 to 2048 x 2048 square costs 260 tokens at every size, and six different
pictures scaled to 512 x 512 all cost 260. Over the whole corpus the measured spread is 252
to 276.5 tokens per image, under 10 percent, which is why this model is the wrong place to
observe a resolution-mix effect and the right declaration for this corpus remains the full
mix, for the reader who will price it on a model that scales with pixels.

## The budget is a deployment choice, and probing it killed the server

One further fact fell out of probing whether the budget can be overridden per request. The
engine's rejection message names the whole ladder of legal budgets: **(70, 140, 280, 560,
1,120)** soft tokens. 280 is the default, 70 is the tier the video path uses, and none of
them is an architectural constant. A deployment that launched on 1120 would charge roughly
four times as much per image with nothing in this file changing -- which is why
`serving.media_preprocessing.image_pixel_budget_px`, not `image_tokens_fixed`, is where a
reader should look for what an image cost on the day.

How this was established matters as much as the finding. A request carrying
`mm_processor_kwargs {"max_soft_tokens": 64}` did not return HTTP 400. The engine logged
"Unsupported max_soft_tokens value: 64. Valid values are (70, 140, 280, 560, 1120)" from
`gemma4_mm.py:483` and then called `sys.exit(1)` inside the processor, which terminated the
API server for every other client on the endpoint. So `per_request_override_supported` is
false in a stronger sense than the protocol contemplates, and the finding generalises past
this model: **a deployment that accepts `mm_processor_kwargs` from untrusted clients has a
one-request denial of service.** No measurement was lost -- the ladder had finished and its
bundle had been pulled when the server died.

## Two budgets the token count does not show

**The multimodal encoder cache.** The engine allocated an 8,192-token encoder cache,
separate from and additional to the 114,544-token KV pool, and sized it for three maximal
video items -- 3 x 2,240 = 6,720 of the 8,192 tokens, leaving a residual 1,472 tokens, about
five images -- even though this run sent no video. At the measured per-image cost of 252 to
276.5 tokens the cache holds 29.6 to 32.5 images, about 31 at the 265 token mean, which is
roughly 7.4 requests worth of this corpus at 4.19 images each. It is the ceiling on how many
images can be in flight through the vision tower at once, and it appears in none of the
report's capacity floors. No saturation symptom -- no error, no latency discontinuity -- was
observed through 128 concurrent users, but cache occupancy was not instrumented, so "it did
not bind" is an inference from the absence of symptoms, demonstrated only up to the
concurrencies this ladder reached.

**The pixel budget.** The preprocessor rescales every image into 645,120 pixels -- 280 soft
tokens times a 48 x 48 pixel cell, which is the 16-pixel patch under the 3 x 3 pooling
kernel that `processor_config.json` declares: 280 x 2,304 = 645,120, about 0.65 megapixels.
This is a budget, not a rejection: the 2048 x 2048 probe image is 4.19 megapixels, six and
a half times the budget, and cost exactly the same 260 tokens as the 32 x 32 one. The corpus
median is well above the budget, so this deployment answers a 2352 x 1695 scanned page after
the scan has been rescaled to about a sixth of its pixels -- and nothing in the token counts
reveals it.

## The host-side media path, implied by the configuration and not measured

A text prompt reaches the GPU as token ids and the host does little but tokenize. A media
prompt is different: every request carries roughly 4.19 base64 JPEGs that the server must
decode, resize and patchify on the Grace cores before the vision tower sees a tensor, and in
vLLM 0.18.2rc1 as configured here that work sits in front of prefill. That is the mechanism
the configuration implies -- **it was not verified in this campaign**, and it cannot be from
this bundle: no GPU utilization, no host/GPU timeline and no phase breakdown of TTFT was
recorded, so the 1.6017 s TTFT at concurrency 4 cannot be split between host media work,
queueing and prefill from anything measured here. Two hardware notes stand regardless. This
tray has 36 physical cores per GPU, which is generous; if host media work is on the critical
path, the same GPU-side capacity can be unreachable on a host with fewer cores per GPU. And
the load generator ran on this same host, so client and server competed for the cores -- the
corpus pre-encodes its media once at construct time (131.1 s to build, 6,124,064,253 bytes
resident) precisely so base64 encoding is not repeatedly charged to the machine under test,
but the JSON serialization of a roughly 1.9 MB request body still is. A deployment with
remote clients does not pay that part.

## Protocol gaps this campaign recorded rather than patched

**The resolution mix cannot be represented.** This corpus has 11,916 distinct resolutions
across its 13,644 images -- almost every image is its own resolution, because these are real
charts, scanned pages and photographs. The 32 rows declared in `image_resolution_mix` are
the most common and cover 4.61 percent of the images, so the fractions sum to 0.046, not 1,
and the schema's SHOULD is knowingly not met: a complete listing would be a 700 KB
declaration no operator can check, and a mean resolution would hide exactly the spread the
field exists for. The bench's `media_shape` now reports `image_resolution_mix_listed_share`
and `image_resolution_mix_distinct` alongside the mix, but the workload schema has no
matching fields, so on this corpus the two disagree about completeness and only the prose
says which is right. The gap is a missing schema term, not a lost measurement, and it
produces no conformance finding: the report's grade is `partial`, and the findings behind
it are listed with the reproduction notes below.

**Audio is accepted and unpriceable.** The checkpoint carries `audio_token_id` and an
`audio_config` key, but `audio_config` is null in these weights and no audio tower is
present, so audio is declared and left unmeasured. The processor specifies 40 ms per token
capped at 750 tokens per clip -- that figure is read from the checkpoint config, not
measured, and ASCEP has no field for any of it. Video is likewise declared from the
checkpoint rather than measured: uniform-count at 32 frames per clip, 70 soft tokens per
frame, hence 2,240 tokens flat regardless of duration -- config arithmetic; no video and no
audio were sent in this campaign. That 2,240-token clip size is what the engine profiled its
encoder cache with at startup.

## What not to conclude

- **Do not size against 128.** That is peak observed throughput from a rung whose TTFT p95
  of 53.8161 s misses its own SLO more than twenty-fold. The last rung meeting every gate
  is 6, and 6 is the quoted capacity; concurrency 7 already fails on its worst window.
- **Do not read the declared demand as met.** No tested point satisfies both the 750 tok/s
  demand and the gates; the sustainable rung delivers a third of demand, and one GB200 at
  TP1 is undersized for this workload as declared.
- **Do not tune this deployment by per-token rate.** TTFT is the gate that binds: it fails
  on the published row from concurrency 8 upward, and in one window of three at
  concurrency 7.
- **Do not read health into the monotonicity or the 0.0 error rate.** Both are guaranteed
  by fixed-length 500-token generations; latencies describe truncated outputs, and the SLO
  and outcome columns are the only pass/fail evidence.
- **Do not read `slo_pass: true` as a passed rung.** The row publishes one median window of
  the three counted; section 5 grades the worst window, and concurrency 7's row passes
  while its outcome is failed. The dispersion block is the bridge between them.
- **Do not read a few percent at the low rungs as a difference between campaigns.** Spread
  is widest where the load is light -- 17.38 percent at concurrency 4, 31.49 percent at
  7 -- and narrowest where the engine is saturated, 0.78 percent at concurrency 64 and
  1.10 percent at 16.
- **Do not treat the media-cache comparison as a published result, or the identical
  throughput as "the cache did nothing".** The cached configuration has no bundle behind
  it and is reported here as an observation; throughput is identical to the token because
  fixed-length generations quantize it to whole requests, and the cache moves TTFT by 0.5
  to 8 percent without touching the capacity answer.
- **Do not quote 280 tokens per image for this model.** The measured means are 265.1 (24
  resolutions) and 265.23 (300 corpus-weighted draws); cost moves with aspect ratio, and 280
  is a default tier selectable from (70, 140, 280, 560, 1,120) -- a deployment choice, not a
  model constant.
- **Do not infer accuracy from token counts.** Everything above about 0.65 megapixels is
  downscaled into a 645,120-pixel budget before the model sees it, silently.
- **Do not extrapolate the KV pool to another engine build.** The uniform-paging behaviour
  and the 114,544-token figure are properties of vLLM 0.18.2rc1.dev73+gdb7a17ecc.
- **Do not scale from one GPU.** TP=1 leaves vision-tower replication unobservable; a
  replicated 570 M-parameter tower costs about 1.1 GiB on every added rank, and the
  inter-node path is declared null.
- **Do not treat the host-critical-path story as measured.** The configuration implies
  host-side media work in front of prefill; nothing here separates it from prefill or
  queueing, and GPU utilization was not recorded.
- **Do not infer video or audio capacity.** Neither was sent; the 32-frame, 2,240-token clip
  and the 750-token audio cap are read from checkpoint configs, not measured.
- **Do not generalise from one corpus and one day.** Molmo2's shape -- 33.5 text tokens and
  4.19 images per request -- is in every number here.

## Corrections this campaign makes to the text bundle's declarations

Because this run actually sent images, it exercised three media fields the text campaign
only transcribed, and two were wrong there. `model.image_tokens_fixed: 280` overstated the
cost by about 5.6 percent -- measured, the means are 265.1 and 265.23 as above.
`video_frame_policy: "n-a"` was an assumption from a text-only run: the checkpoint ships a
`Gemma4VideoProcessor` with `do_sample_frames: true`, `num_frames: 32` and
`max_soft_tokens: 70`, so the policy is uniform-count at a flat 2,240 tokens per clip, and
`input_modalities` was missing `video`. The sibling bundle's files are left exactly as they
were run; the corrections live here.

## What this bundle does not establish

Video capacity: no video was sent, and every video number here is checkpoint config
arithmetic. Audio capacity: no audio was sent, the tower is absent from these weights, and
no field exists for it. Scaling: tensor_parallel 1 only, so the replicated vision tower's
cost per added rank is unobservable and the inter-node path is null. Utilization: no GPU or
host utilization was recorded on either run. Resolution-mix pricing: this model's per-image
cost is nearly flat in resolution, so nothing here prices a model that scales with pixels.
Generality: one corpus, one model, one engine build, one day. And headroom inside the demand
gap: this bundle establishes that one GB200 at TP1 cannot serve the declared 750 tok/s under
these gates -- its sustainable rung delivers a third of it; how much hardware would close
that gap is a forecast this bundle does not make.

## Reproducing this

The bundle is self-contained. `bundle/` contains `bench-config.json`, `declarations/` with
the four pinned layer documents -- byte-identical to the `hardware.json`, `model.json`,
`serving.json` and `workload.json` copies in this directory -- `engine.log` (1,396,721
bytes, inside the bundle and named `engine.log` in the manifest), `environment.json`,
`manifest.json`, `records.jsonl` (35,438,971 bytes) and `run_configs.json`.
`persist.verify_bundle` reports no problems.

**Redaction.** `engine.log` as written by the engine carries the absolute checkpoint path on
the shared cluster filesystem, six times, and one private-range IP address from the
distributed-init line. Both were replaced before publication with `tools/redact_bundle.py`,
which re-seals the manifest and records under a top-level `redactions` key the original
SHA-256 of the file, the replacement text and the occurrence count. The manifest's promise is
therefore not "these are the bytes the engine wrote"; it is "these are the published bytes,
and here is exactly how they differ from what the run wrote". Everything else in the log,
including every scheduler line the KV-pressure reading above rests on, is untouched, and a
reader who re-runs the campaign gets the unredacted path back and can diff the rest byte for
byte.

The report in this directory was re-derived from the bundle by `ascep reduce`, not
hand-edited. That command exists because the reduction that turns raw per-request records
into published rows is code, and code gets fixed; when it is, a bundle must be sufficient
to regenerate the report it backs, or the only way to correct a published figure is to
re-burn the GPU hours. That already happened once here: the reduction gained the dispersion
blocks after this ladder ran, and the published report carries them because the bundle was
re-reduced rather than re-measured. Re-reducing it today reproduces every one of the 9
rungs' figures exactly:

```bash
ascep reduce examples/gb200-gemma4-31b-multi-image/bundle --check
```

`--check` exits 0 when the rebuilt report matches the published one, comparing every figure
and excluding two things: `report_generated_utc`, because a rebuild really is generated at
a new time, and the conformance grade, because a rebuild is an ungraded draft by
construction and the grade is recomputed from the figures by `ascep conformance`.

The report is graded `partial`, written into the file by `ascep conformance --raise`. The
findings are three warnings and no errors: C6 on `capacity_tiers.theoretical` and
`capacity_tiers.recommended`, both legitimately unmeasured and keeping their (U) reasons,
and C8 on the missing `reproduction.container_digest`. There are no C12 findings: every
rung of this `repeats: 3` run publishes its dispersion.

Re-running the ladder is one command:

```
ascep bench bench.json
```

Reproduction needs the same engine build, the same tray, `--limit-mm-per-prompt image=16`,
and `mm_processor_cache_gb` set to 0 -- the latter two are recorded only in `serving.json`
prose, so read it before launching.
