# `gb200-qwen25-vl-32b-video` -- one GB200, 1,000 clips, and a ceiling no floor predicted

> *"How many concurrent video-analysis users can one GB200 carry -- and what happens when the
> thing that stops you is not memory, not compute, and not any floor the protocol models?"*

> **CORRECTION (2026-09-04): every throughput figure on this page was produced by a
> phase-locked fleet and is unreliable in a direction this bundle cannot show.** The bench
> driver ended warm-up at a barrier and stamped the window's start on the next line, so
> every virtual user in a rung entered the measured window at the same instant. The workload
> is built to be deterministic -- fixed output length, constant think time -- so
> every user's request-plus-think cycle has the same length, the fleet stays locked for the
> whole window, and completions arrive in synchronised waves. The reducer credits a
> completion to the window it finished in, so in these 120-second windows a locked fleet
> counts floor(window / cycle) completions per user where a rate must divide by window /
> cycle: the published throughput is a staircase, not a rate. The signature is in this
> campaign's own numbers -- 2:12.00, 4:11.00, 6:10.00, 8:9.00, 10:8.00, 12:7.08,
> 16:6.88, 24:5.00, 32:4.00. This ladder tops out at concurrency 32, and that top rung is
> also the coarsest count on the page: exactly four cycles counted per user, a count
> consistent with any true value from 4.00 cycles up to just under 5.00 -- and it sits
> at the rung the measured tier is read from. Two rungs break the integer pattern: 12 and
> 16 came out at 7.08 and 6.88 cycles per user -- the only non-integer rungs anywhere in
> the audit that the connection-pool cap does not already explain -- because variable video
> decode cost partially broke the determinism there. Do not
> read those two rungs as spared: partial de-phasing by accident is not a de-phased
> measurement -- the offsets were not drawn uniformly, not seeded, and not recorded,
> so neither artefact can be bounded for them, and the fleet in any case re-locks to exact
> integers at 24 and 32. An operator sizing from this curve would be sizing against
> artefacts. Two artefacts are entangled in every one of these figures, and they pull in
> opposite directions: the counting artefact, floor in place of ratio, biases the figure
> down, while a locked fleet hands the engine perfectly aligned batches -- cheaper to
> serve than the mixed-phase traffic real users generate -- which biases it up. In the
> de-phased re-run of the 26B MoE ladder, the rung whose window fit 8.00 cycles almost
> exactly printed 2,301.0 and 2,330.25 tok/s where the locked run printed 2,496.0 every
> time -- about 7 percent lower at the very rung the staircase had cost nothing. Which
> term dominates is per rung and no rung's true value can be recovered from this bundle,
> so the published figures must not be read as merely conservative. TTFT, ITL and
> end-to-end percentiles are per-request measurements and are unaffected, so the SLO gate
> verdicts on this page stand. The bench harness is fixed: de-phasing is on by default, and
> the interval is recorded per window in run_configs.json, so any future bundle declares
> which regime it ran in. This campaign is queued for re-measurement; until it is re-run,
> treat every throughput and requests-per-second figure below as withdrawn, and read the
> latency figures and the SLO verdicts as standing.

This is the video campaign on the same tray, the same checkpoint and the same engine build as
[`../gb200-qwen25-vl-32b-multi-image/`](../gb200-qwen25-vl-32b-multi-image/). Only the modality
changed: 1,000 H.264 clips of egocentric robot manipulation footage, one clip per request,
sampled to 16 frames each.

The headline is a number, and it is small. **Six concurrent users.** Not because the GPU ran
out of memory -- the KV pool had room for roughly 57 requests of this size, nine and a half
times more. Not because throughput collapsed -- it rose monotonically to the top of the ladder
and was still rising when the ladder ended. Not because anything errored -- the error rate is
0.00 at every rung of every repetition. Capacity ran out because time-to-first-token crossed
its gate, and it crossed it while every other gate still had multiples of headroom.

That gap between what the memory arithmetic predicts and what the clock permits is the finding.
An operator who sized this deployment from the KV floor alone would have provisioned one GPU
for 57 users and shipped a service that misses its latency target from the seventh user onward.

There is a second finding underneath it. Before any of ASCEP's four capacity floors could bind,
the engine refused the very first video request outright, from a memory pool the protocol has
no field for. Section "Finding 2" is that story.

| path | what it is |
|---|---|
| `bench.json` | the campaign spec: endpoint, ladder, windows, and the SLO gates declared before the run |
| `hardware.json` | the hardware layer declaration |
| `model.json` | the model layer declaration, with the video token rule derived and then confirmed |
| `serving.json` | the serving layer declaration, including the encoder-cache finding |
| `workload.json` | the workload declaration, measured over all 1,000 corpus records rather than a sample |
| `report.json` | the report, re-derived from the bundle by `ascep reduce` and graded `partial` |
| `bundle/` | the measured bundle the report grades: config, pinned declarations, records, manifest, and `engine.log` |

Every summary number in this README must be recoverable from the report and the bundle. Where a
figure comes from a side measurement rather than the ladder, the text says so.

## What was served

Qwen2.5-VL-32B-Instruct, dense bf16, 33,452,718,336 parameters including a
688,841,984-parameter vision tower, on one GB200 at tensor parallel 1. vLLM
`0.18.2rc1.dev73+gdb7a17ecc`. The launch flags that matter:

```
--max-model-len 32768
--gpu-memory-utilization 0.90
--limit-mm-per-prompt   {"image": 16, "video": 2}
--media-io-kwargs       {"video": {"num_frames": 16}}
--mm-processor-kwargs   {"max_pixels": 1003520, "min_pixels": 3136}
--max-num-batched-tokens 32768
--max-num-seqs 256
```

The engine reported a 396,016-token KV pool in 96.68 GiB, an encoder cache budget of 32,768
tokens, and a maximum concurrency of 12.09x for requests at the full 32,768-token context. Cold
start to `Application startup complete` was 117 s.

The pool arithmetic checks out in the direction that can be checked. This checkpoint's geometry
gives 2 x 2 x 8 x 128 x 64 = 262,144 KV bytes per token, and 396,016 x 262,144 is
103,813,218,304 bytes, or 96.6837 GiB, which is the 96.68 GiB the engine printed. Run it that
way and not the other way: the printed figure is rounded to two decimals and the token count is
not, so dividing back out lands a few tokens short and looks like a discrepancy that is not
one.

## What a video costs this model, measured on every record

All 1,000 corpus records were sent to the live server twice before the ladder ran -- once with
the clip attached and once with the text alone -- so the difference is the server's own media
cost with the chat template held constant. Zero errors. This is the whole corpus, not a sample,
so the figures below are exact population values rather than estimates:

| quantity | min | median | mean | max |
|---|---|---|---|---|
| media tokens | 3,130 | 7,346 | 5,798.73 | 7,346 |
| text tokens | 524 | 547 | 540.42 | 662 |
| prompt tokens | 3,654 | 7,893 | 6,339.15 | 8,008 |
| clip duration, s | 11.233 | 57.550 | 66.521 | 240.400 |

The media cost is **exactly bimodal**. There are two resolutions in this corpus and only two:
633 clips at 960x768, every one of which costs exactly 7,346 media tokens, and 367 at 640x480,
every one of which costs exactly 3,130. The within-group variance is zero across all 1,000
measurements.

That price was predicted from the model's own geometry before it was measured, and the
prediction was exact. A 960x768 frame rounds under `smart_resize` to 756x952, a 27 by 34 grid
of merged cells, so 918 per frame; 16 frames merge pairwise into 8 temporal groups, and
`8 x 918 = 7,344`, plus the 2 vision boundary tokens the server wraps around every media
item, for 7,346. A 640x480 frame rounds to 476x644, a 17 by 23 grid of 391 cells, and
`8 x 391 + 2 = 3,130`. The same boundary pair was measured on this checkpoint in the
multi-image campaign, which is the cross-check that the two bundles describe one model rather
than two coincidences.

## What was run

The ladder is concurrency 2, 4, 6, 8, 10, 12, 16, 24, 32, three repetitions each, 120 s
measurement windows, 120 s drain deadline, 39.94 s of warmup, 500 output tokens per request
with `ignore_eos`, 1.5 s think time, seed 20260903, `unique-prefix` cache policy, percentiles
by Hyndman-Fan type 7, closed loop.

The SLO gates, all fixed before the first window:

| gate | value | why |
|---|---|---|
| `ttft_p95_max_s` | 4.0 | raised from the sibling campaigns' 2.5 |
| `itl_p95_max_s` | 0.0667 | unchanged: 15 tokens per second per user |
| `e2e_p95_max_s` | 120 | unchanged |
| `error_rate_max_pct` | 1.0 | unchanged |

The time-to-first-token gate is the one that differs from the two multi-image bundles, and the
reason is an application argument rather than a results argument. A user who has just submitted
a 58-second clip is waiting for a video to be ingested and understood, not for a chat reply,
and 2.5 s is a chat number. The other three gates are unchanged so that latency-per-token and
error tolerance stay directly comparable across all three bundles on this tray.

Two disclosures about how the ladder was placed, because both could otherwise look like the
gate was fitted to the data.

First, a placement probe ran before the ladder: four rungs, 45 s each, 128 output tokens,
measuring time-to-first-token at concurrency 4, 8, 16 and 32. It returned p95 figures of 2.50,
4.26, 8.03 and 15.39 s. That probe is why the ladder tops out at 32 instead of the sibling
campaigns' 128 -- a ladder whose rungs all fail measures nothing. It did not move the gates,
which were written into `bench.json` before it ran.

Second, and more honestly: a 4.0 s gate does land close to where the probe said the boundary
would be. A reader is entitled to be suspicious of that. The defence is that the whole ladder
is in the report and the gate is one line of `bench.json`, so anyone can re-grade it. At a
2.5-second gate this deployment sustains 4 users; at 15 seconds it sustains all 32 and the
ladder finds no boundary at all. The number that is robust is not "6" -- it is the shape of the
curve below, and the fact that only one of the four gates ever moves.

## The ladder as published

| concurrency | TTFT p50 | TTFT p95 | ITL p95 | e2e p95 | prefill tok/s | output tok/s | errors | gates |
|---|---|---|---|---|---|---|---|---|
| 2 | 1.2758 | 1.8711 | 0.01405 | 8.637 | 1,266.6 | 100.0 | 0.00 | pass |
| 4 | 1.5516 | 2.4824 | 0.01455 | 9.710 | 2,269.1 | 183.3 | 0.00 | pass |
| **6** | **2.2039** | **3.2738** | **0.01522** | **10.892** | **3,184.3** | **250.0** | **0.00** | **pass** |
| 8 | 2.6407 | 4.1295 | 0.01573 | 12.006 | 3,942.1 | 300.0 | 0.00 | fail |
| 10 | 3.0593 | 5.0730 | 0.01673 | 13.722 | 4,364.1 | 333.3 | 0.00 | fail |
| 12 | 3.6844 | 6.1430 | 0.01715 | 14.645 | 4,626.2 | 354.2 | 0.00 | fail |
| 16 | 4.5477 | 7.6251 | 0.01814 | 16.517 | 5,779.8 | 458.3 | 0.00 | fail |
| 24 | 6.5108 | 11.1208 | 0.02217 | 21.954 | 6,512.8 | 500.0 | 0.00 | fail |
| 32 | 7.2801 | 13.5060 | 0.02340 | 26.557 | 6,616.4 | 533.3 | 0.00 | fail |

Every column is monotone in concurrency, and `monotonic_across_ladder` is true. Concurrency 6
passed and was re-confirmed by a separate confirmation window (60 requests, gates held);
concurrency 8 failed. The harness therefore publishes a sustainable tier, which the sibling
multi-image campaign on this same checkpoint could not do.

| tier | users | tok/s | req/s | daily requests | binding | provenance |
|---|---|---|---|---|---|---|
| theoretical | -- | -- | -- | -- | -- | (U) |
| measured | 32 | 533.33 | 1.0667 | 92,160 | slo | M |
| sustainable | 6 | 250.00 | 0.5000 | 43,200 | slo | M |
| recommended | -- | -- | -- | -- | -- | (U) |

The measured tier is an engine ceiling, not a service level: at 32 users time-to-first-token
p95 is 13.5 s. It is the fastest this GPU went, and it is not a number to size against.

## Finding 1: one gate binds, and it binds ten times earlier than memory does

Of the four SLO gates, exactly one ever fails, at every failing rung, for the whole ladder. At
the top rung, concurrency 32:

* `ttft_p95_s` is 13.5060 against a 4.0 gate -- **338 percent of budget**
* `itl_p95_s` is 0.02340 against 0.0667 -- 35 percent of budget
* `e2e_p95_s` is 26.557 against 120 -- 22 percent of budget
* `error_rate_pct` is 0.00 against 1.0 -- zero

Inter-token latency never gets within a factor of two and a half of its gate. This deployment
generates tokens fast and starts slowly, and the slow start is the entire capacity story. At
the boundary rung each of the 6 users receives 41.7 tokens per second, nearly three times the
15 tokens per second the workload asks for -- and capacity is nonetheless exhausted, because
those users waited 3.3 s at p95 to see the first one.

Now compare that against the memory floor. At the boundary rung the measured mean context is
6,881.3 tokens, and the engine's KV pool holds 396,016. That is room for 57.6 concurrent
requests of this size. The measured boundary is 6. **The KV floor overestimates the servable
concurrency of this deployment by a factor of 9.6.**

This is the case ASCEP's capacity-is-the-minimum-of-the-floors rule exists to catch, and it is
worth being precise about what the rule does and does not do here. The rule is correct: the
report names `slo` as the binding constraint at both measured tiers, not `kv`. But the four
floors ASCEP models -- weights, KV, prefill, throughput -- would all have said this GPU was
comfortable. The SLO gate is what caught it, and on a media workload the SLO gate is not a
refinement on the floors. It is the answer.

The mechanism is visible in the TTFT p50 column. At concurrency 2, with essentially nothing to
queue behind, p50 time-to-first-token is 1.2758 s. That is the floor: host-side demux, H.264
decode, frame sampling, patchify, vision tower, prefill, for one request. By concurrency 6 the
p50 is 2.2039 s, so roughly 0.93 s of queueing has been added; by 32 it is 7.2801 s. Prefill
throughput rises with concurrency the whole way -- 1,266.6 to 6,616.4 tok/s -- so the stage is
not stalling, it is saturating, and each arriving request waits behind more work than the last.

## Finding 2: the encoder cache is a fifth floor, and it refused the first request

The first video request sent to this engine did not run slowly. It returned HTTP 400:

```
video item with length 19552 exceeds the pre-allocated encoder cache size 8960
```

The encoder cache is a pool distinct from the KV cache, holding vision-tower output for items
encoded but not yet consumed by prefill. vLLM sizes it from `--max-num-batched-tokens`, so the
default configuration gave 8,960 tokens, and the engine measured the refused clip at 19,552 --
2.18 times the budget, on its own, with nothing else in flight. Note the unit: 19,552 is the
encoder cache's own accounting of the item, not the 7,346 post-merge media tokens the language
model then sees for a 960x768 clip, and this bundle did not establish the conversion between
the two. What it does establish is that one clip overran the default budget by itself, so no
batching or profiling-headroom argument is needed to explain the refusal. Relaunching with
`--max-num-batched-tokens 32768` raised the encoder cache to 32,768 tokens and made the
workload servable.

The cost of that fix is measurable, and it was measured, because the multi-image campaign ran
on the identical checkpoint, tray, and `gpu_memory_utilization` and reported a
**415,584-token** KV pool. This campaign reports **396,016**. Raising the batched-token budget
cost **19,568 KV tokens, 4.78 GiB, 4.71 percent of the original 415,584-token pool** -- or 4.94
percent of the 396,016 that remain. The denominator has to be named, because the two differ.

Two consequences, and they point in opposite directions:

* ASCEP's KV floor is computed against a pool that a media flag can move by 4.71 percent
  without a single field in the serving schema changing. The schema has no field for the
  encoder cache.
* More sharply: on this workload the encoder cache refuses requests while the KV cache is
  empty. A capacity estimate that checked only weights, KV, prefill and throughput would have
  predicted a healthy server and received a 400 on request one. That is not a floor being
  mis-estimated; it is a floor the protocol does not have.

Recorded here as a protocol gap rather than patched by inventing a schema field inside an
example, which is the precedent the two multi-image bundles set.

## Finding 3: the mean media cost describes no request in the corpus

The declared `media_tokens_per_request` is 5,798.73. No request costs anything near it. The
corpus contains requests costing exactly 7,346 tokens and requests costing exactly 3,130, in
proportions 0.633 and 0.367, and nothing in between. The mean is arithmetically correct --
0.633 x 7,346 + 0.367 x 3,130 = 5,798.73 -- and descriptively empty. A reader sizing KV from it
is sizing for a request that does not exist: relative to the mean the true cost is plus 26.7
percent for the 7,346-token majority, which is therefore under-provisioned, and minus 46.0
percent for the 3,130-token minority, which is over-provisioned.

ASCEP already knows this failure mode: `image_resolution_mix` exists precisely so that a mean
cannot hide a spread, and its schema description says so. **There is no video equivalent of
it.** On this workload the frame resolution is the entire cost story, and the only place it can
be declared is prose. For the record, the mix is 960x768 at 0.633 and 640x480 at 0.367, summing
to 1.000 -- which is itself notable, since both multi-image campaigns declared truncated mixes
covering under 5 percent of their images. Here the mix is exhaustive and would have passed the
sum-to-one rule cleanly, if the field existed.

The ladder's own numbers confirm the declaration was right. Measured mean input tokens per rung
range from 6,202.8 to 6,514.1 against the declared 6,339.15, and that spread is exactly what a
two-point distribution with a 4,216-token gap produces when sampled 24 to 128 times per window.
The declaration is validated by the run; the mean is still the wrong thing to hand a reader.

## Finding 4: clip duration drives host work but not one token of model cost

Durations in this corpus span 11.233 to 240.400 s, a 21.4-fold range, at 30 fps native
throughout. Model cost is completely indifferent to it: the shortest clip and the longest clip
are both 960x768 and both cost exactly 7,346 media tokens. Under the deployed uniform-count
policy, duration cannot reach the model at all.

The host feels it in full. At 30 fps the mean request has 1,995.6 frames demuxed and decoded so
that 16 can be kept, and the median has 1,726. Over 99 percent of the decode is discarded, and
the discarded fraction grows with clip length while the GPU-side cost stays flat. An operator
sizing a video service from GPU-side figures alone will therefore mis-size the host, and the
error grows with the length of the footage. This tray's 36 physical Grace cores per GPU is a
generous ratio; a box with fewer cores per GPU could go host-bound before the GPU does.

None of the latency figures in this bundle separate host decode from accelerator time. No
host-side profile was taken. `ttft` here is end to end, and reading it as GPU time would credit
the GPU with work these CPUs did.

One consequence for the declaration: the realized sampling rate varies 21.4-fold across one
corpus under one unchanged policy -- 0.0666 fps on the 240.4 s clip, median 0.2780, mean
0.3207, and 1.4243 fps on the 11.2 s clip. The longest clips are summarized at one frame every
15 seconds. That is a real quality property of the workload, and neither the workload schema
nor the serving schema has a field for a realized-rate distribution; both `video_sampling_fps`
fields mean a *chosen* rate, and no rate was chosen here. Second protocol gap, same disposition
as the first.

## Finding 5: an image-era check correctly diagnosed a video frame budget

`ascep.capacity.media_token_cap_check` was written to detect a hidden per-image token cap by
looking for token counts that stay flat while input magnitude grows. Run on this corpus with
duration as the magnitude, it returns:

```
capped=True -- token count is flat across a 21.4x size range: 11.233333333333333 -> 7346
tokens vs 240.4 -> 7346 tokens (relative difference 0.0% < tolerance 10.0%); the
preprocessor is capping media tokens
```

Which is exactly right, and was not designed for this case. The uniform-count frame budget *is*
a cap, and the check found it.

It could have been fooled, and it is worth saying how. The check compares only the extreme
magnitudes, and on this corpus a second cost driver -- frame resolution -- varies freely and
independently of duration. The two extreme-duration clips happened to share a resolution. Had
the longest clip been 640x480, the check would have compared 3,130 against 7,346, reported a 57
percent difference, and concluded "no cap detected" while the cap was fully in force. To rule
that out, the check was re-run on the 640x480 subset alone, holding resolution constant, and it
returns `capped=True` there too. The diagnosis is not an artifact of which clips sat at the
extremes.

`media_arrival_check` returns `capped=False`: measured prompt 6,339.152 tokens against the
text-only prediction 540.424, a 1,073.0 percent difference. The media unambiguously arrived.

Both are offline helpers. **Neither is invoked by `ascep bench`**, so neither appears in
`report.json`; they were run by hand against the pre-ladder measurement and are reported here
as such.

## Dispersion: how repeatable this is

Every rung is three repetitions, and the report carries min, median, max and spread for each.
Time-to-first-token p95 spread across repetitions runs from 1.86 percent (concurrency 8) to
10.88 percent (concurrency 4). Output throughput spread is 0.00 percent at six of the nine
rungs, 7.06 percent at concurrency 12 and 13.64 percent at concurrency 16.

Those zeros are not suspicious and they are not precision. With `ignore_eos` and a fixed 500
output tokens, throughput is quantized by the completed-request count: 24 completions in a
120-second window is exactly 100.0 tok/s, and one more or fewer request moves it by 4.17 tok/s.
At the lower rungs the count repeated exactly. The two rungs with real spread, 12 and 16, are
where scheduling variation first exceeds one request per window -- the same pattern the
multi-image campaign found, where dispersion was widest near the interesting region.

The boundary rung, concurrency 6, has 3.01 percent spread on the gate that decides it, and its
worst repetition still passes. That is why the confirmation run was able to confirm.

## Conformance

Graded `partial` by `ascep conformance`. No errors; two warnings:

* **C6**, on `capacity_tiers.theoretical` and `capacity_tiers.recommended`: both are left
  unmeasured with (U) reasons, which the rule accepts as legitimate. `ascep bench` observes an
  HTTP endpoint; the roofline and the headroom policy are not things a load generator can see.
* **C8**, on `reproduction.container_digest`: absent. The serving container is a local enroot
  root filesystem with no image manifest or registry digest on disk, so there is nothing honest
  to publish. This caps the report at `partial`, and it should.

The archetype is `video_grounded`, and C9 and C10 -- archetype and term consistency, and
archetype-correct estimators -- pass.

## The bundle

`bundle/` is what the report grades. `ascep reduce --check` rebuilds the report from it and the
rebuild matches the published `report.json` field for field, with only `report_generated_utc`
and the conformance grade excluded.

* `bench-config.json`, 1,292 bytes -- the spec as run
* `declarations/` -- the four declaration files pinned at run time
* `engine.log`, 1,280,321 bytes as published -- see the redaction note below
* `environment.json`, 779 bytes
* `manifest.json`, 1,252 bytes -- sha256 over every artifact, plus the redaction record
* `records.jsonl`, 27,554,967 bytes -- one line per request
* `run_configs.json`, 20,856 bytes

`engine.log` was redacted before publication with `tools/redact_bundle.py`: 7 substitutions
replacing the absolute checkpoint path with `/models/Qwen2.5-VL-32B-Instruct` and one
private-range address with `REDACTED-NODE-IP`. The manifest records the original digest, the
replacement strings and the occurrence counts, and never the removed strings. `verify_bundle`
returns clean over the published bytes, and `tools/check_no_secrets.py` reports no findings.

The corpus itself is not published: 1,000 clips, 4,381,298,125 bytes, third-party footage. It
is pinned by sha256 `340b10426a6a9e693736152b19c446b61ae865d9da03fa6cfc6214ba1339455e` and by
record count in the workload declaration, and `corpus.jsonl` and `media/` are gitignored for
the same reason in every campaign in this directory.

## Reproducing this

You need the checkpoint, a corpus of clips, and one GB200 or equivalent.

```bash
vllm serve <checkpoint> \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 256 \
  --limit-mm-per-prompt '{"image":16,"video":2}' \
  --media-io-kwargs '{"video":{"num_frames":16}}' \
  --mm-processor-kwargs '{"max_pixels":1003520,"min_pixels":3136}' \
  --max-num-batched-tokens 32768

ascep bench bench.json
ascep conformance report.json --raise
ascep reduce bundle --check
```

Two things will bite anyone reproducing this on different footage. `--media-io-kwargs` is what
makes the per-request cost constant; leave it off and cost tracks duration instead, and every
number in this bundle changes. And `--max-num-batched-tokens` must be large enough for your
largest clip's vision-token count or the server returns HTTP 400 rather than running slowly --
compute that count from the frame geometry before launching, not after.

## What this bundle does not establish

* **Nothing about multi-GPU.** One GPU, tensor parallel 1. The vision tower's behaviour under
  tensor parallelism is unobservable at one rank, and `vision_encoder_replicated_per_rank` is
  declared (U) for that reason.
* **Nothing about other frame policies.** Every figure here is under uniform-count at 16
  frames. A rate-based policy makes context length a function of clip duration and invalidates
  the entire cost model in section "What a video costs this model".
* **Nothing about the host/GPU split.** No host-side profile was taken. The claim that decode
  is a large share of TTFT is an inference from the workload's shape, not a measurement.
* **No theoretical or recommended tier.** Both are (U). The roofline needs `ascep size`; the
  recommended tier needs a headroom policy, which is a planning decision and not a measurement.
* **No answer at other resolutions.** Both corpus resolutions sit below the 1,003,520-pixel
  clamp, so the clamp never bound. On higher-resolution footage it would bind silently, by
  downscaling rather than rejecting, and nothing in the token counts would reveal it.
* **The 4.0 s gate is a judgment, not a measurement.** It is declared, and it is the single
  input most able to move the headline number. See the disclosure in "What was run".

## Demand, restated

The workload declares 200 concurrent users at a 0.25 duty cycle -- 50 active sessions --
wanting 15 tokens per second each, for 750 tok/s of demand. That declaration is carried
unchanged from both multi-image campaigns on this tray so the three bundles differ only in
modality and in what the media costs. Nothing in the ladder measures it.

Set against a sustainable tier of 6 users and 250.00 tok/s per GPU, the arithmetic a reader can
do is: 750 tok/s of demand needs 3 GPUs on the throughput term, and 50 active sessions at 6 per
GPU needs 9. The concurrency term binds, and it binds three times harder than the throughput
term -- which is the same thing Finding 1 says, restated in GPUs.

That figure is deliberately not published as a sizing result. `sizing_result` in the report is
(U), because a real one needs a declared headroom policy and `ascep size`, and a number
produced by dividing two measurements in a README is arithmetic, not a capacity claim.
