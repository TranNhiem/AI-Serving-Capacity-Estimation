# 9. Multimodal and reasoning-mode capacity

This chapter extends the capacity model of [chapter 5](05-capacity-model.md) to workloads that carry images or video and to models that reason before they answer. It adds no fourth floor and no fifth tier. What it adds is a set of declarations, one mandatory calibration, and two run-validity checks, because the measured evidence is that the failures in this territory are silent: a multimodal benchmark that is quietly a text benchmark, a thinking-mode throughput figure that counts tokens no user ever received, a context length that was capped by a preprocessor default nobody declared. Every measured number below was read off an 8x H100 vLLM deployment serving Qwen3-VL-class checkpoints, and each is cited because it forced a rule. Where an illustration has no measurement behind it, it is marked as illustrative arithmetic, exactly as elsewhere in this protocol.

## 9.1 The governing rule

An image's, a clip's, or a thinking trace's token cost is MEASURED or ENGINE-REPORTED, never assumed. This is the same move the protocol already makes for the KV pool, where the engine-reported figure is documented as the number to trust, and the reason is stronger here: no client-side tokenizer can predict image expansion, because the expansion is a product of the checkpoint's vision geometry and the server's preprocessing settings, neither of which is visible from the image file. For a media workload the server's own `prompt_tokens` is the only trustworthy count, and a report MUST treat it as such.

ASCEP supplies an analytic predictor — `image_tokens` and `video_tokens` in `ascep.capacity` — and then REQUIRES that the predictor be calibrated against the server's reported `prompt_tokens` on a sample of the real corpus, with the divergence published. The predictor exists to project and to cross-check, never to override measurement; that ordering is not negotiable.

The reason this is a rule and not a suggestion is measured. The same video corpus, sampled the same way, cost ~452 prompt tokens per sampled frame on one checkpoint and ~1,169 on another. Carrying the first checkpoint's tokens-per-frame number across to the second is a 2.6x error in `avg_context_tokens`, and because context length decides which floor binds, it is a 2.6x error applied at exactly the place where the capacity answer is most sensitive. A tokens-per-image figure is a property of one checkpoint and one preprocessor configuration. It does not travel.

## 9.2 There is no fourth floor

A reader confronting a vision-language model for the first time is tempted to add a "vision floor" to the three of §5.3. Do not. Every cost this chapter introduces lands inside the existing floors, and naming the landing spot is most of the work.

The vision tower lands on the **weights floor**. Its parameters are resident in VRAM exactly like the language model's, and they enter the fit check through `vision_encoder_params`. There is one genuinely new weights term, and it is a trap: on the deployments measured here the vision encoder is replicated on every tensor-parallel rank rather than sharded, so the cluster pays for it once per rank instead of once in total and tensor parallelism buys nothing. `vision_encoder_bytes` returns the **per-GPU** figure: the full `params` times the precision width when `replicated_per_rank` is true, and that same amount divided by `tensor_parallel` when it is false. `vision_encoder_replicated_per_rank` is the declaration that selects the branch. Read the return value as bytes on each GPU, not bytes across the deployment; the two differ by exactly the TP width, which is the same factor the mistake introduces, so the units MUST be checked before the number is added to a fit. Sizing a VLM's VRAM from `total_params` alone, when the encoder is not in that count and is replicated besides, understates the weights floor by the encoder size times the TP width. Check whether the published parameter count includes the tower; do not assume.

Media tokens land on the **KV floor**, because media tokens are prompt tokens. Once expanded, an image's tokens occupy KV for the whole session, exactly like the text prefix they sit beside, and they enter `avg_context_tokens` through `media_tokens_per_request` as prefix — present for the whole session, not averaged like output. A 30 s clip sampled at 1 fps and 256 tokens per frame is 7,680 context tokens — illustrative arithmetic, but representative — an order of magnitude over a chat turn, and usually enough to move the binding floor from throughput to KV.

Reasoning tokens land on the **throughput floor**, because reasoning tokens are generated tokens. They consume decode time whether or not the user ever sees them, so wherever generated tokens drive demand they MUST be counted as output. They also occupy KV as they are generated, so they enter the KV floor averaged like output tokens, per the existing convention of §5.1. Three floors, no fourth; what changes is how much each one is fed.

## 9.3 Declaring the model

Two fields are required and non-nullable, with no `(U)` escape: `input_modalities` and `reasoning_modes`. A report that cannot say whether it is a VLM cannot be interpreted at all, and unlike layer geometry this costs nothing to know — the schema therefore does not permit the unknown. A model supporting both thinking and non-thinking declares both in `reasoning_modes` and MUST be measured and reported per mode, because they are two capacity profiles of one model (§9.6).

`image_token_policy` selects the token predictor exactly as `attention_type` selects the KV formula in [chapter 2](02-model.md), and the schema gates follow the same shape: declaring `image` or `video` in `input_modalities` requires `image_token_policy`, `vision_encoder_params` and `vision_encoder_replicated_per_rank`, and each policy then requires its own geometry.

- `fixed-grid` — every image costs the same number of tokens regardless of resolution. Requires `image_tokens_fixed`. The predictor ignores width and height entirely.
- `dynamic-resolution` — the image is patched, pooled and clamped. Requires `vision_patch_px`, `vision_spatial_merge` and `image_tokens_max`; `image_tokens_min` MAY be declared. The worked identity, measured and confirmed against the checkpoint's own config: `patch_size` 16 with `spatial_merge_size` 2 gives one visual token per 32x32 pixels of input, so a 768x768 frame lands at 576 tokens before clamping and the 768x432 frames measured here at 336. `image_tokens_max` is the field that bounds the worst case; a null there means the worst case is unbounded, and the report MUST say so.
- `declared-table` — the measured expansion at each supported resolution, in `image_token_table`. The predictor matches exactly and refuses to interpolate, because interpolating a measured table is inventing a measurement.

For video, `video_frame_policy` declares how a clip becomes frames: `uniform-fps` (a declared sampling rate times the duration), `uniform-count` (a fixed frame count regardless of duration), or `native-timestamped` (the file's own timestamps decide, so the effective rate is a property of the container and MUST be measured from the corpus and declared as `video_sampling_fps` before it is used; `video_frames` treats the declared rate exactly as it treats `uniform-fps` and cannot read timestamps for you, so an undeclared rate here is a projection nobody verified rather than an error the code will catch). `video_temporal_merge` declares how many frames merge in time before tokenisation; on the measured checkpoint it was 2, read from the same config.

None of these values is exotic. `patch_size`, `spatial_merge_size`, `temporal_patch_size`, `longest_edge`, `max_pixels` and their kin live in the checkpoint's own preprocessor configuration, and reading them takes minutes. A `(U)` on any of them is a statement that nobody opened the config, and the conformance checker treats it accordingly.

## 9.4 Declaring the deployment

Media preprocessing is a serving-layer object, `media_preprocessing`, and not a per-request one, for a measured reason: on the engine measured here, per-request `mm_processor_kwargs` returns HTTP 400, and the video sampling rate is fixed at server start. Two clients hitting one endpoint cannot choose differently. The sampling rate, the frame cap and the pixel budget are therefore properties of the deployment, exactly like the maximum model length, and they are declared once in the serving layer: `video_sampling_fps`, `video_max_frames`, `image_pixel_budget_px`, `mm_processor_cache_gb`, plus `per_request_override_supported`, whose default assumption is false because the engine measured here refuses overrides. The workload layer keeps its own `video_sampling_fps` and `video_max_frames` as a statement of what the application needs — the requested value against the serving block's delivered value. A report where the two disagree without a note is describing a workload the server did not run.

`image_pixel_budget_px` is a **total pixel count** — width times height per image, or the summed frame budget after sampling — and it is named for the quantity, not for the dial that sets it: on the H100 deployment measured here the correct value was 16,777,216 (4096 x 4096), while an operator reading an edge-length name writes 4096, which as a pixel budget is a 64 x 64 image, roughly four thousand times too small. The field is required whenever `input_modalities` contains image or video, nullable only with a `(U)` reason, because it is the field that silently binds: §9.5 is entirely about what happens when it does. The pixel budget is the cause and any observed token ceiling is the effect; a report that declares only the effect cannot be acted on, because the fix is one config value and no hardware.

Two warnings, both measured. First, the multimodal processor cache is prefix caching wearing a different hat. Engines cache processed image tensors, and a benchmark replaying the same images gets a speedup the production workload will never see — worse than the text case, because nothing in the response reveals it. `mm_processor_cache_gb` MUST be declared, and if it is non-zero while the corpus repeats media, measured throughput is inflated and the report MUST say so. Second, transport: `image_input_transport` MUST be declared, because base64 inflates a request body by roughly 4/3 — a 1 MB JPEG becomes ~1.37 MB of JSON per request — and at high concurrency that saturates the client before the server notices. The report then blames the server for a client-side bottleneck, which is precisely the misattribution this protocol exists to prevent.

## 9.5 Calibration, and it is mandatory

The procedure is short and not optional. Predict the media expansion with `image_tokens` or `video_tokens` over a sample of the real corpus. Run that corpus against the server. Compare the prediction against the server's own reported `prompt_tokens`. Publish the divergence. A divergence above 10% MUST be published with the figure, and it MUST NOT be silently corrected by tuning the predictor until the numbers agree — a predictor tuned to match one corpus on one server is a measurement wearing a formula's clothes, and presenting it as a prediction is the exact failure this chapter legislates against.

Two checks then decide whether the run is allowed to become a report at all.

**`media_token_cap_check`.** Measure media token cost at two or more input sizes spanning at least 2x. The measured evidence, same model, same 2 fps sampling, same 768 px frames, three clips ingested full-episode:

| clip | measured `prompt_tokens`, default `longest_edge` | after raising the pixel budget |
|---|---:|---:|
| 94.8 s | 12,333 | 32,853 |
| 47.4 s | 12,345 | 16,293 |
| 39.1 s | 12,090 | 13,533 |

The default budget was 25,165,824 px. The tell is in the left column: a clip 2.4x longer than another produced 2% more tokens. **When measured prompt tokens stop responding to input size, the number being measured is the preprocessor's cap, not the workload.** The consequence is not cosmetic: a capacity model fed the capped column concludes the KV floor is flat in clip length and sizes a cluster for a workload nobody can run, when the fix is one config value and no hardware — raising the budget multiplied the real context by 2.7x on the longest clip. The check takes the measured counts and the input `size_ratio`, and it REFUSES to answer on samples spanning less than 2x, because "we looked and found no cap" and "we could not look" are different claims, and a check that cannot distinguish them reports the second as the first. A capped run MUST say so, naming the binding setting.

**`media_arrival_check`.** Measured: the corpus was AV1, the container's decoder turned every clip into zero frames, and every request succeeded as text with no error raised. A multimodal benchmark that is quietly a text benchmark, publishing a media capacity figure measured on no media. The check compares the server's reported `prompt_tokens` against a text-only prediction for the same records: if `measured_prompt_tokens` comes within 5% of `text_only_prompt_tokens`, the media never arrived, and the run MUST be refused rather than reported. A report is a worse outcome than a refusal here, because the report will be believed.

## 9.6 Reasoning modes

A model with a thinking branch has two capacity profiles, and a report MUST name which one it measured. Reasoning tokens consume decode time exactly like visible tokens and occupy KV as they are generated, but they are frequently not returned in the response content, so a harness counting visible characters under-counts output by the entire reasoning volume. `reasoning_tokens_per_request` carries the measured mean, and the distinction is load-bearing: 0 means measured and none emitted, null means the server did not report the field. Conflating them silently understates output.

The measured cap sweep, one checkpoint, same weights, same server, prompts averaging only ~2,530 tokens, sweeping nothing but the output cap:

| declared output cap | `completion_tokens` per request | requests that never closed the thinking block |
|---|---:|---:|
| control, thinking off | 120 | 0.0% |
| `max_tokens` 8,192 | 10,577 | 74.6% |
| `max_tokens` 24,576 | 19,896 | 59.1% |
| `max_tokens` 57,344 | 33,829 | 46.4% |

From 120 to 33,829 completion tokens is 282x, at prompts under three thousand tokens — this is not memory pressure, and it is not a different model. It is the same model filling whatever budget it is given. Two normative conclusions follow. First, `max_output_tokens` MUST be declared, and is required and non-nullable, for any thinking or mixed workload: the traces expand to fill the budget, so an output length reported without its cap describes the harness as much as the model, and is not reproducible. Second, `truncation_rate` — the fraction of requests that hit the output cap — MUST be reported alongside any thinking-mode throughput figure. At 46.4% truncation nearly half the requests returned nothing usable, and because a truncated request by definition ran all the way to the cap while a completed one stopped earlier, the share of *generated tokens* thrown away is higher still than the share of requests: real GPU work, zero delivered capacity. Averaging truncated and completed requests into one throughput number therefore overstates what the cluster can actually deliver, and the overstatement grows with the truncation rate — which is why the rate MUST travel with the figure rather than being recoverable from it. Both the throughput floor and the KV floor move by roughly the cap-sweep factor, which is why reasoning mode is a declaration and not a footnote.

`reasoning_mode_control` declares how the mode is selected — a chat-template flag such as `enable_thinking` passed through `chat_template_kwargs`, a request field such as `reasoning_effort`, a separate model identifier, or always-on — and it is required whenever `reasoning_modes` contains thinking, because without it a reader cannot reproduce the mode the numbers were taken in. The mixed case is declared, not discovered: `reasoning_mode` of `mixed` requires `reasoning_share`, the fraction of requests served in thinking mode, and the capacity model consumes the two profiles in that proportion. A mixed workload reported as a single undifferentiated run hides exactly the 282x spread the table above measures.

## 9.7 The TTFT trap

For a thinking model, the first streamed chunk may carry the first *reasoning* token while the user is still waiting for the first *visible* token, and those two instants differ by the entire reasoning duration — tens of seconds on the deployment measured here. Two conforming harnesses can therefore report TTFT figures that differ by tens of seconds on the same server, both honest, both useless to a reader who does not know which instant was timed. A report MUST state which of the two its TTFT marks, and SHOULD report both. The measurement mechanics and the gating consequences live in [chapter 4](04-measurement.md); this chapter's contribution is the rule that the ambiguity exists and must be declared away.

## 9.8 What conformance requires

Rule C4 already demands that input and output token counts and the context distribution accompany every throughput figure. For a multimodal or thinking workload, C4 now additionally demands:

- the media shape alongside any throughput figure: `images_per_request` and `image_resolution_mix`, or `videos_per_request`, `video_seconds_per_request` and the declared sampling, plus the delivered `media_preprocessing` values they ran against;
- the reasoning mode and, for thinking or mixed, the output cap that produced the reported lengths, with `truncation_rate` beside any thinking-mode throughput figure;
- the calibration divergence between the predictor and the server's `prompt_tokens`, published, with any divergence above 10% shown as a figure rather than tuned away;
- the cap check with its samples — at least two input sizes spanning at least 2x — and, where the run was capped, the name of the binding setting.

A single capacity number that does not name its mode is meaningless for a model that has both, and a media throughput figure without its media shape is a number about a workload the reader cannot reconstruct.

## 9.9 Run-validity checklist

Before a multimodal or thinking run is allowed to become a report, confirm each of the following. A failure here is a refused run, not a qualified one.

- `input_modalities` and `reasoning_modes` are declared, non-null, and consistent with the checkpoint that was actually served.
- The vision geometry was read from the checkpoint's own preprocessor config, not carried over from another model, and `vision_encoder_replicated_per_rank` was verified against the deployment rather than assumed.
- The serving block's delivered `video_sampling_fps`, `video_max_frames` and `image_pixel_budget_px` match what the workload requested, or the disagreement is noted in the report.
- The predictor was calibrated against the server's `prompt_tokens` on the real corpus, and the divergence is published.
- `media_token_cap_check` passed on samples spanning at least 2x, or the run is declared capped with the binding setting named.
- `media_arrival_check` passed: measured prompt tokens are meaningfully above the text-only prediction, so the media demonstrably arrived.
- `mm_processor_cache_gb` and `image_input_transport` are declared, and any cache-on repeated-media inflation is noted.
- The reasoning mode is named, `max_output_tokens` is declared for thinking or mixed workloads, and `truncation_rate` accompanies every thinking-mode throughput figure.
- The TTFT declaration states which instant was timed — first reasoning token or first visible token — per §9.7.
- Every figure carries its tag, its topology per C3, and its binding floor per C5; nothing predicted is presented as measured.

## 9.10 Saying why a value is what it is

The `(U)` mechanism is one-directional. A `_u_reason` field justifies a null, and C1 enforces it; nothing anywhere justified a value, and `additionalProperties`-closed schemas meant an operator could not invent the slot. The gap is visible in what one H100 VLM campaign had to leave unwritten. `mm_processor_cache_gb` was set to 0 on purpose, so every request paid host-side decode and patchify and the measurement carried preprocessing cost rather than hiding it behind a cache — recorded as a bare 0, indistinguishable from a default nobody thought about, under a convention where 0 already means measured-and-none. And `cpu_cores` was 12 not because the node has 12 cores — it has 112 — but because the cluster's QoS grants 12 per GPU, and with the processor cache off the host was co-limiting: 11.33 of 12 cores busy at concurrency 32 while the GPU sat at 91%. A reader assuming inventory would attribute the ceiling to the GPU. In the same campaign's workload file, a `(U)` reason field reads "not (U): this is a measured value but a single observed datapoint" — the `(U)` slot doing duty for a value-justification, because it was the only writable slot. A convention people have to abuse is a missing feature.

Every declaration layer therefore gains an optional `notes` object, keyed by field name, carrying exactly this: why a declared, non-null value is what it is. Three rules keep it honest. A note never substitutes for a value — a null still needs its `(U)` justification, a note written beside a null does not satisfy C1, and a skeleton still does not validate. A note is not evidence: it is the author's claim, unverified by the conformance checker, and it changes no tier and upgrades no grade. And a note lives where the reader looks — one named object per layer, not a per-field suffix scattered through the block — so its presence is visible without diffing against the schema.

The boundary matters as much as the mechanism. A note annotates a field the schema declares; it is not a place to record something the schema has no field for. The same campaign's engine log reads "Using Flash Attention backend on V1 engine", and no layer of this protocol declares which attention backend served a run — a throughput-relevant fact with nowhere to go. Writing it into `notes` under an invented key would put it where no reader would query it and no checker would ever see it, which is how a declaration block turns into a comment field. That gap is a missing field, and it is recorded here as one.

One note is mandatory. For any report whose `input_modalities` contains image or video, C1 requires the hardware layer's `cpu_cores` to carry a note stating whether the number is the machine's core count or an allocation. The measurement directly above is the reason: under a media workload the host CPU decodes and patchifies every image, so `cpu_cores` is a capacity input rather than an inventory fact, and a run that does not say which it published has published a ceiling nobody can attribute.
