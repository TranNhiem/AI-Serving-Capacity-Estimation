# Chapter 2 — Model Declaration

The model layer answers: *what weights, what precision, what attention geometry, and how a non-text input becomes sequence tokens.* These facts feed the **weights floor**, the **KV floor**, and every roofline in `ascep.capacity`. A wrong declaration here is not a small error — it propagates multiplicatively into every capacity figure downstream. This chapter is normative: every ASCEP field listed here is required by `model.schema.json`, and an unmeasured or unknown value MUST be recorded as `null` with a `(U)` entry, per **C1**. Checkpoint and preprocessor configuration keys are quoted throughout as evidence for a declaration, never as fields of our own; where a chapter names one, it is telling a reader which file to open.

The fourth question is the newest and the least self-evident, because two models with identical weights and identical attention geometry can price an image an order of magnitude apart, and the difference is not in any field this chapter lists. Media token geometry, its declarations and its mandatory calibration are [chapter 9](09-multimodal-and-reasoning.md)'s; what belongs here is the warning that a media-bearing model is not fully declared by the fields below. In particular, a per-image cost read out of a checkpoint config and never exercised has been measured wrong by 5.6% on one model and by a factor of four on the same model launched on a different tier, so a checkpoint number is a hypothesis until a server has charged for it.

## Total parameters vs active parameters

A report MUST declare both of these, separately:

| field | definition | used by |
|---|---|---|
| `total_params` | every parameter stored in the checkpoint | `weight_bytes`, `kv_pool_bytes`, `fits` |
| `active_params` | parameters executed per token (for MoE: shared trunk + `top_k` experts) | `roofline_decode_tok_s`, `roofline_prefill_ttft_s` |

**The rule (normative):** memory floors MUST be computed from `total_params`; compute rooflines MUST be computed from `active_params`. Mixing them up is the single most common MoE sizing error.

**The failure it prevents.** For a Mixture-of-Experts model every expert must be resident in VRAM even though only `top_k` run per token. *Illustrative arithmetic:* a model with 128 total experts, 8 active per token, and a shared trunk such that total params are 400B and active params are ~50B. Sizing VRAM from the active count gives `weight_bytes(50e9, "bf16")` ≈ 100 GB and suggests a pair of 141 GB GPUs suffices. The truth is `weight_bytes(400e9, "bf16")` ≈ 800 GB — an **~8× under-provision** that is invisible until the checkpoint fails to load, and that lands *after* procurement.

The symmetric error — using `total_params` in `roofline_decode_tok_s` **at batch 1** — understates the decode roofline by the same factor. That matters because the **roofline efficiency** (measured ÷ theoretical, mandatory under C6's reporting rules) then reads far above 1.0, which the spec defines as evidence of measurement error. At batch 1 a step routes one token to `top_k` experts and reads nothing else, so active params is exact and total params is wrong.

**Above batch 1 the rule inverts, and this is the correction.** A decode step processes `batch_size` tokens, each routed independently, and must read the *union* of every expert they selected — not one token's share of it. `moe_decode_weight_params` computes that union as `moe_experts × (1 − (1 − moe_top_k / moe_experts) ^ batch_size)` experts over the shared trunk, and it saturates fast: at 128 experts and top-8, batch 32 already reads ~87% of the checkpoint and batch 192 reads essentially all of it. So a production-batch MoE streams close to its **total** parameters per step. Pricing those steps at `active_params` treats batching as free for the weight term — true for a dense model, false for this one — and on the 26B/3.8B MoE measured in `examples/gb200-moe-26b-tp1` at batch 476 it understates weight traffic **6.75×**. The error lands inverted in the efficiency ratio, which sits in the denominator: a server running at a real 0.158 of its bound was published at 0.043, and C6's "investigate anything near 1.0" cannot fire against a number held that far below it.

**Normative.** A report whose MoE decode roofline was measured at a batch size above 1 MUST pass `total_params`, `moe_experts` and `moe_top_k` to `roofline_decode_tok_s` alongside `active_params`. Passing a partial geometry is an error, not a fallback: the function raises rather than silently reverting to the flat-active read, because an understated roofline is indistinguishable from a correct one once it is a number in a table. Dense models pass none of the three and are unaffected — `weight_bytes` still uses `total_params`, and `roofline_prefill_ttft_s` still uses `active_params` at every batch, since prefill is FLOP-bound and the FLOPs really are ~2 per active parameter per token however many experts the union touches.

## Dense vs MoE

`model.schema.json` MUST record `architecture: dense | moe`, and for MoE additionally:

- `moe_experts`, `moe_top_k`
- shared-expert parameters counted *inside* `active_params`, not alongside them: a shared expert runs on every token, so it is active compute by definition
- expert placement assumptions, if the serving layer will shard experts independently of TP

**Failure prevented:** an MoE model declared as dense has no `active_params` field at all, and the reporter silently substitutes total params — reproducing the 8× roofline distortion above.

## Precision and quantization

The report MUST declare the *stored* weight precision from the vocabulary of `DTYPE_BYTES` — including sub-byte formats `fp6`, `nvfp4`, `mxfp4`, `int4`, `fp4` — via `dtype_bytes()`. It MUST also declare:

- `overhead_frac`: quantization scales, zero-points, group sizes, and format padding, expressed as a fraction of stored weight bytes. `weight_bytes()` multiplies by `(1 + overhead_frac)`. For bf16/fp16 it is 0; for group-quantized 4-bit formats it is typically 0.03–0.15 depending on group size. If the overhead is not measured, record the weight bytes as `(U)` rather than assuming zero.
- `kv_precision`: the precision of the KV cache, which MAY differ from the weight precision and MUST NOT be silently copied from it.

**Failure prevented:** a 4-bit model declared as exactly 0.5 bytes/param with `overhead_frac` omitted under-reports weight residency by the scale/zero-point overhead; at 141 GB-per-GPU scale that gap is several GiB per replica — enough to flip `fits()` from true to false, or to leave the run with a near-zero KV pool that serializes to batch-size-1 (see the **weights floor**). Declaring KV precision equal to weight precision when the engine actually keeps KV in bf16 inflates the **KV floor's** `sessions` by 2× (or 4× for fp4 weights vs fp16 KV) on paper that reality will not deliver.

## KV geometry: layers, KV heads, head dim

> These three fields drive `kv_bytes_per_token()` and apply to **MHA, GQA and MQA models only**.
> MLA and recurrent/linear-attention models need different fields entirely — see
> *Attention families* below. Declare `attention_type` first; it selects which geometry is
> even meaningful, and the schema will reject a mismatched combination.

For attention-based models these three fields drive everything about the KV floor:

| field | where it lives in a typical HF `config.json` |
|---|---|
| `n_layers` | `num_hidden_layers` |
| `n_kv_heads` | `num_key_value_heads` (falls back to `num_attention_heads` for MHA) |
| `head_dim` | `head_dim`, or `hidden_size / num_attention_heads` — verify, do not assume |
| `kv_precision` | engine config, not model config |

Reports SHOULD state the source of each field (config path and value). **Failure prevented:** deriving `head_dim` by division when the model uses a non-standard projection (some architectures decouple it) silently mis-computes `kv_bytes_per_token` and with it the number of concurrent sessions — the error is invisible because the model still loads.

## GQA/MQA and the TP replication trap

Grouped-query and multi-query attention reduce KV by sharing KV heads across query heads. The trap is that KV heads cannot be split wider than their count: when `tensor_parallel > n_kv_heads`, runtimes **replicate** heads per rank. `kv_heads_per_rank(n_kv_heads, tensor_parallel) = max(1, ceil(n_kv_heads / tensor_parallel))`, and `kv_bytes_per_token()` uses `kv_heads_per_rank × tensor_parallel` effective heads cluster-wide.

Worked table — illustrative, for a model with `n_kv_heads = 2`:

| TP | `kv_heads_per_rank(2, TP)` | effective KV heads (× cluster) | cluster KV bytes/token vs TP=1 |
|---|---|---|---|
| 1 | 2 | 2 | 1× |
| 2 | 1 | 2 | 1× |
| 4 | 1 (replicated) | 4 | 2× |
| 8 | 1 (replicated) | 8 | 4× |

Per-GPU KV footprint *falls* up to TP=2, then the cluster-wide KV pool per token *grows*: at TP=8 the same GPUs hold one quarter of the context tokens they would if heads could split. This is why **C3 (topology binding)** exists — a per-GPU KV figure measured at TP=2 and extrapolated to TP=8 overstates KV capacity by 4×. **Normative:** any KV or capacity figure MUST be bound to the TP width it was derived at, and `gpus_required()` inputs MUST come from a measurement at the same `gpus_per_replica` that will be deployed.

## Sliding-window and hybrid attention: `global_layer_frac`

Models with sliding-window or hybrid attention keep full-length KV only on their *global* layers; local layers cap at the window. The model declaration MUST record:

- `attention_type: sliding-window | hybrid` (from the enum below)
- `sliding_window_tokens`, if applicable
- `global_layer_frac = n_global_layers / n_layers`, in `(0, 1]`

`kv_bytes_per_token()` multiplies by this fraction. **Failure prevented:** leaving `global_layer_frac` at the 1.0 default for a hybrid model over-reports KV bytes per token by several times — pessimistic, so harmless to buyers, but it corrupts cross-model comparison. The dangerous direction is assuming the *small* footprint of a sliding-window model and then serving contexts longer than the window without declaring it. If `calibrate_memory_utilization()` returns a value above 1.0 against your analytic model, a wrong `global_layer_frac` is the first suspect named by the protocol.

**`global_layer_frac` is the asymptote, not the cost — and this is the second required declaration.** A local layer is capped at `sliding_window_tokens`, so it holds `min(context, window)` tokens: the *whole* context until the context outgrows the window, and only then anything less. The share of full-length-equivalent KV a hybrid stack actually holds is therefore context-dependent, and `effective_layer_frac()` computes it:

```
frac = global_layer_frac + (1 − global_layer_frac) × min(avg_context_tokens, sliding_window_tokens) / avg_context_tokens
```

It is 1.0 while the context fits the window, decays as the context grows past it, and reaches `global_layer_frac` only in the limit. Declaring the bare fraction at every context under-reports KV by up to `1 / global_layer_frac`, and that is the over-promising direction: it inflates KV capacity and so inflates concurrent sessions. On the hybrid MoE in `examples/gb200-moe-26b-tp1` — 30 layers, 5 global, a 1,024-token window, an average context of 903 tokens — the bare 1/6 gives 20,480 bytes per token while all 25 local layers are holding the full context and the truth is 122,880: a **6× understatement**, at the shortest contexts, where the model looks most comfortably deployable.

It is detectable before deployment, which is why the protocol prefers an engine-reported KV size. Against that model's engine-reported 492,000 tokens the bare fraction predicts 6,210,713 — 12.6× — while the context-aware fraction predicts 1,035,119, or 2.1×, which is ordinary workspace-and-fragmentation territory. **Normative:** a report for a `sliding-window` or `hybrid` model MUST pass `sliding_window_tokens` and `avg_context_tokens` to `kv_bytes_per_token()` alongside `global_layer_frac`, and MUST state the context the KV figure was computed at, since for these models it is not one number. Passing one of the two is an error, not a fallback. Uniform-attention models pass neither and are unaffected.

## Attention families: the declaration that selects the KV formula

`attention_type` MUST be one of the following. It is not descriptive metadata — it decides
which function computes the KV floor, and a wrong value is an order-of-magnitude error, not a
rounding one.

| `attention_type` | additionally required | KV scales with | function |
|---|---|---|---|
| `full`, `gqa`, `mqa` | `n_kv_heads`, `head_dim` | tokens × heads | `kv_bytes_per_token()` |
| `sliding-window`, `hybrid` | the above + `global_layer_frac`, `sliding_window_tokens`, `avg_context_tokens` | tokens × heads × `effective_layer_frac()` | `kv_bytes_per_token()` |
| `mla` | `kv_lora_rank`, `qk_rope_head_dim` | tokens only | `kv_bytes_per_token_mla()` |
| `linear`, `ssm`, `hybrid-recurrent` | `kv_bytes_per_sequence` | sequences only | `kv_capacity_sessions()` |

**MLA** (multi-head latent attention, DeepSeek-style) caches one compressed latent per layer
plus a decoupled RoPE key, rather than K and V per head. Take `kv_lora_rank` and
`qk_rope_head_dim` verbatim from `config.json` — do not derive them. Two consequences invert
the GQA intuition above: there is no head count in the formula, and **the latent is replicated
across tensor-parallel ranks rather than sharded**, so cluster KV per token is unchanged by TP
width. The TP replication trap does not apply; a different one does, in the opposite
direction — widening TP does not buy proportionally more sessions.

**Failure prevented:** applying the GQA formula to an MLA model overstates KV per token by
~57× for DeepSeek-V3 geometry. On eight GPUs at a 32,000-token context that is a prediction of 3
concurrent sessions where the truth is 153 — enough to reject a deployable model as
undeployable.

**Linear attention, SSM/Mamba and recurrent hybrids** hold a fixed-size state per sequence,
independent of context length, so session capacity is flat in context and these models are
throughput-bound rather than KV-bound at every length. Declare `kv_bytes_per_sequence`.
Hybrid-recurrent stacks declare both it and the per-token geometry; the costs add.

**Failure prevented:** modelling a constant-state architecture per-token predicts a
long-context capacity cliff that does not physically exist, and leads to buying GPUs for a
constraint the model does not have.

## Context length: native maximum vs served length

Declare both `native_max_context_tokens` (the architecture's trained or RoPE-extended ceiling) and, in the measurement layer, the context lengths actually swept. Per **C4**, a throughput curve SHOULD cover at least three context lengths, and `interpolate_throughput()` clamps rather than extrapolates beyond them — points past the measured range MUST be tagged `(U)`. **Failure prevented:** a throughput figure measured at 1k context applied to a 32k workload overstates capacity 2–4×; claiming 128k support because the config says so, without measurement, is a context binding violation.

## Revision pinning

The model declaration MUST pin the artifact: repository ID **and** exact revision (commit hash or content digest), tokenizer revision, and checkpoint format/shard layout. **Failure prevented:** config values silently changing between runs — `num_key_value_heads`, rope scaling, quantization metadata — produce two runs that disagree with no code change, and `calibrate_memory_utilization()` will report an impossible utilization with no discoverable cause. An unpinned model identity makes the reproduction bundle required by **C8** non-functional.
