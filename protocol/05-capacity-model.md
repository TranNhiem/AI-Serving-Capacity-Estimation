# Chapter 5 — The Capacity Model

This chapter is the normative definition of the analytic half of ASCEP. Every formula lives in `ascep/capacity.py`, is pure, and is individually testable. A report that cites an **(I)** value MUST name the function that produced it (conformance rule C2). Wherever a measured value and an analytic value both exist, the measured value MUST be used; the formulas below exist to project, to cross-check, and to fill declared gaps — never to override measurement.

## 5.1 The memory model

### Weights: `weight_bytes`

```
weight_bytes = total_params × dtype_bytes(precision) × (1 + overhead_frac)
```

The argument is **total** parameters, not active. For a Mixture-of-Experts model every expert must be resident even though only `top_k` run per token; using active parameters here is the single most common MoE sizing error and produces clusters that cannot load the model they were bought for. Active parameters belong only in the compute rooflines (§5.2). `overhead_frac` covers quantization scales, zero-points and padding; it is 0 for bf16/fp16. Skipping it for 4-bit formats understates resident weights by 5–15% and mis-attributes the discrepancy to "framework overhead" later.

### KV geometry: `kv_heads_per_rank`, `kv_bytes_per_token`

```
kv_heads_per_rank = max(1, ceil(n_kv_heads / tensor_parallel))
kv_bytes_per_token = 2 × n_layers × global_layer_frac
                     × kv_heads_per_rank × tensor_parallel
                     × head_dim × dtype_bytes(kv_precision)
```

The factor 2 counts K and V. The topology term is the part most reports get wrong: when `tensor_parallel > n_kv_heads`, the heads cannot split further and runtimes **replicate** them, so total KV memory *grows* with TP width. A grouped-query model with 2 KV heads at TP=4 materializes 4 heads' worth of KV, twice the TP=2 figure — the wider deployment holds *fewer* tokens. This is why conformance rule C3 forbids presenting per-GPU KV as topology-independent: extrapolating a TP=1 or TP=2 measurement to TP=8 can be wrong by an integer multiple, in the pessimistic direction.

`global_layer_frac` is the fraction of layers holding full-length KV. Models with sliding-window or hybrid attention keep full context only on their global layers. Leaving it at 1.0 for such a model understates KV capacity by several times — a wrong procurement in the expensive direction. Reports MUST declare the value used; `calibrate_memory_utilization` below will expose a wrong one.

### Attention families: choosing the right KV formula at all

The formula above applies to **MHA, GQA and MQA** only. Two other families are now common, and each breaks a different assumption in it. The declared `attention_type` (chapter 2) selects the formula, and getting it wrong is not a refinement — it is an order-of-magnitude error.

| family | state scales with | function | what breaks if you use the GQA formula |
|---|---|---|---|
| MHA / GQA / MQA | tokens × heads | `kv_bytes_per_token` | — |
| MLA (latent) | tokens only | `kv_bytes_per_token_mla` | overstates KV by ~57× |
| linear / SSM / recurrent | sequences only | `kv_capacity_sessions(kv_bytes_per_sequence=…)` | predicts a long-context cliff that does not exist |

**MLA.** Multi-head latent attention caches one compressed latent per layer plus a small decoupled RoPE key, not K and V per head:

```
kv_bytes_per_token_mla = n_layers × global_layer_frac
                         × (kv_lora_rank + qk_rope_head_dim)
                         × dtype_bytes(kv_precision)
```

There is no factor of 2 and no head count. Critically, **there is no `tensor_parallel` term either**: the latent is replicated on every rank rather than sharded, so cluster-wide KV per token does not fall as TP widens. Widening TP for an MLA model buys compute and aggregate HBM but **not** proportionally more sessions — the opposite of the intuition the GQA case builds. For DeepSeek-V3 geometry (61 layers, `kv_lora_rank` 512, `qk_rope_head_dim` 64) the correct figure is 70,272 bytes/token against 3,997,696 from a naive 128-head GQA calculation: on eight GPUs holding 40 GiB of KV each at a 32,000-token context, that is the difference between predicting **3** concurrent sessions and the real **153**. A team that sized with the wrong formula would conclude the model was undeployable.

**Linear attention, SSM/Mamba, and recurrent hybrids** keep a fixed-size state per sequence, independent of context length. Session capacity is therefore **flat in context**, and such a model is essentially never KV-bound — it is throughput-bound everywhere, so §5.3's crossover never occurs. Declare `kv_bytes_per_sequence` and use `kv_capacity_sessions`. Hybrid stacks that interleave attention and recurrent layers declare **both**; the costs add, with `global_layer_frac` restricting the per-token term to the attention layers.

Reports MUST state which family was assumed. The three give materially different answers for identical hardware, and the schema enforces that the geometry supplied matches the family declared.

### The pool: `kv_pool_bytes`, `kv_capacity_tokens`

```
kv_pool_bytes = max(0, n_gpus × vram_bytes_per_gpu × memory_utilization
                     − weights_bytes − activation_bytes)
kv_capacity_tokens = kv_pool_bytes / kv_bytes_per_token
```

`memory_utilization` is the framework's own cap (vLLM `gpu_memory_utilization`, SGLang `mem_fraction_static`, or the equivalent). It MUST be recorded in the serving-layer declaration. It is the most frequently omitted number in published benchmarks, and without it a KV measurement cannot be reproduced — two runs that differ only here can differ in KV capacity by the entire unreserved fraction of VRAM.

### Fit check: `fits`

`fits` is binary and answers the weights floor: does the model load **and** leave a usable KV pool? A configuration that loads with near-zero KV is not viable — it serializes to batch-size-1 and latency collapses. Callers SHOULD pass a non-zero `min_kv_tokens` reflecting at least one worst-case session, so "theoretically loads" cannot be reported as "deployable".

### Calibration: `calibrate_memory_utilization`

```
effective_utilization = (measured_kv_tokens × kv_bytes_per_token + weights_bytes)
                        / (n_gpus × vram_bytes_per_gpu)
```

**Normative rule.** When the serving engine reports its KV cache size, that figure MUST be used in preference to the analytic pool, and `calibrate_memory_utilization` SHOULD be used to reconcile the two before projecting to other context lengths. Rationale: the analytic model cannot see runtime workspace, allocator fragmentation, CUDA-graph capture, or framework reservations; an engine-reported token count already contains all of them. Projecting from the analytic pool alone typically overstates KV capacity. A calibrated value above 1.0 means the inputs disagree with the measurement — most often a wrong `global_layer_frac`, a wrong KV precision, or undisclosed KV offloading — and MUST be investigated, not published.

## 5.2 The compute rooflines

### Decode: `roofline_decode_tok_s`

Decode is bandwidth-bound, not FLOP-bound. Each autoregressive step performs ~2 FLOPs per active parameter but must *read* every active weight once: at bf16 that is 1 FLOP per byte streamed, far below any accelerator's FLOP-per-byte ridge point. The step time is therefore set by HBM bandwidth:

```
bytes_per_step  = active_params × dtype_bytes(precision)
                + batch_size × avg_context_tokens × kv_bytes_per_token
steps_per_s     = hbm_bandwidth_bytes_s / bytes_per_step
throughput      = steps_per_s * batch_size * efficiency
```

Two structural facts follow. First, **batching amortizes the weight read**: the weight term is paid once per step regardless of `batch_size`, so throughput rises with concurrency until the KV-read term dominates — which is exactly the throughput-vs-context-length crossover of §5.3. Second, the KV term grows with both batch and context, so this roofline is context-dependent; quoting a single theoretical decode rate without its `avg_context_tokens` violates C4 in spirit even for **(T)** numbers. For MoE, `active_params` is the shared trunk plus `top_k` experts, never total params.

### Prefill: `roofline_prefill_ttft_s`

```
ttft_lower_bound = 2 × active_params × prompt_tokens / (flops_per_s × mfu)
```

Prefill is FLOP-bound at roughly 2 FLOPs per parameter per prompt token; `mfu` of 0.3–0.5 is typical for a well-tuned server. This bound excludes queueing, tokenization, scheduling and network, so measured TTFT under load is always larger, often by an order of magnitude. Presenting the roofline as expected TTFT is a category error; it exists to bound claims, not to set expectations.

## 5.3 The capacity floors and the crossover

`capacity_at` computes supportable concurrent users as the **minimum** of independent floors — never the average, never the most convenient:

| floor | formula | binds when |
|---|---|---|
| weights | `fits(...)` — binary | the model doesn't load with a usable pool |
| kv | `users_kv = (kv_tokens / headroom) / avg_context_tokens / duty_cycle` | long context |
| prefill | `users_pre = (prefill_tok_s / headroom) / demand_prefill_tok_s per user` | prompts heavier than the run the throughput figure came from |
| throughput | `users_thr = (throughput_tok_s / headroom) / per_user_tok_s` | short context |
| slo | overrides the constraint label when gates fail | fits, but misses a latency gate |

The prefill floor is the newest of the four and is introduced in full in [chapter 10](10-workload-archetypes.md), because it exists to catch a failure of the *declaration* side rather than of the measurement side. It enters the minimum only when a measured `prefill_tok_s` is supplied; omit it and `capacity_at` computes exactly what it computed before the floor existed, which is what protects every report published against the earlier protocol. The short statement of why it is needed: the throughput floor is denominated in generated tokens only, so a capacity number carried from a benchmark rung to a workload whose prompts are heavier than that rung's overstates capacity by exactly the ratio between the two token mixes. §10.3 derives that identity.

**Which floor binds changes with context length.** At short context, KV per session is small, the pool holds many sessions, and compute is the scarce resource — the throughput floor binds. As context grows, `avg_context_tokens` scales the per-session KV footprint while aggregate throughput simultaneously falls (both the roofline KV term and attention compute grow), so capacity crosses over to the KV floor. Reports MUST state the crossover context length, or state that it was not determined. A report covering only short prompts and projecting to a document workload will silently apply the wrong floor and overstate capacity by multiples.

Two details in `capacity_at` prevent known wrong numbers. Per-user demand is derived as `demand_tok_s() / peak_concurrent_users()` rather than from the raw per-stream target, so the workload's duty cycle stays applied; using the raw target overstates demand by `1/duty_cycle` and over-provisions the cluster. And as C5 requires, the returned `Capacity` always carries `binding_constraint` — a number without its constraint does not say whether to buy more HBM, more bandwidth, or fewer SLO promises.

`gpus_required` inverts the problem: it sweeps whole replicas upward until `capacity_at` meets the workload. Capacity is bought in whole replicas (`gpus_per_replica` is the TP width), so a need of 3 GPUs at TP=2 provisions 4. Both per-GPU inputs MUST come from measurements **at the same TP width** — per C3 and §5.1, per-GPU KV is not a constant across topologies. `max_gpus` existing as a failure mode is deliberate: an unsatisfiable workload MUST produce an error, not an extrapolated number.

## 5.4 Interpolation: `interpolate_throughput`

Throughput between measured context points is piecewise-linear **(I)**; past the longest measured point it clamps rather than extrapolates. Linear extrapolation of a convex-decreasing curve can go negative, and any positive continuation is invented. Callers projecting beyond measured range MUST tag the result **(U)**. A single measured point reported without an interpolation label violates C4.

## 5.5 The four tiers and roofline efficiency

| tier | computed as | use |
|---|---|---|
| theoretical | `roofline_*` | upper bound only |
| measured | best observed, SLO ignored | engine ceiling |
| sustainable | measured restricted to operating points where every SLO gate held for the full window | what users can rely on |
| recommended | `capacity_at(headroom=…)` on sustainable inputs (typical headroom 1.15+) | procurement |

All four MUST be reported (C6); they differ by integer multiples and readers assume the most favourable.

**"Operating points" is load-bearing in that table.** A gate failure disqualifies the sustainable tier only where the failure lies *inside the envelope the capacity claim covers*. A run that fails its TTFT gate at 8,192 tokens does not invalidate a sustainable figure computed at a 2,000-token context; sustainable may legitimately equal measured there. What the report MUST NOT do is quote that figure without its envelope — the claim is "460 users **at ≤2,000 tokens**", and the 8k failure is precisely the evidence that the qualifier is not decorative. *Failure prevented, in both directions:* a blanket "any failing point invalidates the tier" rule rejects honest reports and teaches people to ignore the checker, while dropping the envelope lets a short-context number be read as a general one. The ratio measured ÷ theoretical is the **roofline efficiency**, a required reporting field. Real servers land well below 1.0. A value at or above 1.0 is not a fast server — it indicates a mis-declared active-parameter count, untracked cache hits, or a unit error, and MUST be investigated before publication. The symmetric rule applies to `scaling_efficiency`: values above 1.0 mean the *baseline* was degraded, usually KV-starved, and MUST be flagged, never reported as a win.
