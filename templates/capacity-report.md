# ASCEP Capacity Report — `<model>` on `<hardware>`

> ASCEP v0.1 · report generated `<UTC timestamp>` · protocol conformance: **`<conforming | partial | non-conforming>`**

**Conformance note.** `<Which of C1–C8 this report meets, and which it does not. A level
claimed without a reason cannot be reviewed, so this paragraph is required at every level —
including` conforming `. If any rule is unmet, say which, why, and what it would cost to
close.>`

Every numeric claim below carries a provenance tag. Reports that omit tags are
**non-conforming** and must not be compared against conforming reports.

| tag | meaning |
|---|---|
| **(M)** | measured in this campaign, raw record available under `runs/` |
| **(I)** | computed from an (M) value by a formula in `ascep.capacity`, formula named |
| **(T)** | theoretical roofline — an upper bound, not an expectation |
| **(U)** | unmeasured assumption; the report is only as good as this number |

---

## 1. Hardware

| field | value |
|---|---|
| GPU model / count | |
| VRAM per GPU | |
| Interconnect (intra-node) | `NVLink <gen> / PCIe <gen> / other` |
| Interconnect (inter-node) | `InfiniBand <rate> / RoCE / Ethernet / n-a` |
| Nodes × GPUs per node | |
| CPU model / cores | |
| System RAM | |
| Storage class + model load path | |
| HBM bandwidth per GPU (spec) | |
| Dense BF16 TFLOP/s per GPU (spec) | |
| Driver / CUDA / ROCm version | |
| Node exclusivity during benchmark | `exclusive / shared` |

> Shared nodes invalidate cross-report comparison. State it.

## 2. Model

| field | value |
|---|---|
| Model ID + revision/commit | |
| Total parameters | |
| Active parameters per token | dense: same as total; MoE: trunk + `top_k` experts |
| Architecture | `dense / MoE (E experts, top_k)` |
| Weight precision | |
| KV precision | |
| Layers / KV heads / head dim | |
| Attention type | `full / GQA / MQA / sliding-window / hybrid` |
| Global-layer fraction | fraction holding full-length KV |
| Native max context | |
| Weight bytes on disk | (M) |
| Licence | |

## 3. Serving configuration

| field | value |
|---|---|
| Framework + exact version | |
| Container image digest | |
| Tensor parallel / pipeline parallel | |
| `max_model_len` | |
| `gpu_memory_utilization` (or equivalent) | **required** — see §7 |
| Batching mode | `continuous / static` |
| Max num seqs / max num batched tokens | |
| Prefix caching | `on / off` |
| KV cache offload / quantized KV | |
| Chunked prefill | |
| Speculative decoding | |
| Engine-reported KV cache size | (M) tokens — **the number to trust** |
| Cold-start time to ready | (M) |

## 4. Benchmark results

Per §7 of the protocol: warm-up discarded, `<N>` repeats, `<D>`-second sustained window,
concurrency ladder `<c values>`, outliers handled by `<method>`.

| shape (in/out) | concurrency | TTFT p50 / p95 / p99 (s) | ITL p50 / p95 (s) | e2e p95 / p99 (s) | output tok/s | req/s | GPU util % | GPU mem util % | SLO |
|---|---:|---|---|---|---:|---:|---:|---:|:--:|
| | | | | | | | | | |

**SLO gates applied:** TTFT p95 ≤ `<x>` s · ITL p95 ≤ `<y>` s · e2e p95 ≤ `<z>` s ·
error rate ≤ `<e>` % · all must hold for the full window.

**Scaling efficiency**

| topology | GPUs | tok/s | efficiency vs baseline |
|---|---:|---:|---:|
| | | | |

> Any efficiency > 1.0 must be explained, not celebrated — it means the baseline was degraded.

**Roofline comparison**

| metric | theoretical (T) | measured (M) | efficiency |
|---|---:|---:|---:|
| decode tok/s | | | |
| prefill TTFT | | | |

## 5. Capacity

| tier | max concurrent users | max tok/s | max req/s | daily requests | binding constraint |
|---|---:|---:|---:|---:|---|
| Theoretical (T) | | | | | |
| Measured (M) | | | | | |
| Sustainable (M, SLO-gated) | | | | | |
| **Recommended (I, headroom `<h>`)** | | | | | |

> A capacity figure without its binding constraint is not actionable — it does not say what
> to buy more of. Name it in every row.

## 6. Application workload → required infrastructure

| field | value |
|---|---|
| Application type | |
| Daily active users / concurrent users | |
| Sessions per user per day · avg session length | |
| Peak-to-mean ratio | |
| Duty cycle | fraction of a session actively generating |
| Input / output tokens per request | |
| Avg KV context per active session | (I) |
| Required per-stream token rate | |
| Aggregate demand at peak | (I) tok/s |

**Result**

| | value |
|---|---|
| GPUs required | |
| Replica topology | `<n> × TP<w>` |
| Binding constraint | |
| Utilization at target load | |
| Headroom remaining | |

## 7. Unmeasured assumptions and known limits

List every **(U)** in the report and what it would take to close it. A report whose
conclusion flips on an unmeasured number must say so here, in the same words a reviewer
would use against it.

| assumption | value used | impact if wrong | cost to measure |
|---|---|---|---|
| | | | |

## 8. Reproduction

C8 asks for a bundle a stranger can re-run, not a description of one. ASCEP v0.1 ships no
benchmark driver — chapter 7 specifies the procedure and you run it with your own harness — so
name that harness and its invocation here, exactly as it was run:

```
git clone <your harness repo> && cd <repo>
<the command you actually ran, with the config that produced these numbers>
```

Then validate and grade the artifact this template becomes:

```
ascep validate report.json
ascep conformance report.json
```

| artefact | path |
|---|---|
| Run configs | |
| Raw per-request records | |
| Engine logs | |
| Environment capture | |
| Analysis notebook / script | |
