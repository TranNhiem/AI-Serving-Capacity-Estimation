# Chapter 3 — Serving Configuration

Layer 3 is where identical silicon and identical weights produce capacity reports that differ
by integer multiples. Two reports that agree on layers 1, 2 and 5 and disagree here are
measuring different systems, and comparing them is an error ASCEP exists to prevent. Every
field below MUST be declared in `serving.schema.json`; unknown values MUST be recorded as
`null` with a `(U)` entry per C1, never omitted.

## R1 — Framework identity

The report MUST declare the serving framework, its **exact version string**, and the
**container digest** (not the mutable tag) of the image that produced every measurement.
Bind to layer 4 as required by C8.

**Failure prevented.** Capacity-relevant behaviour — scheduler, chunked prefill defaults,
KV layout — changes between point releases. Worse, a newer image can refuse a model
outright (unsupported attention variant, dropped quantization path, altered weight format),
so the run that "worked last quarter" loads nothing today: no capacity tier survives, and a
report citing only `vllm:latest` cannot even name the software that produced its numbers.
A tag is not a version; a digest is.

## R2 — Parallelism topology

The report MUST declare tensor-parallel width (`tp`), pipeline depth (`pp`), and total GPU
count, and MUST bind every capacity, KV and throughput figure to that triple (C3).

- **Tensor parallelism** shards a layer within one replica. It belongs inside a node
  connected by a high-bandwidth fabric, because every decode step all-reduces. Width beyond
  the node's fast interconnect pays latency, and — as `kv_heads_per_rank` makes concrete —
  width beyond the model's KV head count *replicates* KV heads, shrinking total KV per
  replica instead of growing it.
- **Pipeline parallelism** splits layers across replicas or nodes. It tolerates slower links
  but adds stage bubbles and does not reduce per-replica KV for long sessions evenly.

Per-GPU figures MUST NOT be presented as topology-independent. **Failure prevented.** A
"tokens of KV per GPU" measured at TP=2 on a 2-KV-head grouped-query model, reused at TP=4,
overstates the KV pool by ~2×; `gpus_required` then under-provisions a whole replica. This
is the C3 failure, and it is silent.

## R3 — Batching policy

Declare whether the engine uses **continuous (in-flight) batching** or **static batching**.

- Continuous batching admits sequences as they finish and is the regime every ASCEP
  capacity tier assumes.
- Static batching runs fixed batches to completion; its stragglers idle the GPU, so its
  measured tiers are not comparable and MUST be labelled with the batch size used.

Mixing regimes across reports being compared is a conformance failure. **Failure
prevented.** A static-batched run at batch 64 compared against a continuous-batched run
shows a phantom "2× regression" that is purely the scheduler.

## R4 — Sequence and token budgets

Declare, as integers, `max_model_len` and the scheduler's per-iteration token budget
(e.g. `max_num_batched_tokens`), plus `max_num_seqs` or its equivalent.

**Failure prevented.** `max_model_len` pins the KV slot reserved per sequence in engines
that pre-allocate by maximum length: doubling it can halve concurrent sessions without any
hardware change — a KV-floor change (C5) invisible unless declared. `max_num_batched_tokens`
sets the prefill/decode trade per iteration; a small value starves prefill and inflates
TTFT at load, causing SUSTAINABLE-tier failures that look like, but are not, throughput-floor
problems. Comparing two reports with different unrecorded budgets attributes a scheduler
choice to the hardware.

## R5 — The memory-utilization knob

Declare the framework's static memory cap: the fraction of per-GPU VRAM the engine reserves
for weights plus KV. This is the single most frequently omitted number in published
serving benchmarks.

| framework | field | meaning |
|---|---|---|
| vLLM | `gpu_memory_utilization` | fraction of VRAM for weights + KV |
| SGLang | `mem_fraction_static` | fraction for weights + static KV pool |
| TGI | implicit / per-model default | reserved pool; record effective value |
| TensorRT-LLM | `kv_cache_free_gpu_mem_fraction` | fraction left free for KV |

`kv_pool_bytes` takes this knob as `memory_utilization`; the KV floor moves linearly with
it, so an unrecorded 0.90-vs-0.95 difference is a ~±5–100% capacity difference depending on
how close weights sit to the cap.

**Failure prevented.** Without the knob, a KV measurement cannot be reproduced: a second
team re-running "the same config" at a different default gets a different KV pool and
concludes the original report was wrong. And analytic projections made with an assumed
0.90 disagree with reality by multiples once workspace, fragmentation and CUDA graph
capture are counted. The protocol therefore REQUIRES that when the engine reports its KV
cache size, that measured figure be used in preference to the analytic model, and that
`calibrate_memory_utilization` be run to back-solve the effective utilization before
projecting KV capacity to other context lengths. A calibrated value above 1.0 means the
input KV geometry is wrong — usually `global_layer_frac` or KV precision — and MUST be
investigated, not averaged away.

## R6 — Capacity-altering features

Each of the following changes at least one floor or tier. Each MUST be declared as
enabled/disabled with its configuration; an undeclared feature makes the report partial at
best.

| feature | knob (example) | what it changes | failure if undeclared |
|---|---|---|---|
| Prefix caching | `enable_prefix_caching` | effective prefill cost; shared-system-prompt throughput | cache hits masquerade as compute; roofline efficiency passes 1.0, which per spec indicates measurement error |
| Chunked prefill | `chunked_prefill_size` / `long_prefill_token_threshold` | TTFT tail vs decode interference | an SLO-gate failure is blamed on the model, not the chunk size |
| KV quantization | `kv_cache_dtype=fp8` | `kv_bytes_per_token` halves | analytic kv floor off by 2× against the measured one |
| KV offload to host | `swap_space` / CPU offload | KV pool above VRAM, ITL tail on miss | sustained-tier collapse at the exact concurrency the headline number claims |
| Speculative decoding | draft model + acceptance config | decode tok/s per active step | measured ÷ theoretical approaches or exceeds 1.0 — per spec a measurement error unless the draft is declared and active-param accounting is corrected |
| Scheduler / concurrency limits | `max_num_seqs`, waiting-queue policy | which requests queue, and the tail | open-loop saturation reported as user capacity (the spec's canonical sin) |
| Media preprocessing | `media_preprocessing`, `limit_mm_per_prompt`, `mm_processor_cache_gb` | tokens charged per image, media admitted per request, images that skip preprocessing | see below |

The rule is not that these features are forbidden — several are the correct production
choice — it is that each is a different system. C2 requires tagging every number that
depends on one.

The last row is the one whose failure will not fit in a table cell, and it is the newest.
For a media-bearing workload the serving layer holds cost configuration that no other layer
records: the per-image soft-token tier the server was launched on, which on one measured
processor is selectable across a range whose ends differ by a factor of sixteen; a pixel
budget that silently downscales rather than rejects, so the benchmark measures a mix the
report does not name; a per-prompt media limit whose default of one image turns a
multi-image request into a millisecond-long rejection; and a processor cache that is on by
default and lets a repeating corpus skip preprocessing entirely. Each of these was measured
moving capacity by up to an integer multiple, none is derivable from the model layer, and
each MUST be declared. The mechanisms, the measurements and the run-validity consequences
are [chapter 9](09-multimodal-and-reasoning.md)'s, and are not repeated here.

## R7 — Cross-framework field map

Reports SHOULD use ASCEP field names; the table maps the common engines.

| ASCEP field | vLLM | SGLang | TGI | TensorRT-LLM |
|---|---|---|---|---|
| `tensor_parallel` | `--tensor-parallel-size` | `--tp` | sharding config | `tp_size` |
| `pipeline_parallel` | `--pipeline-parallel-size` | `--pp` | — | `pp_size` |
| `max_model_len` | `--max-model-len` | `--context-length` | `--max-input-length`+`--max-total-tokens` | `max_input_len`/`max_seq_len` |
| `token_budget` | `--max-num-batched-tokens` | `--max-prefill-tokens` | `--max-batch-prefill-tokens` | `max_num_tokens` |
| `max_sequences` | `--max-num-seqs` | `--max-running-requests` | `--max-batch-total-tokens` (tokens) | `max_batch_size` |
| `memory_utilization` | `--gpu-memory-utilization` | `--mem-fraction-static` | implicit | `free_gpu_memory_fraction` |
| `kv_precision` | `--kv-cache-dtype` | `--kv-cache-dtype` | — | `quantization.kv_cache` |

The map is illustrative, not authoritative: field names move between releases, which is
exactly why R1 pins the digest. Where an engine lacks a knob, record the effective
observed behaviour as `(M)` or mark the field `(U)`. Never guess a default.

## Conformance notes

A `serving` declaration missing R1, R3 or R5 fails C1 outright. Figures detached from the
R2 topology fail C3. Feature-driven tier differences (R6) without declarations fail C2, and
for a media-bearing workload that includes R6's last row: a figure whose real constraint is
media in flight rather than tokens in flight describes a different system the moment the
soft-token tier moves. Remember C5: the binding constraint of every capacity figure in later
chapters will have been set by the knobs declared here — this chapter is where `weights`,
`kv`, `throughput` and `slo` outcomes are manufactured. One caveat, measured and recorded in
§9.2: on an engine with a multimodal encoder cache there is a fifth bound that none of those
four names, so for a media-dense workload C5's answer is the binding floor *among the ones
the model prices*, and the report has to say so.
