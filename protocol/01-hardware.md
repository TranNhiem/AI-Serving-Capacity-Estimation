# Chapter 1 — Hardware Declaration

Layer 1 answers one question: *what silicon, how connected*. Every capacity number in the
report is a function of this layer, so this chapter defines the conformance floor for
declaring it. All fields map to `hardware.schema.json`; unknown values MUST be recorded as
`null` with a `(U)` entry (C1), never omitted and never guessed.

## 1.1 Required fields

| field | unit | why it is required | failure it prevents |
|---|---|---|---|
| `accelerator.model` | string | "Radeon", "TPU" or "80GB card" names families, not parts | A reader reproduces on a different SKU with 2× different bandwidth and gets a different answer from the same config |
| `accelerator.count` | int | CPU-visible device count per node and in total | C3 topology binding: without a count, per-GPU KV and throughput figures are unfalsifiable |
| `accelerator.vram_bytes_per_gpu` | bytes | Feeds `kv_pool_bytes` | Analytic and measured KV capacities silently disagree by the difference between SKUs (e.g. 40 vs 80 GiB variants of the same accelerator) |
| `accelerator.hbm_bandwidth_bytes_s` | bytes/s | Sole input to `roofline_decode_tok_s` | Without it the theoretical tier cannot be computed, and C6 (four tiers) cannot be met |
| `accelerator.flops_per_s_dense` | FLOP/s | Sole input to `roofline_prefill_ttft_s` | Same: no roofline, no roofline efficiency, no conformance. Declared as dense FLOP/s at the deployed precision, not sparse — sparse marketing numbers inflate the roofline ~2× and make measured efficiency look broken |
| `accelerator.precision` | string | The precision the bandwidth/FLOP/s figures above are quoted at | Quoting FP8 FLOP/s for a BF16 deployment doubles the theoretical prefill bound and poisons the efficiency check |
| `interconnect.intra_node` | enum + bandwidth | Determines the TP widths that are viable (§1.3) | TP=8 over PCIe is measured, published, and off by multiples versus what the same GPUs achieve over a high-speed fabric |
| `network.inter_node` | enum + bandwidth + topology | Determines whether a model is one node or many, and which parallel dim crosses nodes (§1.4) | A cross-node TP result presented as a single-node option |
| `cpu.model`, `cpu.cores`, `ram_bytes`, `storage.*` | varies | Cold-start and host-side bottlenecks live here (§1.5) | Load time omitted; a capacity report that is true only for an already-warm server |
| `topology.node_exclusive` | bool | Gate on every other number (§1.6) | A shared node invalidates all of it silently |
| `topology.nodes`, `topology.single_node` | int, bool | Multi-node is a different capacity regime (§1.7) | A 2-node result quoted per-GPU as if nodes were fungible |

A conforming report MUST populate every row. If a field genuinely cannot be measured, the
value is `null` and tagged `(U)`; a guessed HBM bandwidth taken from a vendor page for the
wrong SKU is worse than a null, because it is not tagged at all.

## 1.2 Why both HBM bandwidth and FLOP/s are mandatory

The two rooflines exist because the two phases of generation are bound by different
resources:

- **Decode is bandwidth-bound.** Each autoregressive step streams the active weights once
  and re-reads the KV of every in-flight sequence. `roofline_decode_tok_s` is a pure
  function of `hbm_bandwidth_bytes_s`, active parameters, and KV traffic. FLOP/s is
  irrelevant here at any realistic batch size.
- **Prefill is FLOP-bound.** Computing the prompt is dense math at roughly
  `2 * active_params` FLOPs per token. `roofline_prefill_ttft_s` is a pure function of
  `flops_per_s_dense` and an MFU assumption. Bandwidth is irrelevant here.

Declaring only one of the two specs makes one of the two rooflines uncomputable, which
makes the THEORETICAL tier uncomputable, which makes C6 impossible to satisfy. **This is a
conformance rule, not a nicety: a report with one roofline is a report that cannot
self-check.** The failure it prevents: an operator declares FLOP/s only, benchmarks a
decode-heavy workload, and has no upper bound to compare against — so a measurement that is
wrong by 3× (mis-declared active parameters, untracked cache hits) ships as a headline.

Both figures MUST be quoted at the precision actually deployed (Chapter 2), because an
FP8-quantized deployment runs against FP8 rates, and a 4-bit deployment like `nvfp4` has
rates many vendors list separately if at all.

## 1.3 Intra-node interconnect, and why TP width is meaningless without it

Tensor parallelism divides each layer across GPUs and synchronizes activations twice per
layer per step. On a decently sized model at decode, that synchronization traffic is
gigabytes per second sustained. Hence:

**Rule HW-1.** Any TP width MUST be reported together with the intra-node interconnect it
runs over (`interconnect.intra_node`), including measured or vendor-stated bandwidth.
*(Failure prevented: "TP=8" over a high-bandwidth fabric and TP=8 over PCIe ×16 are
different serving systems. Over PCIe the all-reduce dominates decode; throughput can land at
a small fraction of the same GPUs linked by NVLink-class fabric. A TP width declared without
its fabric reads as reproducible and is not.)*

**Rule HW-2.** A report MUST NOT describe a TP width as available "on this model of GPU."
Width is a property of the *node*, not the chip. Declare the node's link topology, and if
TP width exceeds the high-bandwidth island (e.g. TP=4 on a node whose fast fabric covers
only 2 GPUs), the report MUST say so.

**Rule HW-3.** Replicas crossing sockets or NUMA boundaries SHOULD be declared, because CPU
affinity changes measured latency tails at the margin. `(M)` if measured; otherwise `(U)`.

## 1.4 Inter-node fabric, and pipeline vs tensor parallel across nodes

When a model exceeds a node, parallelism must cross the fabric. The choice of *which* axis
crosses it is the single highest-leverage serving decision in multi-node deployments:

- **Tensor parallel across nodes** puts the per-layer all-reduce on the inter-node fabric.
  Where that fabric is an order of magnitude slower than intra-node links — Ethernet or
  RoCE at 100–400 Gb/s versus NVLink-class TB/s — decode throughput saturates on the
  collective long before compute. The result is a cluster that is demonstrably slower than
  a single larger node at the same GPU count.
- **Pipeline parallel across nodes** sends activations once per *layer boundary*, not once
  per layer. Traffic is lower by roughly the number of layers. Decode survives. The costs —
  pipeline bubbles at low batch, KV spread across nodes — are real but bounded.

**Rule HW-4.** Multi-node reports MUST declare which parallel dims cross the inter-node
fabric and the measured or specified fabric bandwidth and topology (fat-tree, rail-optimized,
torus). *(Failure prevented: a 2-node TP=8 result quoted as "8 GPUs" implies equivalence to
one 8-GPU node. They are not equivalent; the inter-node width is the constraint.)*

**Rule HW-5.** When pipeline depth > 1, the report MUST name the microbatch strategy only
to the extent it changes measured throughput, per Chapter 3. This chapter requires only
the declaration that cross-node parallelism exists and on which axis.

## 1.5 CPU, RAM, and storage: cold-start is a capacity number

A serving system that cannot load its weights in acceptable time is not a capacity result.
Weight load is gated by storage throughput, CPU deserialization, and (for some formats)
dequantization on the host. Large models on slow disks take tens of minutes to come up; a
"capacity" report that omits this is true only of an already-running server.

**Rule HW-6.** Reports MUST declare `storage` (medium and sequential-read figures if known)
and MUST report measured cold-start time — process start to first served request — as an
`(M)` value in Chapter 4 terms. *(Failure prevented: autoscaling and failover math assumes
instant startup. A 25-minute unacknowledged load time means the "extra replica" you planned
for burst capacity does not exist at the timescale of the burst.)*

CPU and RAM declarations catch a second failure: host-side preprocessing, tokenizer queues,
and framework runtime overhead binding below the GPU floor. If a benchmark is CPU-bound,
record it; a measurement made under an undeclared host bottleneck will be read as a GPU
property.

## 1.6 Node exclusivity

**Rule HW-7.** `topology.node_exclusive` MUST be declared. If any other tenant, job, or
benchmark ran on the node during measurement, every number in the report is contaminated and
MUST be labelled accordingly. *(Failure prevented: a co-scheduled job takes memory bandwidth
and PCIe lanes. Decode throughput falls by double-digit percentages with no record of why.
Two teams running "the same config" on shared clusters will get different answers and waste
weeks reconciling them.)*

This is the cheapest rule in the chapter to satisfy and the most commonly violated.

## 1.7 Single vs multi-node declaration

**Rule HW-8.** `topology.nodes` MUST be declared, and capacity figures MUST be tagged with
the node count they were measured at. A per-GPU figure from an N-node run MUST NOT be
presented as independent of N (binding C3). Cross-node deployments consume memory bandwidth
and wire time differently than single-node deployments of the same total GPU count;
quoting either as if it were the other is a category error the protocol treats as
non-conforming.

A report covering both regimes SHOULD measure at multiple node counts and declare each, so
that scaling behaviour is exposed rather than assumed.
