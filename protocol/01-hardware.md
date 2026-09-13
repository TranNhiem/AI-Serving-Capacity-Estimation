# Chapter 1 — Hardware Declaration

Layer 1 answers one question: *what silicon, how connected*. Every capacity number in the
report is a function of this layer, so this chapter defines the conformance floor for
declaring it. All fields map to `hardware.schema.json`; unknown values MUST be recorded as
`null` with a `(U)` entry (C1), never omitted and never guessed.

## 1.1 Required fields

| field | unit | required when | why it is required | failure it prevents |
|---|---|---|---|---|
| `accelerator.model` | string | always | "Radeon", "TPU" or "80GB card" names families, not parts | A reader reproduces on a different SKU with 2× different bandwidth and gets a different answer from the same config |
| `accelerator.count` | int | always | CPU-visible device count per node and in total | C3 topology binding: without a count, per-GPU KV and throughput figures are unfalsifiable |
| `memory_architecture` | string (`discrete` \| `unified`) | always | Selects which memory-budget question the rest of the table asks (§1.8); every conditional row keys off it | A reader sums "GPU memory" and system RAM on a unified part and prices a machine that does not exist |
| `accelerator.vram_bytes_per_gpu` | bytes | `memory_architecture == "discrete"` | Feeds `kv_pool_bytes` | Analytic and measured KV capacities silently disagree by the difference between SKUs (e.g. 40 vs 80 GiB variants of the same accelerator). On a unified part this row has no honest answer (§1.8); `usable_memory_bytes` carries the budget |
| `usable_memory_bytes` | bytes | `memory_architecture == "unified"` | Feeds `kv_pool_bytes` on unified parts: the bytes the framework can actually bind for weights plus KV after the OS, the compositor and the wired-memory limit | Declaring total system memory instead overstates the bindable pool by tens of GB; every KV floor computed from it promises concurrency no configuration of the machine can hold |
| `memory_type` | string | `memory_architecture == "unified"` | Annotates the bandwidth figure: `"LPDDR5X"` and `"HBM3e"` at the same bytes/s are different machines — one bus is shared with the CPU and display, one is not | A bandwidth number read as a private GPU resource when it is the whole system's DRAM bus |
| `accelerator.hbm_bandwidth_bytes_s` | bytes/s | always | Sole input to `roofline_decode_tok_s`; accelerator-visible DRAM bandwidth, annotated by `memory_type` | Without it the theoretical tier cannot be computed, and C6 (four tiers) cannot be met |
| `accelerator.flops_per_s_dense` | FLOP/s | always (honest `null` + `(U)` where no dense figure is published — §1.2) | Sole input to `roofline_prefill_ttft_s` | Same: no roofline, no roofline efficiency, no conformance. Declared as dense FLOP/s at the deployed precision, not sparse — sparse marketing numbers inflate the roofline ~2× and make measured efficiency look broken |
| `dense_flops_precision` | enum | whenever the FLOP/s figure above is non-null | Names the precision (`bf16`…`nvfp4`) the quoted dense FLOP/s is measured at | An FP4 headline read as dense BF16 doubles the theoretical prefill bound per precision step and sends efficiency analysis chasing a phantom (HW-10) |
| `thermal_behavior` | enum | `memory_architecture == "unified"` | States whether the part holds its datasheet bandwidth and FLOP/s across the sustained measurement window (`not-assessed` is a legal answer) | A cold-started sprint quoted as sustained capacity; two honest sweeps — cold and heat-soaked — read as a discrepancy that isn't one |
| `interconnect.intra_node` | enum + bandwidth | `gpus_per_node > 1` (single-device parts declare `n-a`) | Determines the TP widths that are viable (§1.3) | TP=8 over PCIe is measured, published, and off by multiples versus what the same GPUs achieve over a high-speed fabric |
| `network.inter_node` | enum + bandwidth + topology | `nodes > 1` | Determines whether a model is one node or many, and which parallel dim crosses nodes (§1.4) | A cross-node TP result presented as a single-node option |
| `cpu.model`, `cpu.cores`, `ram_bytes`, `storage.*` | varies | always | Cold-start and host-side bottlenecks live here (§1.5); on unified parts `ram_bytes` aliases the accelerator pool (HW-10) | Load time omitted; a capacity report that is true only for an already-warm server |
| `topology.node_exclusive` | bool / enum | always | Gate on every other number (§1.6) | A shared node invalidates all of it silently |
| `topology.nodes`, `topology.single_node` | int, bool | always | Multi-node is a different capacity regime (§1.7) | A 2-node result quoted per-GPU as if nodes were fungible |

A conforming report MUST populate every row its condition selects. The conditions are not
discretion: a row whose guard is false is answered by `null` with a `(U)` reason or by the
sanctioned `n-a` string, never by silence and never by a guess. If a field genuinely cannot
be measured, the value is `null` and tagged `(U)`; a guessed HBM bandwidth taken from a
vendor page for the wrong SKU is worse than a null, because it is not tagged at all.

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

One qualification the protocol insists on: mandatoriness attaches to the *declaration*, not
to the existence of a figure. Apple publishes no GPU FLOP/s number at any precision, and
back-computing one from core counts and clocks is guesswork wearing a datasheet's clothes.
On such a part the honest, conforming value is `null` with a `(U)` reason — and the
consequence is accepted, not papered over: the prefill theoretical tier and
`roofline_efficiency` are then null as well, and the report shows that rather than
manufacturing a bound it cannot defend. A report that invents the figure has converted an
unknown into a silent non-conformance, which is the one thing the protocol exists to
prevent.

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

A single-device part answers HW-1 trivially: with nothing to shard across,
`interconnect_intra_node` takes the literal string `n-a` — the same escape
`interconnect_inter_node` already documents for single-node reports. Do not substitute the
part's CPU-side PCIe or package link: quoting it as an "interconnect" invites a reader to
compare it against NVLink-class figures, and the comparison is a category error.

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
property. On a unified part this caveat sharpens: the CPU's traffic rides the same DRAM
bus the serving framework is measuring, so a CPU-bound run depresses the memory roofline
itself, not just the host pipeline in front of it (§1.8).

## 1.6 Node exclusivity

**Rule HW-7.** `topology.node_exclusive` MUST be declared. If any other tenant, job, or
benchmark ran on the node during measurement, every number in the report is contaminated and
MUST be labelled accordingly. *(Failure prevented: a co-scheduled job takes memory bandwidth
and PCIe lanes. Decode throughput falls by double-digit percentages with no record of why.
Two teams running "the same config" on shared clusters will get different answers and waste
weeks reconciling them.)*

This is the cheapest rule in the chapter to satisfy and the most commonly violated.

The declaration admits `exclusive-process`: no other compute or graphics tenant ran during
measurement, while the OS compositor and system services held their unavoidable share. Use
it whenever that sentence is the true one — which on Apple Silicon is always, because no
Mac can honestly declare `exclusive`: WindowServer holds GPU time from boot and cannot be
scheduled out. Without this value every Apple report would carry the HW-7 contamination
banner forever, and a banner that flies on every report from an entire platform stops
meaning anything — which is how readers learn to ignore it on the reports where it still
matters.

## 1.7 Single vs multi-node declaration

**Rule HW-8.** `topology.nodes` MUST be declared, and capacity figures MUST be tagged with
the node count they were measured at. A per-GPU figure from an N-node run MUST NOT be
presented as independent of N (binding C3). Cross-node deployments consume memory bandwidth
and wire time differently than single-node deployments of the same total GPU count;
quoting either as if it were the other is a category error the protocol treats as
non-conforming.

A report covering both regimes SHOULD measure at multiple node counts and declare each, so
that scaling behaviour is exposed rather than assumed.


## 1.8 Unified memory: the wrong question and the right one

The field table above was written against datacentre parts with HBM soldered beside the GPU:
a private pool, described by `vram_bytes_per_gpu`, that nothing else on the machine can
touch. ASCEP also covers small single-device targets — Grace Blackwell superchips, Apple
Silicon — where CPU, GPU, operating system and display compositor allocate from one pool of
LPDDR. Two of the table's premises break there.

**`vram_bytes_per_gpu` is the wrong question.** There is no per-accelerator memory to count.
Any answer is either total system memory — which no serving framework can bind — or a
fabricated partition the silicon does not have. The right question is `usable_memory_bytes`:
the bytes the framework can actually bind for weights plus KV after everything that does not
go away has taken its share. That budget MUST exclude:

- the OS reservation and running system services;
- the display compositor, which on macOS holds GPU memory and GPU time whether or not a
  benchmark is running — WindowServer does not idle politely;
- the platform's GPU wired-memory limit (on Apple Silicon, `iogpu.wired_limit_mb`,
  defaulting to roughly 65–75% of RAM; a 512 GB machine does not offer the framework
  512 GB); and
- whatever the framework itself reserves outside the weight-and-KV pool.

The failure this guards against is directional, and the direction matters: declaring total
system memory overstates the bindable pool by tens of GB **in the over-promising direction**.
KV floors computed against it promise concurrency no configuration of the machine can
deliver; the reader re-runs, fails to reach the headline, and goes looking for the bug in
their own setup, because the wrong number arrives dressed as a measured one. Establishing
the true figure takes minutes — attempt the allocation, read the wired limit, check the
framework's startup log — so there is no honest reason to skip it.

**Rule HW-9.** A report MUST declare `memory_architecture`. On a `unified` part,
`usable_memory_bytes` MUST be the measured or documented bindable budget, not total system
memory, and the method used to establish it MUST appear in `notes`. *(Failure prevented:
the over-promising direction, at machine scale. The KV model prices floors against memory
that will never exist, and every capacity figure derived downstream is unreachable by
construction.)*

**Rule HW-10.** On a `unified` part, `system_ram_bytes` aliases the same silicon as the
accelerator pool and MUST NOT be added to it. *(Failure prevented: a reader summing the two
double-counts the machine and provisions against imaginary capacity. Summing VRAM across
discrete GPUs is legitimate; summing RAM and "GPU memory" on unified silicon counts one
pool twice.)*

These parts also stress the precision honesty §1.2 demands of every part. The headline
rates small superchips advertise are increasingly FP4, not the dense BF16 a prefill
roofline for a BF16 deployment needs; Apple's GPU publishes no FLOP/s figure at any
precision, where the conforming declaration is `null` + `(U)` and a null prefill tier, per
§1.2.

**Rule HW-11.** A quoted `dense_bf16_flops_per_s` MUST carry `dense_flops_precision`. A
sparse or FP4 headline figure quoted as if dense overstates the prefill bound by roughly 2×
per precision step — the DGX Spark's 1 PFLOP is FP4, not dense BF16. *(Failure prevented:
declared as if dense at the deployed precision, that figure would price the prefill
roofline four times higher than the silicon can run it, and an honest 60%-of-roofline
measurement would read as 15% — dispatching the reader on an efficiency hunt for a
shortfall that is a units error.)*

Finally, small single-device parts throttle under sustained load in a way H100-class
parts with datacentre cooling mostly do not. `thermal_behavior` declares whether the part
holds its datasheet bandwidth and FLOP/s across the measurement window. `not-assessed` is a
legal, honest value: it costs the reader certainty, which is cheap, where a fabricated
`sustained` costs them truth, which is not.

**Rule HW-12.** On a part declared `throttles-under-sustained-load`, the measured and
sustainable tiers MUST state the thermal state at the start of the window. *(Failure
prevented: a cold-started sweep and a heat-soaked sweep of the same part are different
measurements. Quoted without the label, two honest reports differ by a double-digit
percentage and two teams burn weeks reconciling a "regression" that is a fan curve.)*
