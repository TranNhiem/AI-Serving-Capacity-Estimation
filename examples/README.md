# Worked examples

Each subdirectory is an artifact you can diff your own against. Every `(I)` number in every
example is recomputed from its `(M)` numbers in CI, so an example cannot drift from the
formulas that produced it.

There are three shapes — two directions of the question, and two ways of answering the first:

| example | asks | shape |
|---|---|---|
| [`moe-26b-h100-tp2`](moe-26b-h100-tp2/) | *given this hardware, what can it serve?* | **capacity report** |
| [`chatbot-10k-dau`](chatbot-10k-dau/) | *given this product, what hardware do I buy?* | **workload declaration** |
| [`qwen3-vl-4b-h100-image-qa`](qwen3-vl-4b-h100-image-qa/), [`gb200-gemma4-31b-tp1`](gb200-gemma4-31b-tp1/) | *the same question, answered by a run rather than a transcription* | **bundle-backed capacity report** |

## Capacity reports

Capacity reports come in two provenance shapes. The hand-assembled shape holds three files:

| file | what it is |
|---|---|
| `run-summary.json` | the measured facts, hand-written, in whatever shape the campaign produced |
| `build_report.py` | maps those facts onto the schema, deriving every `(I)` via `ascep.capacity` |
| `report.json` | **generated** — the conforming artifact. Never hand-edit it |

The split is the point. `run-summary.json` is what you actually observed; `report.json` is what
the protocol says about it. Keeping them apart means the derivation is a readable script rather
than an unexplained jump from measurements to conclusions, and
`tests/test_report_conformance.py` regenerates `report.json` and diffs it — so a number cannot
be quietly edited into agreeing with a conclusion it no longer supports.

The bundle-backed shape has no builder, because nothing was transcribed. `ascep bench` drove the
run and emitted the report, and the evidence ships beside it:

| file | what it is |
|---|---|
| `report.json` | **generated** by the harness — the conforming artifact. Never hand-edit it |
| `vllm_server.out` or `engine.log` | the engine log, mirrored for the whole run rather than referenced where the scheduler wrote it. The name is the example's own; the manifest is what binds it |
| `bundle/manifest.json` | sha256 over every other artifact, including the engine log above it |
| `bundle/environment.json` | probed package versions, including the harness that produced the numbers |
| `bundle/records.jsonl` | one line per request — every latency in the report is recomputable from it |
| `bundle/bench-config.json` | the ladder, the windows, and the SLO gates the run was driven with |
| `bundle/run_configs.json` | what each window in that ladder actually executed |
| `bundle/declarations/` | the four declarations the run was made under |
| the declarations and `bench.json` at the example root | optional, and worth doing: it makes the campaign re-runnable with one command, and the hashed copies in `bundle/declarations/` are what prove the files at the root are the ones that ran |

The provenance is stronger, not weaker. A builder proves the report follows from facts somebody
transcribed; a bundle proves it follows from a run that happened, because the inputs are hashed,
the environment is recorded, and the per-request records are shipped for recomputation.
`tests/test_report_conformance.py` keeps both honest: with a builder it regenerates and diffs,
and without one it requires the bundle and checks it. The absence of a builder is a claim that
has to be backed by artifacts, never an exemption from having any.

| example | model | hardware | topology | conformance | binding constraint |
|---|---|---|---|---|---|
| [`moe-26b-h100-tp2`](moe-26b-h100-tp2/) | MoE, 26B total / 4B active, bf16 | 1× 8-GPU H100 SXM node | TP=2 | **partial** | throughput |
| [`qwen3-vl-4b-h100-image-qa`](qwen3-vl-4b-h100-image-qa/) | dense Qwen3-VL-4B-Instruct, bf16, vision | 1× H100 SXM | TP=1 | **partial** | slo |
| [`gb200-gemma4-31b-tp1`](gb200-gemma4-31b-tp1/) | dense Gemma-4-31B-it, bf16, hybrid attention | 1× NVIDIA GB200, Grace host | TP=1 | **partial** | slo |
| [`gb200-gemma4-31b-multi-image`](gb200-gemma4-31b-multi-image/) | dense Gemma-4-31B-it, bf16, vision | 1× NVIDIA GB200, Grace host | TP=1 | **partial** | slo |
| [`gb200-qwen25-vl-32b-multi-image`](gb200-qwen25-vl-32b-multi-image/) | dense Qwen2.5-VL-32B-Instruct, bf16, vision | 1× NVIDIA GB200, Grace host | TP=1 | **partial** | slo |
| [`gb200-qwen25-vl-32b-video`](gb200-qwen25-vl-32b-video/) | dense Qwen2.5-VL-32B-Instruct, bf16, video | 1× NVIDIA GB200, Grace host | TP=1 | **partial** | slo |

## Workload declarations

A workload declaration is layer 5 alone: a product forecast with no measurements of its own.
It becomes an answer only when composed with per-GPU figures measured somewhere else — which
is exactly what `chatbot-10k-dau` does, borrowing the H100 numbers from the report above and
arriving at **2 GPUs, throughput-bound**, where chapter 6's illustrative figures said 4.

Keeping the two apart is the honest arrangement: the workload half travels between deployments,
the measured half never does. `tests/test_application_sizing.py` recomputes the composition and
also checks that the example's prose still quotes the numbers the formulas produce.

| example | application | DAU | peak concurrency | avg context | binding constraint |
|---|---|---|---|---|---|
| [`chatbot-10k-dau`](chatbot-10k-dau/) | chat assistant, short context | 10,000 (U) | 556 (I) | 1,200 tokens (I) | throughput |

## Same grade, different distance: how to read three `partial` reports

Take the first three capacity reports this directory held, [`moe-26b-h100-tp2`](moe-26b-h100-tp2/),
[`qwen3-vl-4b-h100-image-qa`](qwen3-vl-4b-h100-image-qa/), and
[`gb200-gemma4-31b-tp1`](gb200-gemma4-31b-tp1/). All three are graded `partial`, as is every
media campaign added since. That is
the teaching point. `partial` is not one condition but a spectrum, and the spectrum has a
direction: it runs from "the information was never captured" toward "everything was captured
except this one thing." The three examples do not spread evenly along it. One sits near the
far end, and the other two sit so close together that their declared gap lists are identical —
and they are still not the same distance, because what a `null` costs to close is set by why
it is `null`. Neither the grade nor the gap list alone tells you which report you are looking
at. That is why the protocol publishes the reasons, not just the grade.

[`moe-26b-h100-tp2`](moe-26b-h100-tp2/) — MoE, 26B total with 4B active, bf16, one 8-GPU H100
SXM node, TP=2, bound by `throughput` — is partial because it documents a benchmark campaign
that predates the protocol. Several declarations were simply never captured: the serving
`framework_version`, the `memory_utilization` flag, whether `prefix_caching` was on. Rather
than guess them, they are recorded as `null`, with `unmeasured_assumptions` entries giving the
`impact_if_wrong` and the `cost_to_measure` for each gap. It publishes no `theoretical` tier
at all — the KV geometry the roofline needs was never captured, so the analytic path is
unavailable. What rescues the report is the protocol's own rule of preferring the
engine-reported KV size over the analytic model: the measured path still yields a defensible
answer. That is the honest outcome, and it is more instructive than a polished one. It shows
what `partial` conformance looks like in practice, and that partial is a legitimate,
publishable state — the alternative is people quietly inventing the missing fields.

[`qwen3-vl-4b-h100-image-qa`](qwen3-vl-4b-h100-image-qa/) — dense Qwen3-VL-4B-Instruct, bf16,
one H100 SXM, TP=1, vLLM 0.11.0, open-ended visual question answering over a FineVision
image-QA split — sits much closer to the other end. Its one reproducibility gap is
`container_digest`: the serving process ran from a scheduler-managed Python environment rather
than a pinned immutable image, so there is no digest to record, and a note cannot substitute
for one. Every declaration the first report leaves `null`, this one states:
`framework_version` is `0.11.0`, `memory_utilization` is 0.9, `prefix_caching` is on, and
`engine_reported_kv_cache_tokens` is 429,952. The provenance also differs in kind. The run was
driven end to end by this project's own harness, `ascep bench`, and ships a hashed manifest, a
probed environment capture, the mirrored engine log, and one record per request, so every
figure in the report can be recomputed rather than taken on trust.

[`gb200-gemma4-31b-tp1`](gb200-gemma4-31b-tp1/) — dense Gemma-4-31B-it, 31,273,088,876
parameters, bf16 weights and KV, one NVIDIA GB200, one GPU of a four-GPU Grace-Blackwell tray
with 185.0 GiB of HBM, TP=1, vLLM 0.18.2rc1.dev73+gdb7a17ecc, a synthetic text workload
with a fixed 500-token output under `ignore_eos`, bound by `slo` — arrives at nearly the same
spot as the vision report by a different road. It is this repository's first campaign on
non-H100 hardware, its first on an Arm host, and its first on a hybrid-attention model — a 5:1
local:global layout with a 1,024-token sliding window — and it declares all of that rather
than trading on novelty. Like the vision report it states everything the MoE report left
`null`, ships the same hashed-manifest provenance from `ascep bench`, and is capped by the
same single field: `container_digest` is `null`, and its development-build `framework_version`
is the only handle on what actually served.

The two bundle-backed reports are a matched pair on paper. Their `unmeasured_assumptions`
field lists are identical: `roofline_comparison`, `sizing_result`, the scaling table with one
topology only, `run.tokenizer`, the weights and KV floors under
`capacity_tiers.*.binding_constraint`, the `gpu_util_pct` and `gpu_mem_util_pct` entries a
load generator cannot observe, `run.engine_version`, and the percentiles published below the
section 4.3 advisory sample floor. Same grade, same gap list — and still not the same
distance, because the reason behind a `null` sets its cost to close. The vision report's
digest is `null` because there was no image at all: a scheduler-managed Python environment
served the model, so closing the gap means changing how the service is deployed. The GB200
report's digest is `null` because there *was* an image — an enroot container instantiated from
a local squashfs file — and a squashfs carries no registry digest. Closing that gap means
publishing the image to a registry, not redeploying anything. One null is a deployment
decision; the other is an upload. Identical fields, different invoices.

What the trio does *not* show is one report partial and the others nearly conforming on every
axis. All three leave `run.engine_version` `null` — the load generator observes an HTTP
endpoint, not a build — and none publishes a `theoretical` tier. Even that shared absence has
a direction. The MoE report *cannot* publish one: `n_layers`, `n_kv_heads` and `head_dim` were
never captured, so `kv_bytes_per_token` is uncomputable and the gap is a lost measurement. The
two newer reports simply have not run the analytic path, because the roofline belongs to
`ascep size` and both bundles came from `ascep bench` — a command not yet run, not a fact
destroyed. The tier tables carry the same lesson in numbers. The MoE report's `measured` and
`sustainable` tiers are the same figure, 459.8 — its one declared gate, TTFT p95 at 2.0 s, was
met at the very rung that maximised throughput, so the two tiers coincide rather than the gate
being absent — and it alone carries a `recommended` tier, at 399.9, because it is the only
deployment with a declared headroom policy to derate against. The two harness-driven reports
are bound by `slo`, and for them the tiers split: `measured` 128 against `sustainable` 64 for
the vision report, `measured` 128 request streams against `sustainable` 16 for the GB200. Both
`measured` figures follow chapter 5.5's deliberately gate-blind definition — best observed,
SLO ignored — and the 2x and 8x gaps between best-observed and last-passing-rung are exactly
why that tier is gate-blind: an identical `measured` 128 supports a promise of 64 in one
deployment and only 16 in the other.

The GB200 report also contributes two things neither predecessor has. It is the only report
whose analytic KV model was checked against the engine's own allocation: 2 × 2 bytes × 16 KV
heads × 256 head_dim × 60 layers is 983,040 bytes per token, and times the engine's reported
114,544 tokens that is 104.868 GiB against the 104.88 GiB the engine logged — agreement to
0.011 percent. The same check exposed the engine's own arithmetic as internally inconsistent:
the log claims a maximum concurrency of 10.99x for 32,768 tokens per request, while 114,544
divided by 32,768 is 3.50x, so the scheduler admits roughly three times what the reported pool
implies. The report therefore records `engine_reported_kv_cache_tokens` as a lower bound and
its derived KV floor as conservative in a known direction. A gap with a known sign is not the
same kind of gap as one without: the MoE report cannot say which way its missing fields would
push the answer; this one can. It is also the first report to declare its own narrowness in a
machine-readable field rather than in prose. The whole ladder ran at one context shape —
per-rung mean context 2,043.7 to 2,046.5 tokens, a spread of 0.14 percent — so
`run.single_point` is true, satisfying C4 by declaration where C4 wanted three context lengths
or that flag. The report claims nothing about context scaling and says so in the schema. And
the flag itself is evidence of what running costs: it had never once fired before this
campaign, because the condition tested distinct floats and six near-identical means counted as
six lengths. Reading the code had not exposed that. Running the tool against real hardware
did.

Two of the reports also close a loop, and the third keeps it closed. The MoE report names
`prefix_caching` as the single unmeasured field its conclusion is most sensitive to: if it was
on and the campaign reused prompt prefixes, the measured throughput overstates production,
while every other gap affects projection to untested configurations rather than the headline.
The vision report measures that very field and then defeats it deliberately. Prefix caching is
on in the engine, but the workload's cache policy is unique-prefix — a distinct token
prepended to every request, so no two requests share a prefix and the cache has nothing to
return. The reason is rung ordering: the ladder climbs 1 to 128, and a shared prefix would let
each rung answer partly out of a cache the rung below it filled, the error running one way
only, every higher rung looking faster than it is. The GB200 report runs the same deliberate
combination for the same reason — caching on in the engine, a unique prefix in the workload —
as its ladder climbs 4 to 128. What one report said it would have taken to close its own
headline gap, two others now do as a matter of course. That is the repository demonstrating
its own remedy, twice.

None of that narrowness buys absolution, and adding a third report does not dilute the
verdict. All three are still `partial`. The newest is not better for being newest: the same
development build that let it check its own KV model is the build no digest pins, and the
narrowness it declares so precisely is narrowness nonetheless — one GPU of a tray, one
workload shape, one context length, text only. A narrower gap is not a smaller obligation to
declare it. The digests are `null`, the register says so, and the grades stay what they are
until the gaps close.

## Contributing an example

A conforming report from hardware or a model nobody has published is the most valuable
contribution to this project — more than a code change. See [CONTRIBUTING.md](../CONTRIBUTING.md).

Requirements:

1. If you drove the run with `ascep bench`, write no builder at all: ship the emitted
   `report.json`, the `bundle/` beside it and the mirrored engine log, and let the manifest
   carry the provenance. Otherwise take the hand-assembled route — a `run-summary.json` with
   your measured facts and a `build_report.py` that emits `report.json`. Copy
   `moe-26b-h100-tp2`'s builder and edit it; most of it is declarations. For a workload-only
   contribution, copy `chatbot-10k-dau` instead: a `build_workload.py` emitting a
   `workload.json`, and no measurements to justify.
2. Every `null` carries a `<field>_u_reason` starting with `(U)`, or an entry in
   `unmeasured_assumptions` with `impact_if_wrong` and `cost_to_measure`. Enforced in both
   directions: an unjustified null fails, and so does a justification left behind after the
   field was measured.
3. The `unmeasured_assumptions` entry a reviewer should attack first states what the number
   becomes if the assumption is wrong. "May affect results" is not an `impact_if_wrong`.
4. No internal hostnames, credentials, customer names or pricing — run
   `python tools/check_no_secrets.py` before opening the PR.
5. `pytest tests/` green. Reports are picked up automatically by the `examples/*/report.json`
   glob and regenerated by the `examples/*/build_*.py` glob; there is nothing to register.

Declining to state a tier is allowed and is sometimes the correct answer — this example
publishes no `theoretical` row, because the KV geometry needed for the roofline was never
captured and a weight-bandwidth-only bound would have understated its own error. An honest
`null` with a reason beats a roofline nobody can defend.

Partial reports are welcome. Reports that guess at fields they did not measure are not.
