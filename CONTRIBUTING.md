# Contributing to ASCEP

ASCEP is a measurement and reporting protocol. It gets more useful in exactly two ways: more conforming reports against real infrastructure, and a better protocol underneath them. Contributions of both kinds are welcome. This file describes how each is reviewed. Code of conduct: see `CODE_OF_CONDUCT.md` at the repository root. All participation is subject to it.

## The most valuable contribution: a conforming capacity report

The single highest-leverage thing you can add is a capacity report under `examples/` — a real measurement of a real model on real hardware, declared against all five layers, tagged with provenance, and passing the conformance rules. One conforming report for a 2-KV-head GQA model at TP=4, showing the KV replication penalty in numbers, teaches more than a paragraph of prose ever will.

Reports live in `examples/<model>-<hardware>-<framework>/`. Copy `examples/moe-26b-h100-tp2/` and edit it; the layout there is the one reviewers expect:

- `report.json`, validating against `schemas/capacity-report.schema.json`. Start it with `ascep init -o report.json`, which emits every field the schemas require as `null` or `TODO` — the validation errors on a fresh skeleton are your fill-in list, not a bug. All five layer declarations are sections **inside** it — `hardware`, `model`, `serving`, `run`, `workload` — each validated against the matching schema in `schemas/`. One file, not five: the layers are only meaningful together, and a directory of separate files makes it possible to commit a `model` that no `run` ever measured.
- `build_report.py`, which regenerates `report.json` from your raw measurements. Commit both. CI runs the builder and diffs the result, so a report that cannot be rebuilt from its inputs fails before review.
- Every capacity figure carrying its tier (theoretical, measured, sustainable, recommended), its binding constraint, and its provenance tag, plus the `unmeasured_assumptions` list. `ascep render report.json` produces the Markdown presentation, so there is no `REPORT.md` to write or keep in sync.
- The reproduction bundle required by C8 — run configs, per-request raw records, engine version and container digest, environment capture — or, where it genuinely cannot be published, the paths declared and the omission stated. Without it the report caps at `partial`. **That is a publishable verdict**, and the repository's own example carries it; the rejection is for claiming a level you did not earn, not for earning a modest one.

Full hardware details MAY be coarsened to the schema (e.g. "8× 141 GB HBM GPUs, NVLink-class interconnect") — comparability requires the declared fields, not your datacentre layout.

### Review criteria

Submitted reports are checked mechanically and by review against C1–C11:

| rule | what reviewers verify | typical rejection |
|---|---|---|
| C1 complete declaration | every required schema field present; unknowns recorded as `null` with a `(U)` entry, never omitted | missing `global_layer_frac`; guessed value where a `null` belongs |
| C2 provenance tagging | every number carries exactly one of (M), (I), (T), (U); every (I) names the `ascep.capacity` function that produced it | untagged throughput figure; (I) with no function citation |
| C3 topology binding | capacity/KV/throughput reported with TP width, pipeline depth, GPU count; no per-GPU figure presented as topology-independent | "per-GPU KV" quoted without TP |
| C4 context binding | throughput bound to input/output token counts; single-point runs labelled as such; three-point curve where claimed (SHOULD) | headline tokens/s with no context length |
| C5 binding constraint | every capacity figure names `weights`, `kv`, `throughput` or `slo`; crossover stated or marked undetermined | "supports N users" with no floor named |
| C6 four tiers | all four tiers reported | measured-only report presented as production guidance |
| C7 gates fixed up front | SLO thresholds in the run config, committed before results | gate values that suspiciously match the observed tail |
| C8 reproduction bundle | all four artefacts present and loadable | raw records absent; container digest missing |

A report meeting C1–C5 but not C6–C8 MAY be merged labelled **partial**. Anything less is non-conforming and MUST NOT be merged under `examples/`; put exploratory work in a discussion thread instead. Reviewers also reject on the protocol's own red flags: roofline efficiency at or above 1.0 without an investigation note, or `scaling_efficiency` above 1.0 reported as a win rather than as evidence of a KV-starved baseline.

## Adding a serving-framework adapter

Adapters translate a framework's native output (vLLM, SGLang, TGI, TensorRT-LLM, or any other) into the `run` and `serving` schemas. An adapter MUST:

1. Populate every required schema field it can observe, and emit `null` plus a `(U)` entry for the rest — adapters MUST NOT substitute defaults for unobserved values. Defaulting `memory_utilization` to 0.90 when the framework reported nothing is exactly the silent disagreement C1 exists to prevent.
2. Map the framework's memory-fraction knob explicitly (vLLM `gpu_memory_utilization`, SGLang `mem_fraction_static`, etc.) and record the raw value, untouched.
3. Prefer the engine-reported KV cache size over the analytic `kv_pool_bytes` path, and call `calibrate_memory_utilization` to reconcile the two when both exist.
4. Ship with a fixture: a captured real framework output plus the expected schema output, so refactors are caught by unit tests without needing a GPU.

An adapter MUST NOT change raw measurements — no unit rescaling, no percentile smoothing, no dropped outliers. Emit the framework's numbers as it reported them and let the reduction happen downstream. There is no shared reducer module in v0.1, so if your adapter needs one, propose it as its own PR rather than folding the arithmetic into the adapter where the next framework cannot reuse it or check it.

## Proposing a change to the spec

Open an issue before a PR; spec text changes are debated on their failure modes, not their wording. When you propose a change, state which conforming reports' numbers would change.

**Versioning rule (normative).** Any change that would alter the numbers in an already-conforming report — a formula in `ascep/capacity.py`, a schema field's meaning, a conformance gate — is a **major** version bump. Additive, number-preserving changes (new optional fields, new chapters, clarifications) are minor or patch. Reports cite the protocol version they were produced under, so a silent semantic change would make cross-version comparisons invalid without anyone noticing. If your change alters numbers, say so in the PR title.

## Code style and tests

`ascep/capacity.py` is the analytic half of the protocol: pure functions, no GPU, no network, no I/O. Contributions to it MUST preserve that — every function closed-form, unit-annotated in its name (`_bytes`, `_tokens`, `_s`, `_tok_s`), with its assumptions in the docstring. Use `GIB` and `GB` explicitly; no implicit conversions.

Every formula change ships with unit tests covering: the happy path, boundary inputs (zero batch, `headroom=1.0`, `global_layer_frac` at both ends), and the error paths (negative bandwidth, empty curve). Tests are ordinary `pytest` and MUST run without accelerators of any vendor.

## What must never appear in a contribution

Submissions MUST NOT contain, in code, configs, reports, or bundles:

- **Internal hostnames, IPs, or cluster topology** beyond what the hardware schema declares. Sanitize environment captures before committing.
- **Credentials** of any kind — tokens, keys, kubeconfigs, cloud metadata. Scrubbed means *absent*, not redacted-in-place.
- **Customer or partner pricing.** ASCEP produces GPU counts, not prices; cost figures are regional, contractual and volatile, and including them poisons comparability.
- **Unreleased model weights, checkpoints, or identifying internals.** Reports on unreleased models may describe architecture parameters only with the model owner's explicit permission, stated in the PR.

A contribution containing any of the above will be rejected and history-rewritten, not patched.
