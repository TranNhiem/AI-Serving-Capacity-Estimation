# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project versions the **protocol**. The versioning rule is the one
stated in the README and `protocol/SPEC.md`: any change that would alter the
numbers in an already-conforming report — a formula in `ascep/capacity.py`, a
schema field's meaning, a conformance gate — is a **major** version bump.
Additive, number-preserving changes are minor or patch. Reports cite the
protocol version they were produced under, so the rule exists to keep
cross-version comparisons valid.

## [Unreleased]

### Added

- `ascep init` writes a fillable report skeleton derived from the schemas, for any
  single layer or the whole `capacity-report`. It refuses to overwrite an existing
  file without `--force`, writes the document to stdout so it can be piped, and
  puts its diagnostics on stderr. The skeleton is deliberately *invalid*: every
  value is `null` or `TODO`, so `ascep validate` reads as the fill-in list. Where
  a schema disjunction demands a choice no placeholder can make honestly — a
  workload's sizing basis, a measurement point's context length — `init` names the
  fields instead of inventing values.
- `capacity_tiers.<name>.tier` is now pinned to a `const` matching its key.
  Copying a tier row and forgetting to relabel it previously validated and passed
  every conformance gate, leaving two rows claiming to be the same tier.

### Changed

### Deprecated

### Removed

### Fixed

### Security

- C1 now rejects a leftover `TODO` or an empty string as an error. Every other
  rule in the conformance checker tests `is None`, so scaffolding text occupied
  the slot and read as a declaration — a report with
  `reproduction.raw_records_path: "TODO"` passed C8 while pointing at nothing.
  This also covers `*_u_reason` fields, which C1's null walk deliberately skips.

## [0.1.0] — 2026-08-31

Initial draft release.

### Added

- `protocol/SPEC.md`, the normative specification, including conformance rules
  C1–C8 (complete declaration, provenance tagging, topology binding, context
  binding, binding constraint, four tiers, pre-committed SLO gates, reproduction
  bundle) and the `partial` / `non-conforming` levels, plus the eight normative
  protocol chapters covering hardware, model, serving, measurement, the capacity
  model, application sizing, benchmark procedure and reporting.
- JSON Schema for the five declaration layers: hardware, model, serving, run
  and workload, including first-class `attention_type` declaration — `full`,
  `gqa`, `mqa`, `mla`, `sliding-window`, `hybrid`, `linear`, `ssm`,
  `hybrid-recurrent` — with per-family geometry validation, because the KV
  formula that applies is chosen by this field and the families differ by more
  than an order of magnitude.
- The `ascep` package: `capacity` — the transparent formula set, stdlib-only by
  design so it runs on an air-gapped login node — plus `conformance` (grades a
  report against C1–C8 and flags overstated self-declared levels), `render`
  (emits the Markdown report) and `validation` (schema checks, the only module
  needing the optional `[run]` extra), exposed through the `ascep validate`,
  `ascep conformance`, `ascep render` and `ascep size` CLI commands.
- `templates/capacity-report.md`, the standard report template.
- Worked examples under `examples/`: `moe-26b-h100-tp2`, a capacity report
  (MoE, 26B total / 4B active, bf16, on an 8-GPU H100 SXM node at TP=2), and
  `chatbot-10k-dau`, a workload declaration composed against the report's
  measured figures to arrive at 2 GPUs, throughput-bound.
- CI gates that recompute every `(I)` number in every example from its `(M)`
  numbers and regenerate each `report.json` from its `build_report.py`, so an
  example cannot drift from the formulas, alongside `tools/check_no_secrets.py`
  and accelerator-free `pytest` coverage of the formula set.

### Known gaps

- **No benchmark driver.** The harness that produces `report.json` from a live
  endpoint is not included; it is being generalized from a private benchmark
  campaign. Chapter 7 specifies the procedure so an existing harness can be
  used, and `examples/*/build_report.py` shows how to map results onto the
  schema.
- **One published configuration.** `examples/moe-26b-h100-tp2` is the only worked
  report: NVIDIA H100, vLLM, MoE, bf16, TP=2. Other accelerators, frameworks,
  attention families, precisions and topologies are supported by declaration —
  the protocol never assumes any of them, and the schemas and formulas cover
  them — but "supported by declaration" is a weaker claim than "demonstrated",
  and the difference is not something the repository should blur. A dense-model
  report on the same hardware is the nearest gap.
- **The published example report is deliberately `partial`.** It documents a
  campaign that predates the protocol, so several declarations (engine version,
  memory-utilization flag, prefix-caching state) are recorded as `null` with a
  stated impact rather than guessed. It is published that way because inventing
  the missing fields would be the exact failure C1 exists to prevent.
