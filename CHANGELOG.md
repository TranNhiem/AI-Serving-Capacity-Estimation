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
- `run.single_point`, the field C4 always required and no schema provided. C4
  says a campaign covering fewer than three context lengths MUST be labelled as
  such; there was nowhere to put the label, so the same three rows read as a
  curve. Setting it does not raise the grade — a single point still caps at
  `partial` — it records that the limit was known.

### Changed

- `capacity_tiers.<tier>.headroom` now states in its description that it is a
  **divisor**, matching `sizing_result.headroom_factor`. The two descriptions
  previously said "factor" and "divisor" for the same quantity, so a generator
  and a validator reading them literally would disagree on the recommended tier
  by the square of the headroom.
- `ascep.capacity.fits` raises `ValueError` when `min_kv_tokens` is asserted
  without `kv_per_token`. It previously returned `True` as soon as the weights
  loaded, so a caller who explicitly asked for a KV floor got a "fits" verdict
  for a configuration nobody had checked.

### Deprecated

### Removed

### Fixed

- The `chatbot-10k-dau` document-assistant walkthrough quoted its two capacity
  floors (610 KV, 784 throughput) at **4 GPUs** immediately after a table of
  2-GPU floors, without saying so. The throughput floor appeared to *rise* with
  longer context, which is an artifact of doubling the fleet — at equal GPU
  count it falls from 761 to 392. Both fleet sizes are now labelled and the
  like-for-like comparison is shown.
- `ascep/capacity.py` and `CONTRIBUTING.md` referred contributors to
  `ascep.sweep` and `ascep.metrics`, neither of which exists in this release.
- C1's placeholder check matched `TODO` as a substring, so a real path such as
  `s3://bench/TODO-migration/records.jsonl` was rejected as scaffolding. It now
  matches the whole value, case-folded, plus the one `(U) TODO:` prefix `ascep
  init` writes — the strings the toolchain itself produces, which is what the
  rule always claimed to catch.
- A stale `*_u_reason` beside a filled-in field produced two C1 errors at one
  path with opposite instructions ("remove this" and "fill this in"). The null
  walk owns that case; the placeholder walk now skips it.
- `ascep.init` no longer writes `1970-01-01T00:00:00Z` into `date-time` fields.
  A parseable sentinel is the worst kind of placeholder: it validates, it
  survives `grep TODO`, and it reads as a real generation date. Those fields get
  `TODO` like every other string, and C1 rejects the epoch wherever it appears —
  including in `examples/moe-26b-h100-tp2`, which was publishing it.
- `ascep.init`'s docstring claimed `if`/`then` requirements surface through
  `decisions()`; only `anyOf`/`oneOf` ever did. The docstring now says what
  actually happens and why it is the right behaviour.

### Security

- C1 now rejects a leftover `TODO` or an empty string as an error. Every other
  rule in the conformance checker tests `is None`, so scaffolding text occupied
  the slot and read as a declaration — a report with
  `reproduction.raw_records_path: "TODO"` passed C8 while pointing at nothing.
  This also covers `*_u_reason` fields, which C1's null walk deliberately skips.
- C1 now rejects `NaN` and the infinities. `json.load` parses those bare tokens
  by default, and every comparison against `NaN` is false, so a single such
  value disabled the C6 tier ordering, the C6 roofline ceiling, the C7 gate
  check and the C4 curve count at once — a report declaring nothing graded
  `conforming`.
- A bare `(U)` no longer justifies a null. The tag with no sentence after it
  cleared C1 for any field, so four characters pasted beside every null
  satisfied the rule the protocol is built on.
- C8 now warns when `container_digest` is not `<algorithm>:<hex>`. Any registry
  algorithm is accepted and the paths beside it remain unchecked — this tool
  cannot see the machine they name — but `sha256:0` is malformed rather than
  unverifiable, and the digest is what pins the software the rest of the bundle
  came from.

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
