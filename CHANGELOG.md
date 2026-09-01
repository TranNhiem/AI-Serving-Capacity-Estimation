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

- **Multimodal and reasoning-mode capacity**, as [chapter 9](protocol/09-multimodal-and-reasoning.md)
  plus the declarations and formulas it governs. `input_modalities` and
  `reasoning_modes` are now required and non-nullable in the model layer, because a
  report that cannot say whether it is a VLM, or which of a model's two capacity
  profiles it measured, cannot be interpreted at all. The image and video geometry
  (`image_token_policy` and its per-policy fields, `video_frame_policy`,
  `vision_encoder_params`, `vision_encoder_replicated_per_rank`) is schema-gated on the
  declared modality, so a multimodal declaration cannot be a skeleton. Media
  preprocessing is a **serving-layer** object, `media_preprocessing`, not a per-request
  one: on the engine measured here per-request `mm_processor_kwargs` returns HTTP 400 and
  the sampling rate is fixed at server start, so two clients on one endpoint cannot choose
  differently.
- `ascep.capacity` gains `image_tokens`, `video_frames`, `video_tokens`,
  `media_tokens_per_request` and `vision_encoder_bytes`, validated against six independent
  server measurements (uncapped errors -0.7%, -1.0%, -2.8%; capped +1.6%, -0.5%, -0.4%).
  `Workload` gains `media_tokens_per_request` and `reasoning_tokens_per_request`; media
  tokens are resident for the whole request and are **not** halved in
  `avg_context_tokens`, while reasoning tokens accumulate like output and are. Existing
  workloads are unaffected: both fields default to zero and the published example's
  figures are unchanged.
- Two mandatory run-validity checks, both derived from real failures.
  `media_token_cap_check` catches a preprocessor cap: three clips of 94.8 s, 47.4 s and
  39.1 s all produced ~12.3k tokens under a default pixel budget, so a 2.4x span in input
  moved the measurement by 2%. When measured prompt tokens stop responding to input size,
  the number being measured is the preprocessor's cap, not the workload. The check refuses
  to answer on samples spanning less than 2x, because "we looked and found no cap" and "we
  could not look" size a cluster differently. `media_arrival_check` catches media that
  never arrived: an AV1 corpus that the container's decoder turned into zero frames, with
  every request succeeding as text and no error raised.
- `run.truncation_rate`, the field C4 now requires and no schema provided. Reasoning
  traces expand to fill whatever output budget they are given -- measured at 120
  completion tokens per request with thinking off, and 10,577 / 19,896 / 33,829 at caps of
  8,192 / 24,576 / 57,344, truncating 74.6% / 59.1% / 46.4% of requests at prompts
  averaging only ~2,530 tokens. A throughput figure that averages truncated and completed
  requests together counts tokens no user received.
- **The benchmark driver can send media instead of refusing it.**
  `ascep.bench.workloads.MultimodalJsonlCorpus` replays a LLaVA/ShareGPT-shaped JSONL
  corpus, emitting OpenAI content parts with the media inlined as base64 or referenced by
  URL. It honours the `<image>` / `<video>` marker rather than stripping it: the marker
  count MUST match the record's media references and a missing media file raises at load
  time, naming the line, because silently dropping unreadable media is exactly what
  `media_arrival_check` exists to catch after the GPU hours are spent. Its `media_shape()`
  measures `images_per_request`, `videos_per_request`, the resolution mix and the count of
  records carrying a `reasoning` turn straight off the corpus, and a multimodal run carries
  that dict into its workload manifest -- so the numbers C4 requires come from the corpus
  rather than from someone's recollection. `PromptSource` gains a concrete `render_content`
  hook, defaulting to `render`, so a third party's source keeps working; a text-only source
  still produces a byte-identical `RequestSpec`. `JsonlCorpus` still refuses an unflagged
  media marker: the new class is how a run says "I meant to send it", not a reason to
  soften that refusal.
- Bench configs can select it, through five optional `workload` keys -- `media_root`,
  `image_input_transport`, `media_url_prefix`, `media_max_records` and `prompt_field`, all
  cited to chapter 9. A non-null `media_root` is what selects the multimodal corpus;
  everything else behaves exactly as before. This required splitting the config validator's
  single key table into required and optional halves, because a protocol that grows
  capabilities cannot make every new key a breaking change to every operator's published
  config -- while an unknown key is still rejected by name, since that is how a typo
  becomes a run nobody declared. Four misdeclarations are refused before the first request:
  a `media_root` on a synthetic corpus, a `media_root` that is not a directory, `url`
  transport without a prefix, and a prefix that `base64` transport would silently ignore.

### Changed

- **C4 (context binding) now covers media and reasoning.** A throughput figure MUST
  additionally carry the media shape it was measured at, the `reasoning_mode` the run was
  driven in, and -- for a thinking or mixed workload -- both `max_output_tokens` and the
  measured `truncation_rate`. `0` and `null` are different claims throughout: `0` means
  measured and genuinely absent, `null` means not reported, and conflating them turns an
  unmeasured multimodal workload into a text-only one.
- `tools/migrate_v02_multimodal.py` backfills existing v0.1 documents to the new required
  fields. It is idempotent and has a `--check` mode; it handled all 13 documents in the
  repository without altering any published number.
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
- `capacity_tiers.<tier>.headroom` now states in its description that it is a
  **divisor**, matching `sizing_result.headroom_factor`. The two descriptions
  previously said "factor" and "divisor" for the same quantity, so a generator
  and a validator reading them literally would disagree on the recommended tier
  by the square of the headroom.
- A multimodal corpus reads and base64-encodes each media file **once, at construction**,
  and the request path is a dict lookup. `media_shape()` gained a `media_bytes_resident`
  total so the memory that buys is declared rather than hidden: it is the size of the
  encoded data URLs held in RAM, `0` under `url` transport because nothing is read there,
  and it is the number an operator sizes `media_max_records` against. There is no ceiling
  on it -- asking for the whole corpus holds the whole corpus, which at the FineVision
  corpus's mean media size is about 10 GB for 21,494 records.
- `ascep.capacity.fits` raises `ValueError` when `min_kv_tokens` is asserted
  without `kv_per_token`. It previously returned `True` as soon as the weights
  loaded, so a caller who explicitly asked for a KV floor got a "fits" verdict
  for a configuration nobody had checked.
- **`ascep bench` resolves every path in the config against the config file's own
  directory, outputs included.** This reverses a deliberate, documented decision
  rather than fixing an oversight: chapter 7 stated the two-anchor rule outright and
  argued for it -- `output.*` resolved against the working directory "because they are
  where a particular invocation puts its results rather than part of the declaration
  being replayed". The argument is coherent and it did not survive contact with C8.
  `output.engine_logs_path` resolves against the bundle's parent and MUST sit
  underneath it, so a floating `bundle_dir` floated the C8 check with it. Measured: a
  live H100 run invoked from the repository root against a config in `vlm/` declaring
  `calib/bundle` and `vllm_server.out` loaded its corpus correctly from `vlm/` while
  the bundle was created under the repository root, and C8 refused the run with
  `the declared engine log calib/vllm_server.out is not a file` -- true of the working
  directory, false of the config's. The run was correct in every respect except the
  directory it was launched from, and the only cure was an undocumented `cd` that has
  to be rediscovered by having a run fail. `output.bundle_dir` and `output.report_path`
  now join the config's directory, `output.engine_logs_path` keeps resolving against
  the bundle's parent, which now moves with it, and the C8 refusal names the declared
  string, the path it resolved to, and the rule. The reproduction table still records
  paths as declared, never resolved, so a report stays checkable off the machine that
  produced it. **What this costs:** redirecting a config's outputs by changing
  directory no longer works. An absolute path is still honoured verbatim, which is the
  supported way to send one config's results somewhere else.

### Deprecated

### Removed

### Fixed

- **`ascep bench` computed, and then discarded, the reason for every failed or
  invalid ladder rung.** `grade_rung` and `grade_ladder` build `RungResult.reasons`
  as full sentences with section citations -- which repetition failed, on which gate
  or boundary condition, and why that fails the rung -- but nothing outside the
  grader ever read them: the report row carried only `slo_pass` and `outcome`, and
  the only consumer of `reasons` in the whole package was the line that appends to
  it. Measured consequence on a live H100 calibration ladder: the concurrency-8 rung
  published `slo_pass: true` and `outcome: "failed"` in the same row with no third
  field to reconcile them -- two keys answering different questions (the pooled
  window verdict versus any single failing repetition, section 5) reading as a
  harness contradiction, with the operator's only recourse to re-derive the grading
  by hand from `records.jsonl`, exactly the work `reasons` had already done. Report
  rows now carry `reasons` whenever the outcome is `failed` or `invalid`, and the run
  schema requires it there (an `if`/`then` beside the `itl_population` one), so a
  boundary published with no stated cause cannot validate. The requirement binds only
  rows claiming a non-COMPLETE outcome: every result row under `examples/` is
  `complete` or null, so no published report gains a key or is invalidated.

- **The load generator no longer does blocking file I/O on the request path.**
  `MultimodalJsonlCorpus` read and base64-encoded the media inside `render_content`, which
  the driver calls from its single-threaded asyncio loop between issuing requests. Measured
  on the FineVision corpus that is 1.93 ms per request (1.24 ms read, 0.69 ms encode; p95
  read 1.96 ms, max 16.8 ms): a client-side ceiling of roughly 518 requests per second with
  nothing to do with the server, so a ladder climbing into it would have published the load
  generator's limit as the model's throughput collapse. Worse, each blocking call stalled
  every other in-flight request, so the same milliseconds landed in the ITL and TTFT samples
  of requests that read nothing -- client stalls arriving in the report as server latency,
  which is the one thing the measured tier exists to rule out. No published number is
  affected: no multimodal run has been measured yet.
- An unguessable media MIME type is refused at load, naming the record's line, rather than
  mid-ladder. It joins the missing-file and marker-mismatch checks, which refuse there for
  the same reason: after the GPU hours are spent is the wrong time to learn the corpus was
  unusable.

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
- The three checks that guard the stdlib-only promise — two tests and the
  `capacity.py must import with no third-party deps` step in `ci.yml` — used
  `sys.stdlib_module_names`, which arrived in Python 3.10 while
  `requires-python` is `>=3.9`. The `py3.9` job therefore failed with
  `AttributeError` and the guarantee went unchecked on the oldest supported
  interpreter. All three now detect a third-party import by
  install location (`sysconfig` `purelib`/`platlib`), which works on every
  supported version and states the promise more directly: what is forbidden is
  reaching a *pip-installed* dependency.
- **Goodput was defined as uncomputable.** Chapter 4 §4.2 defined it as counting
  "only requests that met every SLO gate", but every gate in `slo_gates` is a
  window-level p95 or a window-level rate — no individual request meets or fails
  a percentile, so there was no per-request pass/fail to filter on. Goodput is
  now the throughput of a window in which every gate held, undefined where any
  failed, matching how §4.2 and the sustainable tier already used it. A report
  wanting a per-request filter must declare its own thresholds and must not call
  the result goodput.
- Chapter 7 §7 tested throughput collapse on **goodput**, while the declared knob
  is `throughput_collapse_ratio` and the same sentence called it throughput.
  Collapse is a queueing failure independent of the gates, so on the corrected
  goodput definition the old wording made every gate failure indistinguishable
  from a collapse and terminated the ladder before the measured-tier ceiling.
- Chapter 4 §4.2.1 stated one error at two magnitudes in one sentence — "2.33×
  the figure used" and "nearly 2.5× too high". For 400 hidden + 300 visible
  tokens the factor is 700/300 = 2.33, and it is the same 2.33 on both sides
  because per-user demand is the divisor in the throughput floor.
- Chapter 7 §5 required "at least one additional independent repetition" to
  confirm the boundary rung, while §1 and §6 require three at every reported
  operating point — the most load-bearing rung in the campaign appeared to have
  the weakest evidence requirement. The confirmation is now explicitly *in
  addition to* the three, must itself pass, and is justified by the selection
  bias it actually corrects: the boundary is the rung the search stopped at
  because it passed.
- Chapter 7 §6 gave the error-rate denominator as every request "issued or
  admitted". Those populations are identical below saturation and diverge
  exactly under overload, which is the only regime the error rate exists to
  expose: a server rejecting a third of its offered load reported itself
  error-free on the admitted reading. **Issued** is now normative, refusals at
  admission count as failures, and an admitted-only rate may be reported
  alongside but never substituted.
- `ascep bench` no longer accepts `workload.input_tokens` on a real-corpus run.
  The key sized only the synthetic corpus; with `corpus` naming a JSONL file it
  was read, type-checked and range-checked and then used for nothing, so a
  published bench config could carry a prompt-length claim nothing checked -- a
  run whose corpus averaged 722 prompt tokens could declare 4096 and the harness
  would agree in silence. The key now must be `null` whenever the corpus names a
  file (it stays required: an absent key and an explicit `null` say different
  things), a null under the synthetic corpus is refused as that workload's only
  source of prompt length, and the refusal says where the figure belongs
  instead: the measured prompt-token count goes in the workload declaration,
  which the report grades. No published number is affected, because the field
  was inert rather than wrong -- it changed no measurement.
- `ascep bench` no longer discards `workload.output_tokens` when `ignore_eos`
  is false. The two keys encoded only two states -- a fixed output budget with
  EOS suppressed, or no length on the wire at all -- so a config declaring
  `output_tokens: 512, ignore_eos: false` had the number read, type-checked and
  range-checked and then thrown away: every request went out with no output cap,
  the bundle's manifest recorded neither the cap nor its absence, and the number
  survived only in the verbatim copy of the config, the worst of both. Unlike
  the `input_tokens` rule directly above, this one moved measurements. Measured
  on a live H100 serving a 4B VLM against a real image corpus with exactly that
  config: for 90 consecutive seconds the engine logged one running request and
  0.0 prompt tokens/s while generation throughput decayed smoothly from 169 to
  149 tok/s -- a single request that never emitted EOS produced on the order of
  14,000 tokens because nothing capped it, and the concurrency-1 rung's three
  repetitions came out at 9 completions / 126.95 output tok/s, 12 / 113.23, and
  2 / 13.87. That 9x collapse across repetitions was one runaway request eating
  an entire measurement window, and it would have been published as throughput
  variance of the server. The keys now encode three states -- fixed (`true` with
  a positive length), capped (`false` with a positive length: EOS honoured, the
  length a ceiling), and uncapped (`false` with a null length) -- and
  `ignore_eos: true` with a null length is refused, because it asks the server
  to generate until the context limit on every single request. No number
  published in this repository is affected: every workload under `examples/`
  declares `ignore_eos: true`, so none of them was ever in the discarding state.
  The cost fell on live runs of operator configs, and any such run has windows
  whose output throughput is not comparable across its own repetitions.

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
