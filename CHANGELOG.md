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

Additive and number-preserving with four exceptions, and they are the first thing to read. Nothing here changes a figure any existing report publishes — the new record fields default to null, and the new subcommand only reads transcripts and writes files you name — but the synthetic-corpus fix under **Fixed** changes what a synthetic `ascep bench` run *sends*, so re-running an existing bench config against the same engine will not reproduce the numbers it produced before. It should not: the old numbers described a prompt roughly eight times longer than the config declared. The second is the connection-pool fix, also under **Fixed**: every ladder rung above 100 offered exactly 100, so no figure from such a rung will reproduce either. The third is the driver de-phasing fix, also under **Fixed**: every measured window on record opened on a phase-locked fleet, so no throughput figure the staircase touched will reproduce — and unlike the first two, the direction of the error is unknowable per rung, so the affected figures are to be described as unreliable rather than conservative, and only re-measurement settles them. The fourth is the reasoning-denominator fix, the first entry under **Fixed**: any report whose workload declares a non-zero `reasoning_tokens_per_request` published a `max_requests_per_s` and a `daily_requests()` inflated by the ratio of tokens generated to tokens delivered, and those two figures move — downward, by up to two orders of magnitude — when recomputed. Text-only reports are untouched, byte for byte.

### Fixed

- **`gpus_required` could not be given a prefill rate, so the one caller whose answer is a purchase order was the one caller that sized on three floors instead of four.** `capacity_at` grew the prefill floor in v0.5.0 and `gpus_required` did not grow the argument to reach it, which reads as an omission and is not one: this function returns the *first* replica count that clears the floors it was handed, so an axis it cannot be told about is not an unpriced axis in a report — it is a smaller cluster, bought. The reranker is the case the floor was added for and the case this broke hardest. Declaring `output_tokens_per_request` of zero makes the throughput floor infinite, so only KV binds, and 500 concurrent users reading 4,000-token prompts size onto **one GPU** whose memory could indeed hold them; handed the input rate that was measured in the same window, the same declaration needs **ten**, and the constraint the report names changes from memory to compute. Both answers claim 500 users, and only one of them can serve them. `prefill_tok_s_per_gpu` is now a trailing keyword scaled per GPU exactly as the other two supplies already are — an assumption, not an identity, and `scaling_efficiency` remains the tool for checking it. Omitted, the answer is the v0.4.0 answer to the byte and C11 flags the report as published on an unpriced axis, which is the protocol's way of saying so and is not the same as quietly costing less.

- **A negative `kv_tokens`, `throughput_tok_s` or `prefill_tok_s` produced a negative capacity rather than an error.** Each divides straight into a floor, so a negative one makes a negative floor, `min()` picks it because it is the smallest, and the result is a `Capacity` announcing that the cluster serves minus four hundred users and is KV-bound. No reader treats that as a capacity; every downstream consumer does — the sizing loop compares it against a user count, the tier table publishes it, and the report names a `binding_constraint` for it. All three are now refused at the door, which is the only place the sign is still attached to the input that carried it. Zero stays legal and is not folded in with them: `kv_pool_bytes` clamps to zero when the weights do not fit, and "this configuration serves nobody, on the KV axis" is a measurement worth publishing rather than an exception worth raising.

- **`capacity_at` divided a token rate that counts reasoning by a request size that does not, so every thinking-mode capacity figure claimed more requests than those tokens can serve — by exactly `(output + reasoning) / output`.** The numerator is `Workload.demand_tok_s()`, which sums `output_tokens_per_request` and `reasoning_tokens_per_request` because both are generated on the GPU, both occupy a decode slot, and the whole point of the reasoning field is that the second stream is real work. The denominator was `output_tokens_per_request` alone. Dividing one by the other prices a request at the fraction of itself the user gets to read, and the error is not subtle at the operating points this project measures: on the thinking-mode figure `demand_tok_s()`'s own docstring cites — 120 visible tokens behind 33,709 reasoning tokens — it is a **281.9× overstatement**, 32,152,725 requests a day where the honest arithmetic gives 114,054. The direction is the one that matters: a cluster that cannot serve a workload looks like it can, and `gpus_required` under-provisions to match. No test caught it, and 982 were passing, because every fixture with a non-zero reasoning count asserted on `max_tokens_per_s`, which was always right — the defect lived entirely in the conversion from tokens to requests. `reasoning_tokens_per_request` defaults to 0, so the new term is `+ 0` on every text workload and no published text report moves; the figures that move are the ones that were never true. This is the third defect of one shape in `ascep/capacity.py` this cycle, after the MoE expert union and the sliding-window asymptote: a formula written when only one thing was being counted, left in place when a second was added, erring toward the larger cluster. Two tests pin it — one that the denominator counts every token the numerator does, one that a zero-reasoning workload keeps the requests/s it published — and the first asserts the 281.9× number as well as the identity, because an identity test alone would still pass if both sides drifted together.

- **A text-only ladder run against a multimodal checkpoint was graded non-conforming for omitting a note about image decoding it never did.** `_check_c1_cpu_cores_note` gated on the model's declared `input_modalities`, so a run that sent nothing but text against a vision-language model failed C1 with a message asserting that the host CPU decodes and patchifies every image — a premise the report's own workload contradicts. Capability and actuality are different gates and only one of them is about the run: the rule now keys on `workload.images_per_request` or `workload.videos_per_request` being positive, through a `_workload_carries_media` predicate that also replaces the two hand-inlined copies of the same test already sitting in C4 and C9, a third copy having been the alternative. Null stays non-media there on purpose — C4 already refuses a null count on a media-capable model, and treating null as media here would report one unanswered question as two findings — and `media_tokens_per_request` is deliberately not consulted, because it prices prompt bytes rather than the host decode the rule charges for. Reports that did carry media are graded exactly as before. The cost of the old behaviour was not the false error itself but what a false error does to a grade: a reader who can see that a rule's stated reason is untrue learns to route around the rule rather than satisfy it.

- **The publication gate has been failing on its own test fixtures since the commit that fixed it, so the last check before a public push has reported two findings on every clean tree.** `tests/test_check_no_secrets.py` builds a corpus that has to trip several rules at once, and two of its lines — a private-range address and a `/home/<user>/` cluster path — were spelt out in full. The scanner reads the repository; the test file is in the repository. So `python tools/check_no_secrets.py`, the command `ci.yml` runs and the command the pull-request template asks a contributor to run before pushing, has exited non-zero since 41f0aad on a tree with nothing wrong with it. The fixture is now assembled from pieces the way the fake PAT beside it already was, and a new test asserts that no generic pattern matches anywhere in that module — because the repair is otherwise invisible: the next fixture written out in full still passes its own test and silently re-reds the gate for everyone. What this prevents is not a leak. It is the habit a permanently red gate teaches, which is to stop reading it, and that habit is the precondition for a real finding being scrolled past.

- **Five protocol chapters described a capacity model two floors out of date, and one of them told the reader that an error was safe in the direction it is not.** Chapter 5's §5.1 KV formula still showed the layer fraction as a single flat term, silently disagreeing with the `effective_layer_frac` fix shipped below it in this same release, and now shows both cases with the sliding-window one normative for `sliding-window` and `hybrid` models. Chapter 5's opening also called TP extrapolation an error "in the pessimistic direction" where chapter 2's replication table and chapter 3's R2 both say the same operation *overstates* the KV pool and therefore the session count — the exact wrong-direction safety claim the sliding-window entry below was written to retire, one chapter away from it. `SPEC.md` listed three floors under C5 where the checker and chapter 8 name four plus the `slo` override, and its conformance paragraph still defined **partial** against C1-C8 though chapters 9 and 10 have added C9-C12; both now defer to chapter 8's C5 row as the normative enumeration so the next floor is added in one place. Chapter 7's pre-publication checklist and chapter 3's floor enumeration were each missing `prefill`. Chapter 4 carried two arithmetic errors in the worked knee example — "three rungs early" for a gap of two, and a single "2.1×" applied to concurrency when 2.1× is the throughput ratio and the concurrency ratio is 4× — which understated its own case by half while teaching the reader to conflate the two units.

- **A bundle could declare that it had been stripped of the evidence behind a figure, and `ascep reduce` would re-derive that figure anyway — from a substituted population, to a more permissive answer, with no warning.** `manifest.json` has carried a top-level `elisions` map since the first large campaign: `"records.jsonl:token_ts"` mapped to the publisher's reason, written because 26.8 million per-token arrival stamps came to 444 MB and could not be published. Nothing read it. Zero consumers in `ascep/` or `tools/`. So the write path re-reduced a bundle whose `token_ts` lists were empty, and `metrics.reduce_window` did exactly what it should do for a harness that never stamped: found no pooled inter-token gaps and fell back to the per-request-mean population. That is the statistic `records.py`'s own docstring says was wrong enough to force a `(U)` on the first published report, because a mean taken inside one request cannot see a stall inside that request, and catching that stall is the entire job of an ITL gate. Against the declared `itl_p95_s` gate of 0.0667 s the substitution moved concurrency 384 from 0.0682 (fail) to 0.0393 (pass) and concurrency 512 from 0.0730 (fail) to 0.0469 (pass) — a rebuilt ladder that reports no ceiling where a ceiling was measured, and therefore a larger capacity claim derived from strictly less evidence. `rebuild_report` now refuses such a bundle outright, naming the elided `file:field` and quoting the manifest's own declared reason, because the honest output of a reduction that cannot see the evidence is a refusal and not a number.

  The same gap made the read path lie in the opposite direction: `ascep reduce --check` reported the lawful loss as `rebuilt report differs from report.json at: run, capacity_tiers, unmeasured_assumptions`, which reads as *the published report is wrong* when in fact the published report holds the better figure and the bundle is the incomplete party. `--check` now performs the rebuild, strips every figure the elision backs from **both** sides before comparing, and reports what it skipped with the manifest's reason beside it — so a pass says "everything this bundle can still reproduce still matches" rather than the stronger thing it has not earned. The strip follows a tainted figure everywhere it appears in a rung row: the value, its `_u_reason` when the rebuild has to explain an absence the measured run never had, and its entry inside the row's `dispersion` spread. It deliberately does not strip the whole `dispersion` block, because the TTFT, end-to-end and throughput spreads beside the tainted one reproduce exactly, and over-skipping is how a partial check quietly becomes a full pass.

  Three smaller decisions inside the fix, each chosen against the failure it prevents. An elision naming a `records.jsonl` field the impact table does not describe **taints anyway** and `--check` refuses to scope it: membership in that table is knowledge of the blast radius, not a precondition for having one, and the alternative exempts the next field somebody elides. A malformed entry in the `elisions` map is dropped by itself rather than voiding the map, because a discarded declaration is an undeclared elision — the same "one bad rule silently drops every rule" shape as the deny-list defect below. And the tainted rebuilt values are never returned, printed or written at any point; they are known to come from the substituted population, and a number that has left the function looks measured.

- **The pre-publication secrets gate had two ways to report a tree clean without reading it, and both were exercised in this repository's own workflow.** `tools/check_no_secrets.py` resolved the site-specific deny-list as `root / ".ascep-denylist"` — relative to whatever tree `--path` named. The deny-list is gitignored and lives in the checkout, so pointing the scanner at a subdirectory, an unpacked bundle, or a `git archive` export staged under `/tmp` found no deny-list beside it and silently fell back to the generic patterns alone. The run still printed `OK: no findings` and exited 0. That is exactly the move an operator makes to scan precisely the bytes about to be published, and it is exactly the move that disarmed every hostname, codename and customer-name rule while reporting success. The deny-list is now resolved from the checkout as well as from the scanned tree, and the success line names which files supplied the rules, because "OK" from 16 generic patterns and "OK" from those plus 23 site terms are different claims and only the operator can tell which one was intended. `tools/redact_bundle.py`, which calls `scan()` once per bundle root, was scanning without the deny-list for the same reason and is fixed by the same change.

  Separately, the file walk was `root.rglob("*")`, which yields nothing when the root is a file, so `--path some/report.md` scanned zero files and pronounced the document clean. A file root now yields itself. Both defects share a shape worth naming: the gate did not fail, it passed — and a gate that passes without looking is worse than no gate, because it is believed.

- **The secrets gate no longer re-derives the same verdict over every published bundle on every run.** It is CPU-bound at roughly 5 MB/s per pattern and it walks the whole tree, so its cost grows with the archive this project exists to accumulate: a full serial scan of the 190 MB tracked tree took 18.5 minutes against the 16 generic patterns alone, and the deny-list nearly triples the pattern count. `--jobs` spreads the files across processes and `--cache` files the content digest of every file that scanned clean under the digest of the exact rule set that cleared it. Measured on that tree: 39 patterns in 4 minutes 57 seconds at 378% CPU, so the gate now runs its full rule set in a quarter of the wall clock the old one needed for a third of the rules. A warm cache over a 29 MB bundle went from 209 s to 0.09 s. The rule-set digest is what keeps this from becoming a third way to pass without checking — add one deny-list term and every entry is void, because "clean" is only ever a claim about a specific set of rules — and new bytes, edited bytes and an unreadable cache file all fall through to a real scan. CI runs without `--cache` so every publication is still checked from scratch.

  Two other candidate optimizations were implemented, measured, and removed: folding all patterns into one alternation as a per-line prefilter ran at 0.9× (CPython's `re` backtracks rather than building a DFA, so N branches cost about what N searches cost), and searching whole file contents instead of line by line ran at 1.0×. Neither is in the tree. The prefilter did surface a real trap on its way out, recorded here because it is easy to reintroduce: most deny-list rules begin with a global `(?i)`, and Python rejects a global inline flag anywhere but the start of an expression, so the alternation never compiled and the "optimization" silently disabled itself while returning correct results.

- **A sliding-window layer now costs what it holds at the context actually served, instead of what it would hold at infinite context.** `kv_bytes_per_token` multiplied by `global_layer_frac` flat. That fraction is `n_global_layers / n_layers`, and it is the **limit** the per-token cost approaches as context grows without bound — not the cost at any context a workload actually runs at. A local layer is capped at `sliding_window_tokens`, so it holds `min(context, window)` tokens: the whole context until the context outgrows the window, and only then anything less. The correct share is `global_layer_frac + (1 - global_layer_frac) * min(context, window) / context`, which is 1.0 while the context fits the window and decays from there. The bare fraction under-reports KV by up to `1 / global_layer_frac`, and it does so worst at the *shortest* contexts, where a model looks most comfortably deployable.

  Found while checking the second known protocol gap and confirmed against an engine rather than argued. The hybrid 26B MoE in the campaign now in flight has 30 layers, 5 of them global, a 1,024-token window, and an average context of 903.35 tokens — shorter than the window, so not one of its 25 local layers is capped and the stack is holding full-length KV everywhere. The declared 1/6 prices that at 20,480 bytes per token; the truth is 122,880, a **6× understatement** in the direction that inflates KV capacity and therefore over-promises concurrent sessions. Chapter 2 said the opposite: that leaving the fraction at its 1.0 default is "pessimistic, so harmless to buyers". Pessimism was never the risk on this axis. vLLM's own accounting settles it. Against the engine-reported 492,000 KV tokens the bare fraction predicts 6,210,713 — 12.6× the engine — while the context-aware fraction predicts 1,035,119, or 2.1×, which is the ordinary workspace, fragmentation and CUDA-graph gap that `calibrate_memory_utilization` exists to absorb. The protocol already names a wrong `global_layer_frac` as the first suspect when that calibration disagrees; the disagreement was 12.6× and the suspect was the formula.

  `effective_layer_frac` is now exported and `kv_bytes_per_token` takes `sliding_window_tokens` and `avg_context_tokens`, which chapter 2 makes normative for any `sliding-window` or `hybrid` report along with a statement of the context the KV figure was computed at — for these models it is not one number. Passing one of the two raises rather than falling back to the asymptote, for the same reason the MoE geometry does: the caller has already said which model they wanted, and an under-reported KV cost is indistinguishable from a correct one once it is a number in a table. Uniform full attention passes neither and gets byte-identical arithmetic, and `global_layer_frac` used bare still returns exactly what it returned before, so no published report moves. `tests/test_attention_families.py` pins the flat-1.0 regime inside the window, the decay outside it, monotonicity, the `global_layer_frac`-to-1.0 band, the limit, the closed form between the regimes, the half-declaration error, and the zero-context division that would otherwise return infinity and read downstream as a KV capacity of zero tokens.

- **An MoE decode roofline is now priced by the experts the whole batch selected, not by one token's share of them, so `roofline_efficiency` is a ratio a reader can act on instead of one too small to trip the rule that reads it.** `roofline_decode_tok_s` computed its per-step byte total with `active_params * dtype_bytes(precision)` as the weight term. `active_params` is a per-token figure — the shared trunk plus `top_k` experts — and a decode step does not process one token, it processes `batch_size` of them, each routed independently to its own experts. The step must read the union. The old formula therefore claimed batching amortizes an MoE's weight read the way it amortizes a dense model's, which is the one thing a sparse model does not do: the union saturates almost immediately. An expert escapes a step only if all `batch_size` tokens miss it, so the expected count is `moe_experts * (1 - (1 - moe_top_k / moe_experts) ** batch_size)`, and at 128 experts and top-8 that is 8.0 experts at batch 1, 111.8 at batch 32, and 128.0 — the entire checkpoint, every step — by batch 192. A production-batch MoE streams close to its **total** parameters per step and gets the KV-amortization benefit of batching with almost none of the weight-amortization benefit.

  The magnitude was measured on the 26B/3.8B MoE ladder, not estimated. At rung 512 (effective batch 476, 903.35 average context, 8 TB/s HBM) the flat-active weight read is 7.65 GB per step; the true union is 51.61 GB, understating weight traffic **6.75×**. Diluted by the 8.8 GB of KV traffic the two formulas share, the published roofline falls from 231,470 to 63,027 output tok/s, a factor of 3.67. That factor lands inverted in the efficiency ratio, because the roofline is its denominator: the same measured 9,984.0 tok/s reads as 0.043 against the old bound and 0.158 against the corrected one. The direction is what makes it dangerous rather than merely imprecise. C6 mandates `roofline_efficiency` and defines a value near or above 1.0 as evidence of measurement error to be investigated and not published — a tripwire that only fires from below. Inflating the denominator 3.67× pushes every MoE efficiency four times further from the wire, so the check most likely to catch a mis-declared `active_params`, an untracked prefix-cache hit or a phase-locked fleet is disarmed on exactly the architecture where `active_params` is hardest to declare correctly. An efficiency of 0.04 also invites the opposite error in a reader: it reads as a server leaving 96% on the table and argues for tuning effort against a bound that was never reachable.

  The protocol text mandated the wrong formula, which is why the code matched it. `protocol/02-model.md` framed `total_params` in `roofline_decode_tok_s` as "the symmetric error" against sizing VRAM from `active_params`, and closed "using active params in compute keeps the ratio honest" — true at batch 1 and inverted above it, since the correct answer converges on total params as the batch grows. `protocol/05-capacity-model.md` carried the same claim as a formula block and the flat assertion "for MoE, `active_params` is the shared trunk plus `top_k` experts, never total params". Both are corrected, both now state the union model and its saturation, and chapter 2 carries the normative rule: a report whose MoE decode roofline was measured above batch 1 MUST pass `total_params`, `moe_experts` and `moe_top_k`. A partial geometry raises rather than falling back, because an understated roofline is indistinguishable from a correct one once it is a number in a table, and the caller who passed two of the three arguments has already said which model they wanted.

  Additive in fact, despite being a formula change in `ascep/capacity.py`. Every one of the seven published examples reports `roofline_decode_tok_s: null` with a `(U)` reason, so no figure on record moves; a dense model passes none of the three new arguments and gets byte-identical arithmetic; `weight_bytes` still uses `total_params` and `roofline_prefill_ttft_s` still uses `active_params` at every batch, since prefill is FLOP-bound and the FLOPs are ~2 per active parameter per token however many experts the union touches. The new `moe_decode_weight_params` is exported so a reader can recompute the weight term independently of the roofline that consumes it. `tests/test_moe_decode_roofline.py` pins the two endpoints the formula must hit exactly — batch 1 equals `active_params`, large batch equals `total_params` — because an implementation can be wrong everywhere in between while satisfying either one alone; it also pins monotonicity and the `active`-to-`total` band, the two degenerate geometries that divide by zero or by nothing (`moe_experts == moe_top_k`, and `total_params == active_params`), the `total_params < active_params` declaration error that would otherwise make the per-expert size negative and shrink the weight read as the batch grew, and the float underflow at extreme batch that must land on `total_params` rather than on nan.

- **A measured window now opens on a fleet that is already running at uniformly random phase, so a rung's output throughput is a measured rate again instead of `floor(W/C)` standing where one should be.** `ascep/bench/driver.py::run_window` ended warm-up with `await asyncio.gather(*(warm() for _ in range(policy.concurrency)))` — a barrier — and stamped `t0 = clock()` on the next line, so every virtual user entered the measured window at the same instant. The workload then guaranteed they stayed together: vLLM `ignore_eos` fixes every response at exactly the declared output length and `think_time_s` is a constant, so every user's request-plus-think cycle has the same length C, the fleet holds its phase for the whole window, and completions arrive in synchronised waves at t0+C, t0+2C, and so on. `reduce_window` attributes a completion to the window it finished in, so a window of length W collected exactly `floor(W/C)` completions per user, N times over — a step function of C, published as a smooth rate, biased downward by as much as `1 - floor(W/C)/(W/C)`, and worst exactly where a ladder is read from: the top rungs, where W and C are comparable.

  It was caught by disbelieving an arithmetic coincidence, again. On the first full re-measurement of the in-progress GB200 26B MoE ladder after the connection-pool fix (TP=1, 120 s windows, 1,170 fixed output tokens, three repetitions per rung, byte-identical at most rungs — which read as determinism confirmed and was in fact the fingerprint), rung 256 and rung 384 both published exactly 7,488.0 output tok/s. A knee was the read, until rung 512 reported 9,867.0: a genuine saturation plateau cannot be exceeded by a heavier offered load, so the flatness was an artefact rather than the engine. The completion histogram said which one. Bucketed at five seconds, every completion of all 384 users landed inside a single bucket, three times, 45 seconds apart — 384 at 45 s, 384 at 90 s, 384 at 135 s, every other bucket zero, the third wave landing outside the 120 s window entirely — and the spread of the first 384 issues was 0.086 s. The fleet was released within 86 milliseconds and never drifted. The ladder's own collapse detector then completed the frame-up: rung 640 reported 5,206.5, and 5,206.5 against 9,867.0 is a ratio of 0.53, below the declared 0.7, so the ladder censored itself — one publication away from a saturation knee the server never reached and a collapse that never happened.

  The defect is dangerous rather than merely wrong because its error has no stable sign, and the published bundle cannot tell you which sign it had. The counting artefact biases a locked figure down, most at rungs whose W/C has a large fractional part. But the lock also biases the figure up, at every rung: a phase-locked fleet presents the engine with a perfectly aligned batch — 32 prefills together, then 32 lock-step decodes no prefill ever interrupts — while a de-phased fleet presents the continuously mixed-phase traffic a real serving fleet actually is, where every decode step pays for the chunked prefill sharing it. Rung 32 is the whole argument in one pair of numbers: the locked window happened to fit almost exactly 8.00 cycles, so the staircase cost it nothing, and the locked run still published 256 completions and 2,496.0 tok/s against the de-phased re-run's 236 and 239 completions and 2,301.0 and 2,330.25 tok/s — about seven percent lower with the lock removed, and that difference is the engine, not the counting. Which term dominates is per rung and unknowable from the bundle, so a correction notice must say the affected figures are unreliable in an unknown direction — not conservative, though the first audit of this very defect called them exactly that. Nor can the counting term be undone by arithmetic after the fact, which was the next thing tried: dividing by W/C instead of flooring predicts about 8,554 and 9,984 output tok/s for the locked ladder's rungs 256 and 384, and the de-phased re-run of those two rungs measures 7,488.0 at 256 — unchanged, in all three repetitions, because that window happens to fit three cycles almost exactly — and 7,644.0, 7,546.5 and 7,595.25 at 384. The predictions are high by 14 and 31 percent, both in the same direction, because they correct one artefact and leave the other in place. Only re-measurement settles a rung. The blast radius is all four published GB200 campaigns: `completions_per_user = requests_per_s * window_s / concurrency` lands on an exact integer at nearly every rung of all four ladders, including `gb200-qwen25-vl-32b-video`, which the connection-pool cap never touched and which locks at every rung except 12 and 16 (7.08 and 6.88 completions per user), where variable video decode cost partially broke the determinism. The two H100 examples were checked rather than assumed, and neither is withdrawn: `moe-26b-h100-tp2` is forecast-only and publishes no measured bundle at all, and `qwen3-vl-4b-h100-image-qa` publishes one whose 180-second windows resolve 48.0, 53.0 and 52.0 completions per user at concurrency 1, falling to 11.05, 10.85 and 10.84 at 128 — non-integer at most rungs and different in every repetition of the same rung, which is what a fleet at mixed phase looks like and what a locked one cannot produce. The reason is in its own records: that workload lets the model stop at EOS rather than fixing the response length, and its 17,680 records carry 452 distinct output lengths with a median of 345, so no two users share a cycle and the fleet scatters itself within the first window. TTFT, ITL and end-to-end percentiles are per-request measures and are unaffected; every SLO gate verdict stands. What moves is throughput — the curve the knee is read from, and therefore the figure an operator sizes hardware from.

  The driver now de-phases the fleet before it opens the window. Between the warm-up barrier and `t0` it estimates the cycle as the median warm-up end-to-end latency plus `think_time_s`, clamped to `window_s`; draws one offset per user from `U[0, cycle)` using a `random.Random` seeded by `blake2b("dephase:{concurrency}:{repetition}")`, so two runs of one config draw identical offsets and a repetition stays reproducible; creates the user tasks immediately, each waiting out its own offset before it starts issuing; then waits one whole estimated cycle and only then stamps `t0`. Every user is therefore already running, at a uniformly random phase, when the window opens. Requests issued during the interval are warm-up — `apply_boundary_rules` already marks anything issued before `t0` out of window, and `warmup_count`, `warmup_s_actual` and the session counters are now snapshotted after the interval so they include it — while a de-phasing request that finishes inside the window still counts as a completion of it under the existing rule that credits a completion to the window it finished in, which is precisely what makes the de-phased completion rate unbiased. The fix was verified on the cluster against the deployed driver before any campaign was re-measured: 24 users, 0.10 s service, a 0.25 s window, and the run recorded a `dephase_s` of 0.1003, a first-issue spread of 0.0903 s, and 2.667 completions per user where a locked fleet produces exactly 2.0. This change is number-changing, not additive: re-running an existing bench config against the same engine will not reproduce the throughput of any rung the lock touched, and the lock touched nearly every rung. It should not — those figures counted waves, not service.

  The fix shipped with a defect of its own, and the suite caught it before the commit did. `t1`, the instant the window closes, was initialised to `math.inf` and stamped only once the main coroutine woke from the de-phasing wait — correct against every adapter that talks to a server, because awaiting a socket hands control back to the event loop, and wrong against one that answers in-process without awaiting anything. Several fakes in this repository's own suite do exactly that. The first virtual user then looped against an infinite bound, appending a record per iteration and never yielding, so the main coroutine was never rescheduled and `t1` never became finite: a single test in `tests/test_cli_bench.py` took the whole suite down at 32.3 GB of resident memory, having issued 539,247 requests in one second. `t0`, `t1` and the drain deadline are now arithmetic on the launch instant, computed before a single user task exists — the de-phasing interval is known before it is waited out, so observing its end was never worth anything — and the user tasks are still created before the wait rather than after it, on purpose, so that the fleet is already mid-request when the window opens instead of meeting a second, cleaner barrier. `tests/test_bench_driver.py` gains an adapter that fabricates its timestamps and returns without awaiting; it raises once it is still being issued requests past the window's declared total budget, which is the only way to catch this, since a coroutine that never suspends also starves any timer `asyncio` could have used to stop it. Reverted against the old release order, that test fails on the 539,247th request rather than hanging.

  Sixth cannot-fire defect in this release, and it survived behind two masks. The first was the connection-pool cap itself: while every rung above 100 offered exactly 100, the high end of every ladder reported the pool's answer, so the one stretch of curve where the staircase is unmistakable — where two concurrencies' floors collide, 256 and 384 both quantising to exactly 768 completions and the identical 7,488.0 tok/s — was already occupied by a louder lie. Removing that lie is what exposed this one: the first ladder run above the old cap is the ladder that caught it. The second mask was the test suite: every test drives a fixture whose cycle is short enough that W/C stays far above the resolution where `floor(W/C)` and W/C visibly diverge, so a completion-count assertion passes identically on locked and de-phased code — green checkmarks standing exactly where the evidence should have been, for the second defect in a row. This one was found the way the cap was, by refusing to believe that two different loads measure exactly alike: 7,488.0, twice.

- **The connection-pool cap underneath the bench client is gone, so the driver's closed loop is once again the only thing that decides how many requests are in flight.** The adapter in `ascep/bench/adapters/openai_compat.py` constructed `httpx.AsyncClient(timeout=..., headers=..., transport=...)` with no `limits=` argument, and httpx answers that silence with `Limits(max_connections=100)` — a second concurrency controller, inside the client, beneath the one the config is supposed to operate. Every ladder rung above 100 offered exactly 100, no matter what the config declared and no matter what the report published, and nothing in the bundle announced the override: the requests that cleared the pool all ran, all finished, and all measured normally on every metric except the one nobody was recording.

  It was caught by an arithmetic coincidence too tidy to be hardware. On the in-progress GB200 26B MoE campaign (the same pre-release checkpoint `examples/moe-26b-h100-tp2` withholds the identifier of), rung 128 and rung 192 both reported exactly 500 completed requests and exactly 4,875.0 output tok/s — byte-identical across two different concurrencies and across all three repetitions of each. That arithmetic is the tell: 4,875 tok/s divided by 1,170 tokens per request times a 120 s window is exactly 500.0 requests, which is what saturation looks like, so the plateau read as a genuine knee. Sampling the engine's own `vllm:num_requests_running` gauge settled it: pinned at exactly 100.0 in two samples 20 s apart while the declared rung was 256. 100 is the httpx default.

  The defect is dangerous rather than merely wrong because it lies in both directions at once. The report published a measured maximum for a load the campaign never generated — flattering, because an operator sizes hardware from a rung that never ran. And the requests the pool held back queued inside the client after `issued_ts` was already stamped, so the client's own backlog was billed to the server as time-to-first-token — the same bug pointing the other way. The blast radius inside this repository is exactly three top rungs: `gb200-gemma4-31b-tp1` (1,250.0 tok/s), `gb200-gemma4-31b-multi-image` (833.3 tok/s) and `gb200-qwen25-vl-32b-multi-image` (700.0 tok/s) each report a measured `max_concurrent_users` of 128 from a rung that never offered more than 100. All three ladders climb 4 to 64 and then 128, so every rung below the cap stands exactly as measured, and only the top rung of each is affected. Correction notices are posted at the top of all three READMEs and the campaigns are queued for re-measurement. `gb200-qwen25-vl-32b-video` tops out at rung 32 and is unaffected.

  The client now passes `limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)`, making the closed loop the sole limiter, and the fix was proven against live hardware before anything was re-measured: 256 concurrent streams against the GB200 server reached a peak of 256 in flight with zero errors, which also rules out a server-side admission limit or a file-descriptor ceiling as a second cap waiting underneath the first. Re-running the same rung 128 that had been pinned at 500 completed and 4,875.0 tok/s now measures 514 and 5,011.5, with the gauge reading 128.0. This change is number-changing, not additive: re-running an existing bench config against the same engine will not reproduce the figures it produced before at any rung above 100. It should not — those figures described an offered load the run never made.

  Fifth cannot-fire defect in this release, and the mechanism of its survival is worth naming. The adapter's tests drive an injected `httpx.MockTransport`, and an injected transport bypasses the connection pool entirely — so a functional concurrency test would have passed on the very bug it exists to catch, with green checkmarks standing exactly where the evidence should have been. The regression test therefore asserts on the real pool's limit rather than on anything a mock can simulate. The others in this family were found by reading and by running the tool; this one was found by disbelieving an arithmetic coincidence.

- **A text corpus now reads the `prompt_field` the config declares, instead of a hardcoded `"messages"` that no protocol section names and no published config carries.** The optional-key table has documented `prompt_field` with a default of `"conversations"` since it was added, and the multimodal reader honours it — but the text branch of `_build_workload` passed `field="messages"` as a literal and never looked at the config at all. Two consequences, and the second is the dangerous one. Every post-training corpus nests its prompt under `conversations`, so a text campaign against a real dataset died at load with "no key 'messages'", and the documented way to fix it did nothing; the only route left was pre-flattening the dataset into a second file, at which point the manifest digest pins the copy rather than the dataset the campaign claims to have served. And a corpus that happens to carry both keys would have been read from `messages` while the report published the operator's declared `prompt_field` — a prompt-token figure measured against text nobody selected, with nothing in the bundle to reveal it. No example in this repository uses a file corpus without `media_root`, which is why the branch went four campaigns without being executed once.

  The two readers now take the first `human` turn from the same function rather than from two copies of the rule, so `prompt_field: conversations` cannot come to mean a different turn in a text campaign than in a media one. The text reader also accepts the turn list itself, not only a dotted path ending at a string, because that is the shape an operator writes and the shape the media reader already wanted. A list of OpenAI content parts is still refused as multimodal rather than mistaken for a conversation with no human turn: the two are distinguished by whether any entry carries a `from` key, so the operator is pointed at the image the corpus contains rather than at the conversation format it does not.

- **`strip_media_placeholders` is reachable from a bench config, so the error message telling an operator to pass it stops naming something they cannot reach.** The text reader refuses a prompt still carrying `<image>` and says to "pass `strip_media_placeholders=True`" — a Python keyword argument with no config key behind it, which before the fix above was unreachable anyway because the text branch could not be entered by any real corpus. It is now a declared optional key, and it is refused alongside `media_root` or the synthetic corpus rather than ignored: the key declares the text-only variant of a media-bearing corpus, so accepting it on a run that sends every image would let a report claim a stripping that never happened.

- **`SyntheticCorpus` pads with common English words instead of random hex strings, so a declared `input_tokens` buys the prefill it says it does.** `ascep bench` ships no tokenizer and sizes synthetic prompts by word count, which is a documented approximation and a defensible one — but only if one word costs the served model about one token, and that is a property of the words rather than of the counting. The filler was `w1a2b3c4d`, which is close to the worst case for a subword tokenizer. Measured against Gemma 4 on a GB200: 256 words of hex filler tokenized to 2,044 tokens, or 7.98 tokens per word. A config declaring `input_tokens: 1500` was sending roughly 12,000, and every quantity computed downstream of the prompt — `avg_context_tokens`, the KV floor the run is compared against, TTFT, `measured_input_output_ratio` and therefore the prefill floor of chapter 5 — described traffic nobody declared. The same 256 words drawn from the new `_FILLER_WORDS` list tokenize to exactly 256, and a 1,500-word prompt to exactly 1,500.

  Nothing in the suite could have caught this, which is the more useful half of the finding. The generator's contract is that the *word* count lands on the target exactly, and it always did; every synthetic test asserts on that number and every one of them passed. The defect lived in the gap between the word count and the token count, and closing it needed a real tokenizer at the other end of a socket. Two tests now pin the property from this side: that the filler comes from the closed vocabulary, and that two rendered prompts diverge inside sixteen words, which is the vLLM prefix-cache block size and therefore the length at which a shared head would start letting one rung answer out of the cache another rung filled.

- **`run.single_point` can now actually be set, so a ladder measured at one context length stops publishing itself as a context curve.** The flag exists to satisfy C4, which asks for three context lengths *or* an honest declaration that there was only one, and bench set it by counting `len(set(...))` over the per-rung `context_tokens`. But `context_tokens` is a per-rung **mean** of measured lengths, so one declared shape produces one distinct float per rung and never fewer: a real GB200 ladder at a single 1,500-token shape reported 2043.65, 2043.94, 2045.28, 2045.46, 2045.48 and 2046.50 — six context lengths by that count, spread across 0.14 percent. The condition `< 3` was therefore false for every campaign bench has ever run, single-shape or not, and the flag was unreachable: present, documented, and never once taken. The visible cost is that a single-shape draft claimed a curve and silenced the one C4 finding written for exactly that campaign. Counting now groups means that lie within 5 percent of each other, two orders of magnitude above the sampling spread above and far below any separation worth interpolating over.

  Third of its kind in this release, and the pattern is worth naming: a rule that cannot fire is one a reader will eventually trust and a maintainer will eventually break without noticing. The other two were caught by reading; this one was caught by running the tool against real hardware and looking at the boolean it emitted.

- **A ladder of fewer than three rungs is refused during config validation instead of after the run.** `run.results` carries `minItems: 3` and there is one row per rung, so a two-rung ladder cannot produce a report that validates. What it did instead was climb both rungs, print their summaries, and then fail draft validation with `is too short` and `this is a defect in bench` — which sends the operator looking for a bug in the report writer rather than at the ladder they declared, after the windows have already been spent. `run.single_point` is not an escape hatch here: it labels a campaign measured at one *context length*, which is the other axis, and bench sets it automatically.

- **`ascep reduce --check` no longer compares the conformance grade, so it can pass on a report that has been graded.** A rebuild is an ungraded draft by construction — `reduce` says so on stderr and the design is right, because the grade belongs to the figures and the figures are new — but `--check` then diffed the published `conformance` against that draft default and reported `conformance` and `conformance_note` as differences. Every published report is graded, so the check failed on every worked example in this repository while every measured figure matched exactly. That is worse than not shipping the check: an operator whose first three runs of a verification command all fail on artifacts the project itself publishes learns to ignore its exit code, and the next failure is a real one. The comparison now folds the grade out the way it already folds out `report_generated_utc`, and the success message names both exclusions rather than claiming a bare match. The fold is a fold, not a skip: everything after the note's opening paragraph is written by the run and stays in the comparison, and a report whose rows no longer reduce from its own records still fails. Both GB200 multi-image bundles now exit 0, which is the first time either has.

  Fourth cannot-fire defect in this release, and the first found by *using* the tool on a real bundle rather than by reading it. There was no CLI-level test for `reduce --check` at all; the two added with the fix are the ones that would have caught it, and they run the command through `main()` against a graded report rather than calling the diff helper directly.

### Added

- **`WindowPolicy.dephase` (default `true`) controls whether the driver scatters fleet entry across one estimated cycle before stamping `t0`, and turning it off has to be asked for by name.** Lock-step entry is the deviation, not a neutral sampling choice — a real serving fleet is mixed-phase traffic, and an aligned one is cheaper to serve at every rung — so the honest configuration is the default and the historical one is opt-in: a config that wants the old behaviour writes `"dephase": false`, and the choice then sits in `run_configs.json` where a reviewer can see it was made, never silently inherited. `WindowRun.dephase_s` is persisted into `run_configs.json` alongside `policy.dephase` and records the estimated cycle the run spread its entry offsets across; `null` means the fleet entered the window in lock-step, which is what lets a reviewer holding a bundle tell whether a plateau in its ladder is the server saturating or `floor(W/C)` stepping down, without re-running anything. `rereduce` defaults an absent `dephase` key to `false`, because a bundle written before this release was locked by definition and re-reducing it must reproduce the arithmetic it was published under. Finally, `ascep bench` now warns per rung when a window resolved fewer than three completions per user: de-phasing removes the systematic bias, but it cannot sharpen what one short window resolves, and a throughput figure resting on two completions per user should not read as measured without a WARNING beside it.

- **`peak_in_flight`: every window now records how many requests were actually concurrent at once, and the report, the log and a warning all read it.** Nothing in a bundle recorded the real in-flight count, which is why a 100-connection pool was indistinguishable from a saturation point — the config showed offered load, the figures showed served load, and the gap between the two lived nowhere. Each record in the window is treated as the half-open interval `[issued_ts, end_ts)` on a sweep line, ends processed before starts at equal timestamps, and a record whose `end_ts` is null is held open to the end because it never finished. The figure is carried on `WindowSummary`, emitted on the per-rung report row, and declared optional in `run.schema.json` so reports published before it existed stay valid — its absence means the run was not checked, never that it passed. Where it does appear it is non-nullable: the sweep always terminates and zero is a measured finding, so a null state nothing can produce would be a branch no test could cover.

  The per-rung log line now carries the peak, and a WARNING fires when it falls below half the declared concurrency. The threshold is half rather than anything tighter because a closed loop with think time leaves users idle between requests, so the in-flight count sits legitimately below the declared value by roughly the duty cycle, and a tighter rule would cry wolf on every healthy run until operators learned to mute it — which is when it would have mattered. The signature the warning names is a peak pinned at the same value across several rungs: the exact shape this defect would have drawn, had this field existed to draw it, is a plateau at 100 on rungs declaring 128, 192 and 256.

- **`examples/gb200-qwen25-vl-32b-video`: the first video campaign, and the first bundle in this repository whose binding constraint is not one of the four floors.** Qwen2.5-VL-32B-Instruct on one GB200 at TP=1, 1,000 H.264 clips of egocentric robot manipulation footage, one clip per request at 16 sampled frames, ladder 2 to 32 with three repetitions. The measured tier is 32 users and 533.33 tok/s; the sustainable tier is **6 users and 250.00 tok/s**, and the gap between that 6 and the roughly 57 requests the 396,016-token KV pool would hold at this context is the finding. Of the four SLO gates only TTFT ever fails, and at the top rung it is at 338 percent of budget while ITL sits at 35 percent, end-to-end at 22 percent, and the error rate at 0.00 everywhere. An operator sizing this deployment from the KV floor would have over-provisioned concurrency by a factor of 9.6.

  Underneath it, a memory consumer the protocol does not model. The first video request was refused with HTTP 400 -- a video item of length 19,552 against a pre-allocated encoder cache of 8,960 -- before the KV cache held anything at all. The encoder cache is sized from `--max-num-batched-tokens`, and raising it to 32,768 to make the workload servable cost 19,568 KV tokens, 4.78 GiB, 4.71 percent of the original pool. Neither the encoder cache nor its cost has a field in any schema. Two smaller gaps came with it: `image_resolution_mix` has no video counterpart, on a workload where frame resolution is the entire cost driver and the corpus is exactly bimodal at 7,346 and 3,130 media tokens; and neither schema can hold a realized sampling-rate distribution, which under a count-based frame policy spans 21.4-fold across one corpus. All three are recorded in the bundle's notes as protocol gaps rather than patched by inventing fields inside an example.

- **`tools/redact_bundle.py`: the third move when an engine log carries an internal path.** `check_no_secrets.py` refuses to publish a bundle whose `engine.log` contains the operator's absolute checkpoint path or a private-range address, and it is right to — both were in the two GB200 multi-image logs. The operator's remaining moves were all bad: publish the leak, delete the log and lose the artifact the manifest pins, or edit it in place and leave the manifest disagreeing with the bytes, which trades a leak for a broken reproduction claim. This tool substitutes named literal strings, re-seals the digests, and records under a top-level `redactions` key the original SHA-256 of each touched file, the replacement text and the occurrence count. The manifest's promise narrows from "these are the bytes the run wrote" to "these are the published bytes, and here is exactly how they differ" — and the second promise is the one a reader outside the operator's network can actually check.

  Four refusals matter more than the substitution. It will not touch a bundle that fails `verify_bundle` going in, because a substitution record over unidentifiable bytes looks like provenance and is none. It will not redact a string that matches a credential pattern: a leaked token is revoked and the artifact regenerated, never renamed, and renaming leaves the live secret in the operator's history while producing a bundle that only looks clean. It refuses a `--replace` that matches nothing, because that is almost always a typo in the very string the operator believed they were removing. And if the scanner still finds anything after the substitution, it restores the originals before refusing — a half-done redaction that is kept is the one outcome worse than the leak. `OLD` is a literal, not a regex, because an operator redacting under time pressure must not have to think about what `.` means.

- **`ascep agent-profile`, and `ascep/agent_profile.py` behind it: the code_agent block stops being a declaration.** 0.5.0 gave `agent_loop` a schema and gave `context_growth_tokens_per_turn` and `kv_residency` a place in the calculator, but left every one of those numbers to be declared — provenance (U) or (I). A declared `turns_per_session` is a guess that multiplies straight through to demand, and a declared `kv_residency` sets the divisor of the KV floor. This turns them into measurements. The input is what a coding agent already writes down: `opencode export <sessionID>` emits `{info, messages[]}` where each assistant message carries its parts, and the parts carry per-call token counts and epoch-millisecond tool clocks. Every agent-loop quantity ASCEP needs is recoverable from that, and the module derives them as (M).

  One structural fact drives the whole module, and getting it wrong is the failure mode it was written to avoid: **a request is a step, not a message.** A tool-calling turn issues one API call, runs its tools, and issues another, and OpenCode records both as `step-finish` parts of the *same* assistant message. Counting messages would report a six-call turn as one request, under-stating `requests_per_session` by exactly the tool-calling factor — which is the quantity being measured. `turns_per_session` counts messages (excluding compaction summaries, which are protocol overhead rather than application turns); `requests_per_session` counts steps.

  Two derivations are worth stating because the arithmetic is where an agent profile goes quietly wrong. `tool_blocked_seconds` is the duration of the **union** of the tool intervals, not their sum: tools run concurrently, and summing overlapping intervals inflates the figure past the session wall clock, pushing `kv_residency` above 1.0 and claiming the engine held more KV than the session existed for. `generating_seconds` is the total assistant-message wall clock **minus** the tool time falling inside those windows, because a message's clock spans both the generation and the tool executions it triggered. `kv_residency` is then `(generating + tool_blocked) / session_seconds`, which is the fraction of the session during which the engine is either generating or holding a prefix it will resume from the moment a tool returns — and the module guarantees it is never below `duty_cycle`, because `capacity_at` raises on that pair and emitting it would produce a workload file that cannot be loaded. `context_growth_tokens_per_turn` averages the positive prompt deltas between consecutive steps and drops any delta spanning a compaction boundary: a compaction shrinks the prompt, and averaging that negative into a growth figure understates growth and under-prices the KV floor.

  `compaction_resume_tokens` is null rather than zero when no session in the sample compacted. Zero would claim a session resumed from an empty prompt; null says the sample did not observe one, and the command says so on stderr because the two read alike in a JSON file.

  The command takes several exports and averages across them, warns on stderr when `turns_per_session`, `requests_per_session`, `input_tokens_per_request` or `kv_residency` spread by more than 2x across the sample — a mean over sessions that were different *kinds* of work describes no session that ran — and with `--into WORKLOAD` merges the measured block into an existing declaration rather than leaving a fragment to be spliced in by hand. The merge does three things the hand-merge forgets: it drops the `_u_reason` sibling of every field that is now measured, so the report stops claiming a gap it has just closed; it re-derives `avg_context_tokens`, `demand_tok_s`, `peak_concurrent_users` and `active_sessions` from the new inputs, because a declaration whose stored derived values no longer follow from its stored inputs still validates and still renders and its GPU count is wrong; and it refuses outright when `avg_context_tokens_tag` is (M), since replacing a direct measurement with the estimator is the one edit that must never happen quietly. `session_max_context_tokens` is not in the transcript — it is a serving choice that governs when the loop compacts — so it is supplied by flag, and a re-profile that omits the flag inherits the value already declared rather than erasing it.

- **`ascep agent-profile --shapes`, and the replay path in the driver that consumes it.** The aggregate a profile produces is a set of means, and a mean cannot be replayed: a workload declaring 8.3 turns per session and 19,525 average context tokens does not tell a load generator what to send first. `--shapes` writes the sequence instead — for every session, every step in order with its prompt size, its generated size, the seconds the client idled after it, and whether the prefix survived into it. The last of those is the one that decides whether the run measures anything real. A coding agent resends its history, so step N's prompt literally begins with step N−1's and the engine prefills only the delta; replay the steps as independent prompts and every request is a cold prefill, which understates capacity by the whole prefix-cache hit rate. Two things break the chain and both are recorded rather than assumed: a compaction, after which the loop resumes from a summary, and OpenCode's in-place pruning of old tool output, which shortens the prompt mid-session. The compaction turn's *own* request is not one of them — it sends the conversation it is about to discard, so it continues the prefix, and treating it as a reset would charge a full prefill of the largest prompt in the session.

  A turn's gaps are derived, because transcripts timestamp messages and not steps. Between two steps of one turn the client is blocked on tools, and the turn's merged tool time is spread evenly across its internal gaps rather than charged to one of them: both hold KV for the same total, but charging it to one replays a burst followed by a stall, which is a harsher test of the scheduler than the capture justifies. After a turn's last step the gap runs to the next turn's arrival and is a human composing the next instruction. The two are distinguished by their turn index rather than by a flag, and they are not interchangeable for KV — this module counts tool time as resident and inter-turn time as not.

- **`ascep/bench/sessions.py`: the replay itself.** `load_shapes` reads a captured file and refuses a bad one at load rather than eight hours into a ladder; `ReplaySessionPlan` turns each step into a request whose prompt has the size and the prefix relationship the capture recorded, and whose generated length is the captured step's rather than one figure declared for the whole run. Prompts are built by appending to the previous step's actual text, so the assertion the report rests on — that the engine's prefix cache hits as it would for a real agent — is a property of the bytes rather than of the token counts. Every render is a pure function of the seed, the session index and the step index: no cursor, no cross-rung state, and session indices that the driver spaces apart per operating point. Sessions are assigned round-robin rather than sampled, so the mix of session lengths over M draws is a function of M alone and two rungs stay comparable; a random draw would give a 128-user rung a different mix than the same rung at 8. Where a target token count cannot be hit exactly it raises instead of approximating, because a prompt a few tokens short moves prefill cost by an amount no report shows.

  One thing here is a performance property that is really a correctness property. The driver builds each prompt inline, on the event loop it is timing with, so the cost of building it lands in every other virtual user's latency sample. Rendering a step by rebuilding every step before it is quadratic in the session's own length, and at the context lengths agent workloads actually reach — measured here at 120,000 prompt tokens — that was half a second of blocking CPU per request. A harness in that state reports a server that slows under concurrency when what slows is the harness. The plan therefore memoises the last rendered step of each in-flight session and grows from the base's known token count in one batch rather than a doubling ramp: the same session went from 5.46 s of CPU to 0.08 s, worst single request from 502 ms to 5 ms. The memo is keyed by the arguments of a pure function and returns exactly what recomputing would, which is what makes it safe — a test asserts that rendering the steps in reverse order produces byte-identical text.

- **`run_window` takes an optional `session_plan`, and `WindowRun` reports sessions started and completed.** Called as before, with a spec callback, the driver is unchanged. Given a plan instead, a virtual user issues one captured session's steps in order, waiting each captured gap, and only then begins another — which is the only thing an agent loop actually changes about offered load, and the reason it cannot be approximated by a think time. Supplying both sources, or neither, is refused rather than resolved, and so is a non-zero think time alongside a plan: the gaps are already in the capture, the two would sum, and the run would report a server less loaded than either declaration describes. Session indices are offset by a digest of the operating point so that two rungs of one ladder never replay the same prompts — without it rung eight finds rung one's strings already in the engine's prefix cache, and measured capacity climbs with concurrency because the later rungs stopped doing prefill. The two session counts exist because a session the window cut short contributes only its early steps, and early steps carry the short prompts: nothing in the records distinguishes a truncated session from a genuinely brief one, so a window shorter than a few session lengths would quietly report a mean context below the capture and a prefill floor to match.

- **`ascep bench` can now run the replay: `workload.replay_sessions`, and §10.8 behind it.** One optional boolean, true when `corpus` names a shapes file rather than a prompt corpus. It is an explicit switch and not something inferred from the file's contents, because the two modes measure different things and a config that fell into the wrong one would publish agent numbers for independent-request traffic with nothing in the report saying so. Three declarations the replay would otherwise ignore are refused by name before the first request — `output_tokens`, a non-zero `think_time_s`, and a `cache_policy` of `unique-prefix` — joining the existing rules that already refuse a declared `input_tokens` against any corpus file and an `ignore_eos` with no length beside it. Those two carry no replay-specific clause on purpose: both fire before the replay is reached, so a second check would be a rule that had never once run, and an unreachable rule is one a reader trusts and a maintainer breaks with no test to notice. Each is a claim the published config makes about traffic the capture actually decides, and the last of them is the one that matters most: a replayed session shares prefixes on purpose, so declaring unique prefixes denies in the config exactly what the run measures. The shapes file is loaded and validated at config-check time for the usual reason: a ladder that discovers a malformed capture in hour eight has already spent the GPU hours.

  `--dry-run` gains the comparison that decides whether the run is worth starting: how many sessions were captured, how many steps they hold, and their median and longest wall clock set against the declared `window_s`. When the longest exceeds the window it says so. Not a refusal — a short window is a legitimate smoke run and the operator may well know it — but it is the one thing about a replay that a report cannot recover from silently. The sessions the clock cuts off contribute only their opening turns, those turns carry the short prompts, and both the measured context and the prefill floor come out below the capture in the flattering direction.

  The workload's manifest is written rather than inherited. Most of the base manifest describes a prompt source or a single declared output length, and a session replay has neither, so `output_basis` takes a fourth value — `captured-per-step`, beside fixed, capped and model-decided — and a `session_plan` block carries the plan's digest, size and draw rule. Two runs whose plan digests match replayed the same traffic, which is the property a reproduction bundle has to make checkable rather than assert. The per-window `sessions_started` and `sessions_completed` land in the bundle beside the boundary, and are absent from a text run's windows rather than zero: a reader scanning for the truncation bias should not have to work out which of the two things a zero means.

- **`RequestRecord.session_id` and `RequestRecord.turn_index`, both null by default.** Without them a file of records from an agent run is indistinguishable from a file of independent requests: the reduction cannot tell that five of them shared a KV prefix and that the gaps between them were tool calls, so it charges a cold prefill every time and reports a duty cycle of 1.0 for a loop that was idle most of the wall clock. Both floors come out wrong in the optimistic direction. `turn_index` counts application turns rather than requests, so the several requests of one tool-calling turn share an index; step order within a turn is recovered from `issued_ts`, which is strictly increasing within a session because the loop is closed. Records written before these fields existed still load, which matters because chapter 7 §8 requires raw records in the reproduction bundle and a schema addition that orphans the bundles already cited in the report defeats the point of keeping them.

### Changed

- **Report assembly moved out of `ascep/bench/run.py` into `ascep/bench/report.py`, and the seam two modules already stood on stopped being spelt with a leading underscore.** `run.py` was 2,151 lines doing two jobs: one decides what to measure — refusing the config, building the workload, climbing the ladder, ruling the window boundaries — and the other turns the kept windows into report JSON. Only the first needs an engine, and the second already had a caller that has none: `rereduce.rebuild_report` re-derives a published report from its bundle and reached `run._build_report` and `run._ladder_policy` through the underscore, deliberately, because a second implementation would drift from the harness the first time either was edited and a rebuilt report that disagreed with the one it was checking would then be indistinguishable from a tampered bundle — which is the single question `ascep reduce --check` exists to answer. So the borrowing is now a declared interface: `run.load_config`, `run.ladder_policy` and `report.build_report` are public, `report.py` is 881 lines that read finished measurements and write JSON and open no socket, and `run.py` is 1,236 lines that decide. Nothing computes differently and no figure moves; the split is dependency-cut rather than judged, so the boundary is where the imports already were rather than where a reader guessed they should be. The new module is stdlib-only for the reason the package initialiser gives for all of the analytic half: re-deriving someone else's published report from their records must not require our HTTP client.

- **The twelve conformance rules are registered once, in one table, instead of three lists that had to agree.** `check()` called the twelve checkers by hand, `_compute_level` compared against a hardcoded C1-C5 tuple for the fatal band, and the findings were sorted by rule id as text. Three places to add C13 is two places to forget it, and both failures are silent in the direction that matters: a rule left out of the walk never fires and reads exactly like a rule that finds nothing, and a fatal rule left out of the band downgrades a non-conforming report to partial. There is now a single `_RULES` tuple carrying each rule's id, its checker and whether it is fatal; the walk iterates it, the fatal set is derived from it, and findings sort by position in it. The sort was not a cosmetic change — as text, C10, C11 and C12 fall between C1 and C2, so the three newest rules, the media, archetype and dispersion ones a reader is least likely to know, were the three most likely to be scrolled past in a long verdict. An unrecognised rule id sorts last rather than raising, because a verdict about someone else's report is still worth printing when one line of it is unexpected. Three tests pin the table: that every checker defined in the module appears in it exactly once, discovered by name so the test cannot be satisfied by editing the test; that the fatal column is exactly the C1-C5 boundary `protocol/SPEC.md` publishes; and that C2 precedes C10. No grade changes — the same twelve rules run, on the same reports, with the same severities.

## [0.5.0] — 2026-09-02

**This release is additive and number-preserving, and that is the first thing a reader upgrading to it should know.** Every new field is optional, and every new computation has a declared-off branch that returns the v0.4.0 expression verbatim: capacity_at called without the new trailing keyword returns a detail dict with exactly the seven keys it always had, avg_context_tokens returns the original expression character-for-character when the growth field is zero, and sessions are divided by `duty_cycle` exactly as before when `kv_residency` is null. Omit the new fields and every number a report publishes is unchanged. Both examples that predate this release — `moe-26b-h100-tp2` and `qwen3-vl-4b-h100-image-qa` — were regraded under 0.5.0 and come out exactly as they did under 0.4.0, with no findings from any of the new rules; the third, `gb200-gemma4-31b-tp1`, is new here and has no earlier grade to preserve, but it draws no findings from the new rules either. There is one honest caveat, and it is in the checker rather than the calculator: C11 lets the conformance checker notice an overstatement it was previously blind to, so a v0.4.0 report whose capacity tiers were read off a ladder measured at a different token mix can move from `conforming` to `partial` without any of its numbers changing. That is the checker gaining eyesight, not the report getting worse, and it is why this release is a minor and not a major.

### Added

- **A fourth capacity floor: prefill.** `Constraint.PREFILL` joins weights, KV, throughput and SLO, and capacity_at takes a new trailing keyword `prefill_tok_s`: when supplied, a fourth floor enters the min, and when omitted the function is the v0.4.0 function byte for byte. The gap it closes is exact. Throughput numbers in every published report are measured on some benchmark token mix, and a capacity estimate divides them against a declared workload with its own mix — yet nothing priced the input side of either. Let rho_w be the declared workload's input:output ratio and rho_m the benchmark rung's measured one. Then, exactly:

      users_prefill / users_throughput = rho_m / rho_w

  This is not an approximation: it was verified in code to six decimal places across three values of rho_m. Two consequences follow exactly rather than heuristically. The prefill floor binds precisely when the declared workload is more input-heavy than the run the throughput number came from — and only then, which is why a rung measured at the declared mix needs no correction at all. And when the floor binds and is ignored, capacity is overstated by exactly rho_w / rho_m. A statement like "this engine serves 128 streams" was always a statement about streams of *this mix*, and 0.4.0 had no machinery to say so.

  On the demand side, Workload.demand_prefill_tok_s mirrors demand_tok_s: the aggregate input tokens per second the workload requires at peak. With a declared `target_tok_s_per_user` it is the active-session count times the generation rate times prompt-to-generated ratio, where the ratio is declared by the workload, not assumed by the code; it raises rather than returning zero when the stream generates nothing, because zero would delete the prefill floor for exactly the embedding and reranking services it exists to price. Without a per-stream rate it falls back to `avg_session_seconds`.

- **The harness now measures the prefill axis.** Two optional per-rung fields, `prefill_tok_s` and `measured_input_output_ratio`, are computed over the same window, the same completed records and the same completion filter as `output_tok_s`. The decision that matters is the denominator: window seconds, not summed TTFT. Summed TTFT gives the rate *while prefilling*, which on an unsaturated rung reads an order of magnitude high and is not comparable with `output_tok_s`; the floor needs the rate the engine actually sustained. The published Qwen3-VL-4B image-QA bundle supplies the evidence that the axis is worth measuring this way: the measured input:output ratio is stable across the saturated part of the ladder — 2.40 at 16 streams, 2.39 at 64, 2.40 at 128 — and reads 2.62 at a single stream, where the run is TTFT-dominated rather than throughput-bound. The mix is a property of the corpus and does not drift with load once the engine is saturated, which is precisely what makes a rung's ratio safe to compare against a declared workload's.

- **`workload.archetypes`: a closed vocabulary that lets a checker cross-examine everything else.** Five values — chat_assistant, image_grounded, video_grounded, code_agent, `other` — unique items, minimum one, optional in v0.5.0 and a candidate for required in v1.0. The reason an archetype is not decoration is that it is the only field that gives the checker a fixed point from which the rest of the workload can be interrogated. Before this field existed, a positive `media_tokens_per_request` on a workload that is actually a plain chat assistant was unremarkable to any rule, because no rule knew which numbers a chat workload should carry; a declared archetype is what turns "every media count is zero on a media workload" from a shrug into an error. The schema gate pairs declaring code_agent with a non-null `agent_loop`, and declaring anything else with a null one, and pairs `"type": "array"` with `contains` — because `contains` is vacuously true on a non-array, and without the pairing a null `archetypes` would silently force a non-null `agent_loop` onto every legacy report.

- **`agent_loop`, and the two pre-archetype estimators a tool-calling loop breaks.** Both estimators are correct for chat and wrong for an agent, and both were defaults rather than declared mistakes. The first is context. `context_growth_tokens_per_turn`, default zero, adds the re-read term g·(N−1)/2 to avg_context_tokens when the loop accumulates — pricing an accumulating transcript with the chat estimator drops that term, and on a many-turn code agent the dropped term exceeds the whole chat estimate. Worked example, illustrative arithmetic: a workload whose chat estimate is 1,200 tokens rises to 20,700 once the growth term is declared. When g is zero the original expression is returned verbatim, so every existing workload is untouched.

  The second is KV residency. `duty_cycle` is the fraction of a session spent *generating*, and through 0.4.0 it was also used as the fraction for which a session holds its KV — crediting every non-generating session with having released its blocks, which no engine does for a session parked on a tool call. `kv_residency`, a fraction in [0, 1] null by default, replaces `duty_cycle` in the divisor when declared and leaves it in place when not. Verified: a code-agent workload gives users_kv of 332.1 at a residency of 0.3 against 99.6 at 1.0 — a factor of exactly 3.3333. capacity_at raises ValueError when `kv_residency` falls below `duty_cycle`, because a residency under the generation fraction claims sessions hand back context they are still generating from.

- **Three new conformance rules, all of which cap at `partial` and never sink a report.** C9 cross-examines the archetype against the workload's own numbers: errors for a media archetype with every media count zero or null, for positive media terms under a non-media archetype, and for code_agent with no `agent_loop`. Its one warning — a media archetype with a declared, measured-untagged `avg_context_tokens` and no `media_tokens_per_request` — stays deliberately quiet when the context is measured (the server's prompt-token total already contains the media expansion and does not split it) and when the context is null (a report declining to publish a forecast context has no under-priced KV floor to warn about). Both exemptions were forced by the published Qwen3-VL report, which is honest on exactly those two points. C10 is scoped to code_agent alone: an error when `avg_context_tokens_tag` is (I) and no note cites `requests_per_session` or `context_growth_tokens_per_turn` — the vocabulary of an accumulating estimator — a harder error for a tag outside M and I, and a warning when `duty_cycle` is below one and `kv_residency` is null.

  C11 is the rule the prefill floor exists to inform: it pairs each measurement-derived capacity tier with the nearest ladder rung by `context_tokens`, and warns at `run.results.{index}.prefill_tok_s` when the declared mix exceeds the rung's measured mix by more than a factor of 1.5. Below the threshold no warning is owed, and that is not leniency — a rung measured at the declared mix was already paying the declared prefill cost while it generated, so its `output_tok_s` embeds that load and the floor built on it is sound; the floor only becomes load-bearing when a number is carried across a mix change. The rung's ratio comes from `measured_input_output_ratio` when declared and is otherwise derived from the rung's own `input_tokens` and `output_tokens`, which every run has always recorded — the fallback that lets C11 grade a report written before the field existed on whether its numbers are actually overstated, rather than warning about every such report on a technicality.

  All three rules sit outside the C1–C5 tuple that forces `non-conforming`: an optional declaration must not be able to sink a report, so their findings cap a grade at `partial`. Promoting them is a v1.0 item, tied to `archetypes` becoming required.

- **Chapter 10, and eleven negative-corpus cases.** The floor gets a normative home rather than an entry in someone else's chapter: chapter 10 is the protocol's chapter on the prefill axis — the demand and supply sides of the floor, the mix identity, and the measurement requirements on `prefill_tok_s`. The negative corpus — the suite of deliberately broken reports the checker must reject — grew by three cases to eleven, one per new rule, so C9, C10 and C11 are each pinned to a report that would pass without them.

### Changed

- **Chapter 9 no longer says there is no fourth floor — because now there is one.** The chapter's discussion of capacity floors asserted that weights, KV and throughput exhaust the list. It was accurate when written and is not now, and the identity above is the price of carrying it forward. Chapter 9 still adds no floor of its own -- the section's real point is that vision costs land inside floors that already exist -- so it is retitled toward that point and now names chapter 10 as where the fourth floor lives. Media tokens gained a second landing spot there too: they are prompt tokens, so they feed the prefill floor as well as the KV floor, and a media workload is input-heavy almost by construction.

### Fixed

- **A service that generates nothing now reports its request rate.** max_requests_per_s derived requests per second from generated tokens alone, which reports 0 req/s for an embedding or reranking endpoint — a service serving requests at full rate that happens to return no tokens, reduced by the arithmetic to serving nothing at all. When nothing is generated, capacity_at now divides by the prompt side instead. Verified: 30,000 prefill tok/s over 1,000 prompt tokens yields 30 req/s where the old path yielded 0.

## [0.4.0] — 2026-09-02

### Added

- **`ascep conformance report.json --raise` writes the computed level into the
  report.** Every draft the harness emits ends its note with "`ascep conformance`
  is the command that may raise the claim", and chapter 7 §10 says the same in
  normative voice, but nothing fulfilled the promise: the checker printed a grade
  to the terminal and the file went on claiming `non-conforming` forever. A grade
  that exists only in a terminal is not part of the artifact, and the artifact is
  what circulates. With the flag, an understated claim is replaced by the computed
  level and the draft paragraph in `conformance_note` gives way to one saying the
  grade was computed rather than assumed; any caveat the harness appended after
  that paragraph — the censoring sentence, the lower-bound caveat, the cache
  caveat — is carried across untouched, since dropping one would turn a declared
  lower bound into a bare maximum. Without the flag nothing is ever written, so
  the checker stays safe to run on someone else's report. It never lowers a claim:
  a report claiming more than the checks support keeps its OVERSTATED line and its
  bytes, because a flag that quietly rewrote an overstatement into an accurate
  claim would erase the evidence that anyone overstated.

- **`examples/qwen3-vl-4b-h100-image-qa` — the first example this project
  measured itself.** Qwen3-VL-4B-Instruct, dense, bf16, one H100 SXM at TP=1,
  vLLM 0.11.0, answering open-ended visual questions over a 2,000-record
  FineVision QA split with one base64-inline image per request. The ladder
  climbed 1 → 128 streams in three 180-second repetitions per rung and left
  17,680 request records behind it. Measured capacity is 128 streams at
  2,668.1 tok/s; sustainable is 64 at 2,564.8 tok/s; both name `slo` as the
  binding floor, TTFT p95 having gone from 2.310 s at 64 streams to 9.598 s at
  128 against a 4.0 s gate.

  It is three firsts at once, and each is a different kind of evidence. It is
  the first **multimodal** report, so chapter 9 now has a worked example rather
  than only a specification. It is the first **bundle-backed** one: there is no
  `build_report.py`, because nothing was transcribed — the bundle *is* the
  artifact, and every latency in the report recomputes from
  `bundle/records.jsonl` under a sha256 manifest. And it is the first report
  produced end to end by `ascep bench`, which is how the measured-tier defect
  above was found at all: the repository's own harness wrote a report its own
  conformance checker rejected. It also closes the gap 0.1.0 recorded as "a
  dense-model report on the same hardware is the nearest gap".

  It grades `partial`, for three warnings and no errors. `container_digest` is
  null because the engine ran from a scheduler-managed environment rather than a
  pinned image; the `theoretical` and `recommended` tiers are absent because the
  roofline belongs to `ascep size` and a headroom factor is a policy nobody has
  declared for this deployment. Prefix caching is on in the engine and
  deliberately defeated by the workload's unique-prefix cache policy, since the
  rungs climb in order and a shared prefix would let every higher rung answer
  partly out of a cache the rung below it filled.

### Fixed

- **The measured tier was SLO-gated, which erased the distinction between the
  measured and sustainable tiers that chapter 5 §5.5 exists to draw.** Tier
  selection in `ascep/bench/run.py` kept only rungs graded `COMPLETE`, and a rung
  is graded `COMPLETE` only when every declared SLO gate held — so the "best
  observed, SLO ignored" engine ceiling was in fact gated, and on any ladder
  containing a failing rung the measured tier silently collapsed onto the
  sustainable tier. A reader comparing the two headline numbers would see
  identical figures and reasonably conclude the engine stops where the SLO stops.
  Worse, on a ladder where no rung passed all its gates, the report published no
  measured tier at all — "no rung completed its declared repetitions" — even
  though the harness had plainly measured a ceiling. Chapter 5 §5.5 had the rule
  right, and the conformance checker had it right too (C7 flags a sustainable
  tier that equals measured despite a failing operating point at or below the
  average context); the harness alone was the outlier among the three components.

  The repository's own first self-measured example exposed it: a Qwen3-VL-4B
  image-QA ladder on one H100, where rungs 1 through 64 passed and rung 128
  failed its TTFT gate (9.598 s p95 against a declared 4.0 s gate). Under the
  previous release both tiers would have read 64 streams / 2,564.8 tok/s. They
  now read 128 streams / 2,668.1 tok/s measured, binding constraint `slo`,
  against 64 streams / 2,564.8 tok/s sustainable — the engine ceiling versus
  what users can rely on, which is the entire point of having two tiers.

  The fix selects the measured tier from operating points — rungs graded
  `COMPLETE` or `FAILED` — per chapter 7's outcome vocabulary, in which a failed
  rung is a real negative boundary, while `INVALID` and `ABORTED` still claim no
  operating point and stay out. It also adds `_observed_constraint`: the old
  `_boundary_constraint` searched only *above* a given rung for the binding
  floor, so on the top rung of a ladder that failed there the headline tier
  carried a null constraint — a report saying "the ceiling is 128 streams"
  while declining to say what stopped it. The new helper reads the floor
  from the failed rung itself (`throughput` if it completed nothing, else `slo`)
  and falls back to the above-only search otherwise.

  This alters the numbers in already-conforming reports — every published ladder
  with a failing rung will now show a higher measured tier and a populated
  binding constraint — and per the project's versioning rule that is precisely
  why this release is 0.4.0 rather than 0.3.1.

- **`ASCEP_VERSION` stayed at 0.2.0 through the 0.3.0 release, so reports could
  not cite the protocol they were produced under.** 0.3.0 was cut precisely
  because the ITL fix alters the numbers a conforming report publishes; its own
  commit message invokes the versioning rule. It bumped `__version__` and left
  the protocol constant behind, and the protocol constant is the one thing a
  report carries. For the length of that release a report reduced the corrected
  way was indistinguishable from one reduced the broken way, by exactly the
  field the spec designates to tell them apart — "cross-version comparison of
  numbers requires the majors to match", against a version that did not move.
  The two constants are now bumped together, as 0.2.0's release did and 0.3.0's
  did not, and a test requires the major and minor to agree. Nothing enforced
  that before, which is why it slipped in silence; the patch level may still run
  ahead, since a CLI fix that touches no number is a package release and not a
  protocol one.
- **The reproduction bundle pinned the whole serving stack and not the harness.**
  `environment.json` records fourteen packages, from vLLM down to the httpx that
  sits inside every TTFT this tool measures, and said nothing about `ascep`
  itself. Adding it to that list would not have worked: every cluster run in this
  repository is launched from a checkout on `PYTHONPATH` with nothing installed,
  where `importlib.metadata` reports the one distribution that provably ran as
  absent. The capture now writes "ascep_package_version" and
  "ascep_protocol_version" from the imported module, so they cannot disagree with
  what the process is executing. A reader of an old bundle can now answer the
  first question an old measurement raises — whether it predates the fix to the
  code that computed it.
- **`ascep bench` could not emit a draft that `ascep conformance` would grade
  above `non-conforming`.** Three nulls the harness chose itself, none of them a
  property of the run, put five findings in every draft it has ever written:
  `capacity_tiers.measured.binding_constraint` and its `sustainable` twin
  (C5 errors, a capacity number with no named floor), and the `provenance` tags
  on `capacity_tiers.theoretical`, `capacity_tiers.recommended` and
  `sizing_result` (C1 errors with no lawful answer, since the schema defines no
  "provenance_u_reason" to hold a sibling reason). The operator who ran the
  benchmark could not fix any of them. All three are now the harness's to say:
  a filled tier names the floor the ladder actually hit, and a row the harness
  leaves empty carries the `U` tag that means it states nothing — the same
  convention `examples/moe-26b-h100-tp2` already used.
- **The floor a ladder hit is now named rather than withheld on principle.** The
  harness argued that latency cannot decide which of the three floors binds, and
  so named none. Chapter 5's own table settles it: `slo` overrides the
  constraint label when gates fail. The first rung above a filled tier that
  failed its declared gates is the observed boundary, and it labels the tier
  `slo`, or `throughput` when that rung delivered nothing at all — a collapse,
  not a missed latency gate. A ladder exhausted without any failing rung still
  names nothing and still fails C5, which is the correct grade: its figure is a
  lower bound, and the remedy is to declare more rungs, not to invent a label.
  The register entry that recorded the constraint as unbound now records what
  remains true — that the weights and KV floors were never evaluated, so a floor
  lower than the observed one would not have been seen.
- **A single-context campaign now declares itself one.** The harness knew it had
  measured one context length and left `run.single_point` at its default, so C4
  asked the author to state a limit the harness could already see. Setting it
  does not raise the grade; it stops an unlabelled point from reading as a curve.

### Known gaps

- **`unmeasured_assumptions[].field` is specified as a dotted path, and the
  harness does not always write one.** `ascep bench` emits readable designators
  instead — a path with an explanatory parenthetical, a glob across the tiers, a
  pair of sibling fields named together, and in one case a caveat about
  percentiles that are *published* rather than null, which is not a field at
  all. The example test that guards the register against listing a field the
  report went on to measure can only resolve entries that are path-shaped, so it
  now skips the rest and says so. That is the right trade for a check whose
  purpose is catching an understated report — the alternative reading turns it
  into pressure to delete honest caveats — but it does mean those entries are
  unchecked. Either the harness should write bare paths and move the prose into
  `impact_if_wrong`, or the schema should admit a second entry shape for a
  caveat that backs no single field. Deciding which is a v0.5 question, and it
  changes a report's bytes, so it is not being made quietly here.

## [0.3.0] — 2026-09-02

One defect is fixed that changes the numbers a conforming report publishes: the
harness was reducing the wrong distribution and calling it ITL, and correcting
it moves published `itl_p95_s` values and therefore moves which rung a ladder
reports as sustainable. Under the versioning rule a change that alters a
conforming report's numbers is a breaking bump, so this is 0.3.0. It is cut
immediately after the fix rather than batched, because every rung graded on the
old reduction is grading the transport and not the decoder, and no published
report should spend another release cycle on the wrong population.

### Added

- **Section 4.1.1, "Chunk gaps are not token gaps", the normative rules for the
  two ITL populations.** A pooled-gaps ITL percentile now requires
  `tokens_per_stream_chunk` per rung. Above 1.05 tokens per chunk the pooled
  population MUST NOT be published as ITL and `itl_population` MUST read
  `per-request-mean`, because each pooled gap then spans several tokens and the
  percentile measures the transport, not the decoder. The observed chunk gaps
  MUST still be published as `stream_chunk_gap_p50_s` / `stream_chunk_gap_p95_s`
  whenever per-token stamps exist: a smoothness SLO written against what the
  client actually receives is legitimate, it is simply not ITL. The factor is
  load-dependent -- 1.00 at concurrency 1 and 6.65 at concurrency 128 in one
  measured run -- so it is declared per rung, never once per run, where a
  single run-level figure would average the one regime where pooling is honest
  into the regime where it is not. See
  [§4.1.1](protocol/04-measurement.md).
- **Three fields on `run.schema.json`'s results rows:** `tokens_per_stream_chunk`,
  `stream_chunk_gap_p50_s` and `stream_chunk_gap_p95_s`, each with a
  `*_u_reason` companion, so a rung whose transport was not instrumented says so
  under C1 rather than falling silent.

### Changed

- **Breaking: a results row declaring `itl_population: "pooled-gaps"` now
  requires `tokens_per_stream_chunk`.** Null with a `(U)` reason is accepted;
  silence is not, because an unsupported pooled percentile is indistinguishable
  from a supported one, and the whole point of the population field is lost if
  the reader cannot tell which kind they are looking at. Rows on the per-request
  population owe nothing: their denominator is the server's own token count,
  which no transport batching can inflate. Existing pooled-gaps reports must add
  the field; a report that does not now grades non-conforming.
- **The rendered report table now says which ITL it is showing.** The ITL cell
  carries its population in parentheses, and a "tok per chunk; chunk gap p50 /
  p95" column appears whenever any row declares those fields. The column is
  omitted entirely when no row does, so a report from a harness that never
  measured the transport does not grow a column of identical `(U)` cells.
- `examples/negative/baseline.json` and the eight regenerated cases now declare
  `tokens_per_stream_chunk` of 1.0 and chunk-gap figures equal to their ITL
  figures -- what a clean per-token stream looks like -- so the fixtures read
  as worked examples of the new rule rather than as rows that predate it.

### Fixed

- **The harness published the pooled gaps between streamed chunks as the ITL
  percentile, and under load those are not the decoder's gaps.** `reduce_window`
  computed ITL percentiles from the pooled gaps between streamed content chunks
  and labelled the result `pooled-gaps`, which section 4.1 defines as one sample
  per decode step. The two are the same population only while the server emits
  one token per SSE delta; under load vLLM folds several decode steps into one
  delta, so each "gap" spans several tokens and the percentile measures the
  transport. The harness already knew: its OpenAI-compatible adapter appends an
  "itl-granularity: N chunks for M tokens" note to the record whenever the two
  disagree, and 99.5% of the 16,720 records in the H100 rehearsal carried one.
  It noted the discrepancy per record and then reduced as if it had not
  happened. Measured on that rehearsal (Qwen3-VL-4B, 1x H100, vLLM 0.11.0,
  closed-loop ladder, gates ttft_p95 4.0 s / itl_p95 0.05 s / e2e_p95 60 s /
  error rate 1.0%) -- tokens per streamed chunk, pooled p95, then per-request
  p95, by concurrency: 1 -> 1.00, 0.0060 s, 0.0060 s; 2 -> 1.03, 0.0063 s,
  0.0063 s; 4 -> 1.08, 0.0066 s, 0.0068 s; 8 -> 1.20, 0.0072 s, 0.0076 s; 16 ->
  1.43, 0.0288 s, 0.0092 s; 32 -> 1.95, 0.1100 s, 0.0130 s; 64 -> 3.54, 0.2374
  s, 0.0196 s; 128 -> 6.65, 0.2953 s, 0.0247 s. On the pooled population every
  rung above 16 breaches the ITL gate and sustainable capacity is concurrency 16
  at 1,180 output tok/s, ITL-bound. On the per-request population every rung
  passes the ITL gate and 128 fails on TTFT instead (9.670 s against the 4.0 s
  gate), so sustainable capacity is concurrency 64 at 2,503 output tok/s,
  TTFT-bound. That is 2.1x on capacity and a different named binding constraint,
  from a ladder that was correct in every other respect. Throughput across the
  ladder runs 93, 184, 354, 660, 1,180, 1,876, 2,503, 2,643 tok/s, so the knee
  really is at 64 and the pooled reading placed the boundary three rungs early.
  Two facts make the fix trustworthy rather than convenient: below the threshold
  the two populations agree to three decimals (0.0060 s against 0.0060 s at
  concurrency 1), and just above it the switch is conservative -- at concurrency
  4 and 8 the per-request figure is the higher of the two -- so the rule does
  not buy capacity where coalescing is mild. The coalescing is server-side
  rather than assumed to be: a client that parses SSE can see that one delta
  carried the text of several tokens, and no transport layer can synthesise
  token text.
- **The chunk count was off by one, and on short replies the error trips the
  population switch by itself.** The adapter files the first content chunk's
  timestamp in `first_token_ts` and only later ones in `token_ts`, so the count
  must be `len(token_ts) + 1`. Counting `token_ts` alone reports tokens /
  (tokens - 1) for a stream that is exactly one token per chunk: 1.02 on a
  50-token reply and 1.33 on a four-token one. The second trips any threshold
  worth setting, so a workload of short answers -- a VLM classification corpus
  -- would have had its ITL population switched by where the harness files one
  timestamp rather than by anything the server did.
- **The reproduction bundle's environment capture named no library
  version.** On a real H100 run `environment.json` was 212 bytes -- driver,
  GPU model, Python version, platform -- so a bundle whose entire purpose is
  to pin the software answered "CPython 3.10.13 on Linux". The one framework
  version anywhere in the report lived in the serving declaration, typed by
  the operator, with nothing in the bundle corroborating it, and C8 calls the
  artifact the *environment capture*, so a reader sees the file and assumes
  the environment was captured. It now records a `packages` mapping read from
  `importlib.metadata` -- never by importing, since importing torch costs
  seconds and gigabytes to read back a string and fails outright on a
  client-only host. Both sides of the measurement are pinned: the engine that
  generates the tokens and the `httpx` stack that times them. A distribution
  that is not installed is **absent** from the mapping rather than null,
  because a null here would carry the `(U)` "we looked and could not tell"
  when the truth is that it is not there. Its "packages_source" sibling -- a
  bundle-artifact key, not a declarable field -- records that the
  versions describe the harness process, since client and server need not be
  the same environment and a capture that does not say which side it saw will
  be trusted for both.
- **`container_digest` was refused with a reason that asserted the wrong
  cause.** A null read "(U) the reproduction bundle did not record this",
  which describes a bundle that failed to write something. The usual cause is
  that there was no container to take a digest of -- a bare-metal or conda
  run -- and bench cannot tell the two apart from a null in its config, so it
  now asserts no cause and states the consequence instead: with no digest,
  nothing in the bundle pins the software except the environment capture
  beside it.

## [0.2.0] — 2026-09-02

Two conformance gates changed and one declared field was renamed, so under the
versioning rule this is a breaking release. It is cut before the first
multimodal worked example is measured, so that the example's declarations, the
schemas they validate against and the version stamped into its hashed
reproduction bundle all name the same protocol.

### Added

- **Declared-value notes (`notes`).** Every declaration layer -- hardware,
  model, serving, workload, and the nested `media_preprocessing` block -- gains
  an optional `notes` object, keyed by field name, recording why a non-null
  declared value is what it is: the `mm_processor_cache_gb` of 0 set on purpose
  so preprocessing cost stays inside the measurement, the `cpu_cores` that is a
  cluster allocation and not the node's. The `(U)` mechanism could
  only ever justify a null, and authors were already abusing `(U)` reason fields
  to carry value-justifications; a convention people have to abuse is a missing
  feature. Three rules keep notes honest: a note never substitutes for a value
  (a null still needs its `(U)` sentence, and a note beside a null does not
  satisfy C1), a note is not evidence and upgrades no tier, and a note lives in
  one named object per layer so its presence is visible without diffing against
  the schema. See [§9.10](protocol/09-multimodal-and-reasoning.md).
- **Breaking for existing multimodal reports: C1 now requires a note on the
  hardware layer's `cpu_cores` whenever `input_modalities` contains image or
  video.** Under a media workload the host CPU decodes and patchifies every
  image, so the core count is a capacity input rather than an inventory fact.
  Measured: across 6,587 one-second samples the serving process peaked at 11.53
  of its 12 allocated cores on a node Slurm reports as having 112, and in the
  samples where it was above 10 the GPU averaged 59% -- pinned host, unsaturated
  accelerator, a ceiling no reader could attribute. An
  existing multimodal report without the note now grades non-conforming. Nothing
  published here is affected: the only example report is text-only.
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

- **Breaking rename in `media_preprocessing`: the pixel-budget field is now
  `image_pixel_budget_px`.** The 0.1.0 name, "image_longest_edge_px", described
  something the field is not, and the two quantities differ by orders of
  magnitude: on the H100 VLM run that prompted this the correct value was
  16,777,216 (4096 x 4096, a total-pixel cap), while a reader following the old
  name writes 4096 -- which, as a pixel budget, is a 64 x 64 image, roughly
  4,096 times too small -- into the one field chapter 9 says binds silently.
  Two days after 0.1.0, before the first multimodal worked example bakes the
  name in, is the last cheap moment to fix it. The old name is not aliased and
  is refused outright: a schema that accepts both names accepts both meanings.
  Reports declaring media preprocessing under 0.1.0 must rename the field; the
  schema description and chapter 9 now name the quantity explicitly. No example
  under `examples/` is affected -- the published report is text-only and
  declares no `media_preprocessing` block.
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

- **C1 charged every report the harness writes with eight errors, in the one
  block a reviewer reads first.** A section-7 `unmeasured_assumptions` entry
  whose `value_used` is null says no substitute was plugged in -- the field was
  simply left unmeasured -- and the entry around it already names the field, the
  consequence if it is wrong and the cost of closing it, which is more than a
  `(U)` sentence carries. C1 saw only a null and demanded a justification, and
  both remedies it prescribes are worthless here: the sibling key is rejected by
  the entry's own `additionalProperties: false`, and the section-7 remedy only
  works spelled as the bare leaf "value_used", which makes the register declare
  itself an unmeasured assumption and clears its own null on the way past.
  Either way the block gains an entry recording nothing about the deployment --
  and the third option, needing no ceremony at all, is to invent a value, which
  is the fabrication the register exists to prevent. A null `value_used` is now
  exempt, scoped to `unmeasured_assumptions.<index>` so the name alone buys
  nothing anywhere else in a report. Numbers are unchanged; a report that graded
  non-conforming only on these findings now grades on its merits.
- **Four media-preprocessing fields promised a `(U)` reason with nowhere to
  write one.** `video_sampling_fps`, `video_max_frames`, `image_pixel_budget_px`
  and `mm_processor_cache_gb` are nullable inside a closed object that declared
  no justification siblings, so C1's demand was unsatisfiable and the author's
  only exit was the whole-report register. The siblings now exist. A structural
  test sweeps all six schemas for the shape rather than the instance -- a
  nullable property in a closed object with no local way to justify the null --
  and the five survivors are exempt for a stated reason: the section-7
  `value_used` above, and four `provenance` fields where "U" is already a value
  in the enum, so the tag is written in band.
- **`examples/chatbot-10k-dau/build_workload.py` pinned the protocol version
  in its own source.** It now imports `ASCEP_VERSION`, so the declaration it
  regenerates names the release it was actually produced under. A builder that
  hardcodes the version keeps stamping the release it was written under, and
  the version-consistency test does not look inside `examples/`, so the drift
  would have been silent and the artifact would have claimed a protocol the
  code no longer implements. The regenerated `workload.json` differs only in
  that line; every forecast and derived figure is unchanged.
- **Removed an orphan justification slot, "floor_crossover_u_reason", that
  justified a field no schema declares.** The property it appeared to serve is
  `floor_crossover_context_tokens`, whose real sibling was already present, so
  the orphan could only ever be written by an author who then believed a null
  was justified when the checker had never looked. A second structural test now
  fails on any justification property whose target does not exist.
- **`ascep bench` measured the context length of every rung and never published
  it.** The identifier `context_tokens` did not appear anywhere in
  `ascep/bench/`, though the comment above the row's token counts already said
  "C4 binds every throughput figure in this row to the context length beside
  it". Because the key was absent rather than null, C1 never fired, and the
  visible symptom was C4 reporting the opposite of the truth: on a real
  three-rung H100 ladder with zero errors it warned that *zero* distinct context
  lengths had been measured, when one had, at about 1316 tokens per request. The
  Markdown renderer meanwhile printed "not reported context" beside both halves
  of the sum it was holding, and C7's chapter-5.5 envelope exemption -- keyed on
  `row["context_tokens"] <= envelope` -- could never apply to a report this
  harness wrote. Rows now carry `context_tokens` as the mean of the per-request
  `input_tokens + output_tokens` the server accounted. It is taken per record,
  not as `mean(input) + mean(output)`: each of those means is over whichever
  records carried that count, so their sum can be a context no request ever had.

- **The rendered Markdown report showed a rung that did not complete as a clean
  pass.** The benchmark table's SLO column was driven by `slo_pass` alone, which
  grades only the pooled sustained window, so a rung whose `outcome` was `failed`
  -- any single counted repetition having failed, section 5 -- printed `pass` and
  nothing else. The document a reviewer actually reads was thinner than the JSON
  it came from: it could certify a capacity boundary grounded in an unfinished
  rung without ever saying so, let alone why. The SLO cell now carries the
  completion outcome beside the window verdict whenever the two can disagree, and
  a "Rungs that did not complete" block after the gates line prints each such
  rung's recorded reason verbatim, keyed by concurrency. Reports whose rungs all
  completed render byte-identically to before: no heading, no placeholder, not
  even a blank line.

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
