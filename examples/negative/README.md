# Negative corpus: reports that are wrong in exactly one way each

Every other example in this repository shows a correct report passing the checker. This corpus shows the opposite: eight reports, each deliberately wrong in one way, each failing for the stated reason rather than by coincidence. If you implement your own checker against the ASCEP conformance rules, these are the artifacts you validate it against. A checker that passes c3, or that fails c1 with a C4 finding, has a bug -- and you only know that because the intended failure is pinned here.

## How it is built

`baseline.json` is a synthetic fixture -- a single node of 4x A100 80GB running a dense 13B bf16 model at TP=4 with vLLM and a 4096-token chat context -- that grades conforming with zero findings. Every case directory contains that file plus one edit. Because the baseline is clean, any finding the checker reports on a case is attributable to the edit, not to the surrounding report.

This matters because the first draft of this corpus got it wrong. That version mutated a real published report that already graded partial and already tripped C4, C6, C7 and C8. Five of the eight cases "passed" on findings that were present before the mutation, so the corpus proved nothing about the mutation. One edit even removed a pre-existing finding instead of adding one, and the case graded better than the unmutated report. A negative test built on dirty input cannot tell you which layer or which edit produced the failure, so the corpus was rebuilt on a clean synthetic baseline.

## Cases

| Case | Rule | The edit | Grade |
|------|------|----------|-------|
| c1 | C1 | `hardware.node_exclusivity` set to null with no reason | non-conforming |
| c2 | C2 | `model.weight_bytes_tag` deleted | non-conforming |
| c3 | C3 | `serving.gpu_count` set to 8 against TP=4, PP=1 | non-conforming |
| c4 | C4 | `scaling.2.context_tokens` set to null | partial |
| c5 | C5 | `capacity_tiers.measured.binding_constraint` set to null | non-conforming |
| c6 | C6 | `capacity_tiers.measured.max_concurrent_users` set below the sustainable tier | partial |
| c7 | C7 | `run.slo_gates.declared_before_run` set to false | partial |
| c8 | C8 | `reproduction.raw_records_path` set to null | partial |

The mix of grades is deliberate. A case graded partial is publishable and honest despite the defect; a case graded non-conforming is not. The one grade the corpus never expects is conforming -- if a case grades conforming, the checker missed the edit.

Every case also reports OVERSTATED, because each one inherits the baseline's `"conformance": "conforming"` claim while grading worse than that. This is left in on purpose. It is what an overstated report actually looks like in the wild: nobody sets out to claim a grade they have not earned, they change something and do not re-grade. The claim is a field in the file like any other, and the checker's job is to disagree with it.

## What C2 cannot show on its own

There is a subtlety worth reporting precisely, because it is a finding about the protocol itself. Every `provenance` field in the schema can be set to null, which fires C2 -- but the same edit always fires C1 as well. C1 requires a null to carry a sibling `<field>_u_reason` string starting "(U)", and the schemas set `additionalProperties: false` on the objects that hold provenance fields, so the sibling cannot be added. A null provenance is therefore always two violations at once, and no case can isolate C2 by nulling provenance.

Provenance is the one field the protocol does not let you declare unknown, and that is correct design rather than a bug. Everywhere else, "we looked and could not tell" is a publishable state: you measured the GPUs but not the interconnect, and you say so with a null and a reason. Provenance has no such state, because it already has a value for the weakest case -- **(U)**, unmeasured assumption. A number you cannot source is not a number of unknown provenance; it is an assumption, and (U) is what says so. Leaving the field null would let an author sidestep the choice between the four tags rather than make it, which is exactly the laundering C2 exists to stop.

Case c2 therefore uses `model.weight_bytes_tag` instead. It is a plain optional sibling field carrying the provenance tag for the model's byte count, and deleting it fires C2 alone, with no C1 entanglement.

## What these cases do not test

Two limits, stated plainly.

First, this corpus tests the checker's semantic rules, not the JSON Schemas. Every case is deliberately schema-valid, so the grader -- the C-rules -- is what catches the defect. A defect the schema rejects, such as a missing required key or a string where a number belongs, never reaches the C-rules at all. Publishing such a case here would teach the wrong lesson about which layer caught what: you would believe your grader handles a class of defect that your schema validator actually handles, and you might then run the grader without schema validation and miss that class entirely.

Second, C8 is a structural check, not an existence check. It asks whether the report declares a reproduction bundle -- configs, raw records, logs, environment capture, container digest -- not whether the bundle exists at the declared paths. `baseline.json` exploits this: it cites a fixture-bundle directory that was never created, and grades conforming anyway. That is within C8's terms, but it means a conforming grade is evidence of a declared bundle, not of a bundle anyone has opened. Checking the contents is a separate step: `ascep.bench.persist.verify_bundle` re-hashes a bundle against the manifest written alongside it and names any artifact that has been edited or removed. Like the rest of the analytic path it is stdlib-only, so you can run it on a downloaded archive without installing anything belonging to the people who published it.

## Regenerating

Run `python examples/negative/build_corpus.py` to rebuild all eight cases from `baseline.json`. `tests/test_negative_corpus.py` fails if a committed report has drifted from what the builder emits, so edit the builder or the baseline -- not the generated case files -- when the corpus needs to change.
