## What kind of change is this

Pick one. If it is more than one, split it — a spec change buried in an example update is how
a conforming report's numbers change without anyone noticing.

- [ ] New or updated example report under `examples/`
- [ ] Change to the protocol or schemas (`protocol/`, `schemas/`)
- [ ] Change to the tooling (`ascep/`, `tests/`, CI)

---

### Example report

The contribution this project most wants. Checked against the criteria in CONTRIBUTING.md.

- [ ] Five layer declarations present; every `null` carries a `<field>_u_reason` or an
      `unmeasured_assumptions` entry with `impact_if_wrong` and `cost_to_measure`
- [ ] `python examples/<name>/build_report.py` regenerates the committed `report.json`
      byte-for-byte — CI runs `git diff --exit-code` over `examples/`, so a locally-built
      artifact and a committed one that differ fails there, not in review
- [ ] `ascep validate report.json` passes against the schemas
- [ ] `ascep conformance report.json` runs clean, and the level it computes is the one the
      report claims. `partial` is a mergable verdict; a claim above the computed level is not,
      and the checker prints **OVERSTATED** when they differ. That line is what reviewers
      look at first
- [ ] C3/C4: every capacity, KV and throughput figure is bound to its TP width, GPU count and
      context length; nothing presented as topology-independent
- [ ] C5: every capacity figure names its binding constraint
- [ ] C7: SLO gates fixed in the run config before measurement, not fitted to the observed tail
- [ ] `unmeasured_assumptions` names the assumption a reviewer should attack first, and its
      `impact_if_wrong` says what the number becomes if it is — "may affect results" is not an
      impact statement

### Protocol or schema change

- [ ] I have stated which published reports change a number or a verdict under this change.
      **None** is an acceptable answer; **didn't check** is not
- [ ] If any number changes: this is a breaking change, the PR title says so, and the version
      bump follows the rule in CONTRIBUTING.md — anything that alters a conforming report's
      numbers is a major bump, because a silent semantic change makes cross-version
      comparisons invalid
- [ ] Every new MUST has a schema field enforcing it — a MUST that C1 cannot check is prose,
      not a rule
- [ ] The examples are updated in this same PR if their reports are affected; otherwise CI
      regenerates stale artifacts and goes red

### Tooling change

- [ ] `pytest tests/` green on a machine with no accelerator
- [ ] `ruff check` and `ruff format` clean
- [ ] The bare-install job passes: `ascep/capacity.py` stays stdlib-only. The machines where
      capacity questions get asked are often the ones where you cannot `pip install` anything;
      a dependency added here is a dependency added on an air-gapped login node
- [ ] Any formula change ships tests for the happy path, boundary inputs and error paths

---

### Every PR

- [ ] `python tools/check_no_secrets.py` run locally. The site-specific denylist is
      deliberately not committed, so CI's generic patterns are weaker than a local run
- [ ] No hostnames, IPs, credentials, internal paths, job IDs, customer names or pricing
      anywhere in the diff. Scrubbed means absent, not redacted-in-place

<!-- Unticked boxes that stay unticked get asked about. Leave a note where a check does not
apply instead of silently skipping it. -->
