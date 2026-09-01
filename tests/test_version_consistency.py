"""One version number, declared in five places, which must never disagree.

Reports record the protocol version they were produced under, and CONTRIBUTING makes that
load-bearing: a change that alters a conforming report's numbers is a major bump, so a reader
comparing two reports relies on the version to tell them whether the comparison is valid. If
`ascep version` says 0.2.0 while the wheel metadata says 0.1.0, every report the tool stamps
carries a number that does not identify the code that produced it, and nothing downstream can
detect that.

Drift here is silent and easy: bumping `__init__.py` and forgetting `pyproject.toml` breaks
nothing that any other test looks at.

Deliberately regex rather than `tomllib`/`yaml`: `tomllib` is 3.11+ and the CI matrix includes
3.9, and pulling in pyyaml would make a version check depend on an optional extra. Both files
declare the version on one unambiguous top-level line, so a regex is sufficient and keeps this
runnable on a bare interpreter.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from ascep import ASCEP_VERSION, __version__

ROOT = pathlib.Path(__file__).parent.parent
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def _first(pattern: str, path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text, re.M)
    assert match, f"no line matching {pattern!r} in {path.name}"
    return match.group(1)


def test_the_package_and_the_protocol_both_declare_a_real_version():
    """A malformed version is worse than a stale one — it sorts unpredictably."""
    assert SEMVER.match(__version__), f"package version is not semver: {__version__!r}"
    assert SEMVER.match(ASCEP_VERSION), f"protocol version is not semver: {ASCEP_VERSION!r}"


def test_the_protocol_version_does_not_lag_the_package():
    """The failure this exists for happened. Release 0.3.0 was cut *because* the ITL fix
    altered the numbers a conforming report publishes -- the commit message says so and
    invokes the versioning rule by name -- and it bumped `__version__` while leaving
    `ASCEP_VERSION` at 0.2.0. Reports cite the protocol version and nothing else, so for the
    length of that release a report reduced the corrected way was indistinguishable from one
    reduced the broken way, by exactly the field the spec designates to tell them apart.

    Major and minor must agree; the patch level may run ahead, because a fix to the CLI that
    touches no number is a package release and not a protocol one. Under 0.x the minor is
    the breaking axis, so this is the same rule the CHANGELOG states in prose.
    """
    package = tuple(int(p) for p in __version__.split(".")[:2])
    protocol = tuple(int(p) for p in ASCEP_VERSION.split(".")[:2])
    assert protocol == package, (
        f"package {__version__} ships against protocol {ASCEP_VERSION}; a release that "
        "changes what a report says must move the version the report cites"
    )


def test_pyproject_agrees_with_the_package():
    """The wheel's metadata version is what `pip` and PyPI show; `__version__` is what the
    CLI prints and what gets stamped into reports. A user reconciling a report against an
    installed package compares exactly these two."""
    declared = _first(r'^version\s*=\s*"([^"]+)"', ROOT / "pyproject.toml")
    assert declared == __version__, (
        f"pyproject.toml says {declared}, ascep/__init__.py says {__version__}"
    )


def test_citation_agrees_with_the_package():
    """CITATION.cff is what GitHub's citation widget and every reference manager read.

    A stale version here misattributes results to the wrong revision of the protocol, which
    for a measurement standard is the whole ballgame.
    """
    declared = _first(r"^version:\s*['\"]?([0-9][^'\"\s]*)", ROOT / "CITATION.cff")
    assert declared == __version__, (
        f"CITATION.cff says {declared}, ascep/__init__.py says {__version__}"
    )


def test_the_changelog_has_an_entry_for_the_current_version():
    """A release with no changelog entry is a release nobody can review.

    Only the presence of the heading is checked, not its contents: `[0.1.0]` must exist as a
    released section, so bumping the version forces the author past the point where writing
    the entry is the obvious next step.
    """
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(rf"^##\s*\[{re.escape(__version__)}\]", re.M)
    assert heading.search(changelog), (
        f"CHANGELOG.md has no released section for {__version__}; "
        "add one rather than shipping an unexplained version"
    )
    unreleased = changelog.find("## [Unreleased]")
    current = changelog.find(f"## [{__version__}]")
    assert unreleased != -1, "CHANGELOG.md lost its [Unreleased] section"
    assert unreleased < current, "[Unreleased] must stay above the released sections"


@pytest.mark.parametrize("path", ["README.md", "protocol/SPEC.md"])
def test_prose_does_not_advertise_a_version_the_code_does_not_ship(path):
    """Guards the specific failure of a bumped package and un-bumped documentation.

    A README still saying v0.1 after a breaking 1.0 tells readers the numbers they measured
    under 0.1 are still comparable, which is precisely the claim the versioning rule exists to
    control. Only *stale* references are flagged: a forward reference like "expect churn before
    v0.2" is roadmap language and stays legal, because a test that banned it would be fixed by
    deleting the roadmap rather than by updating the version.
    """
    text = (ROOT / path).read_text(encoding="utf-8")
    shipped = tuple(int(p) for p in __version__.split(".")[:2])
    advertised = {tuple(int(p) for p in v.split(".")) for v in re.findall(r"\bv(\d+\.\d+)\b", text)}
    stale = sorted(v for v in advertised if v < shipped)
    assert not stale, (
        f"{path} still advertises "
        + ", ".join(f"v{a}.{b}" for a, b in stale)
        + f"; the package ships {__version__}"
    )
