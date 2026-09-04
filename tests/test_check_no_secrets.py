"""The publication gate, and the two ways it used to pass without checking anything.

check_no_secrets.py is the last thing between private infrastructure and a public push, so
the only failure that matters here is the false pass: a run that prints "OK: no findings"
and exits 0 over bytes it never read, or over rules it silently dropped. Both had happened.

The first: ``iter_files`` walked ``root.rglob("*")``, which yields nothing for a file, so
``--path some/report.md`` scanned zero files and reported the document clean.

The second: the deny-list was resolved as ``root / ".ascep-denylist"``. Scan a subdirectory,
a ``git archive`` export or a bundle staged under /tmp -- exactly what an operator does
before publishing -- and there is no deny-list beside it, so every site-specific hostname,
codename and customer name stopped being checked while the run still said OK. The deny-list
belongs to the checkout, not to the tree being scanned.

The third group covers the clean-file cache, which exists because a full scan of the
published bundles takes tens of minutes and grows with every campaign. A cache in a safety
gate is a third way to pass without checking, so most of these tests are about the ways it
must refuse to: unseen bytes, edited bytes, and a rule that did not exist last time.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_no_secrets", ROOT / "tools" / "check_no_secrets.py"
)
cns = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cns)

# Fabricated, and it must stay fabricated. It is shaped like a GitHub classic PAT so the
# generic patterns fire on it; nothing that was ever live goes in a fixture.
FAKE_PAT = "ghp_" + "A" * 36


def test_a_file_path_scans_that_file_instead_of_reporting_it_clean(tmp_path, capsys):
    leak = tmp_path / "report.md"
    leak.write_text(f"the token is {FAKE_PAT}\n", encoding="utf-8")

    assert cns.scan(leak) == 1, "scanning a file directly must read it, not yield nothing"
    assert "report.md:1" in capsys.readouterr().out


def test_a_clean_file_path_still_passes(tmp_path):
    ok = tmp_path / "report.md"
    ok.write_text("nothing interesting here\n", encoding="utf-8")

    assert cns.scan(ok) == 0


def test_a_directory_scan_is_unchanged(tmp_path, capsys):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text(f"{FAKE_PAT}\n", encoding="utf-8")

    assert cns.scan(tmp_path) == 1
    assert "sub/a.md:1" in capsys.readouterr().out, "paths stay relative to the scanned root"


def test_the_checkouts_denylist_applies_to_a_tree_outside_the_checkout(tmp_path, monkeypatch):
    """The case that matters: staging an export elsewhere must not disarm the deny-list."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".ascep-denylist").write_text("(?i)internal-codename-x\n", encoding="utf-8")
    monkeypatch.setattr(cns, "REPO_ROOT", checkout)

    export = tmp_path / "staged-export"
    export.mkdir()
    (export / "README.md").write_text("built on Internal-Codename-X\n", encoding="utf-8")

    assert cns.scan(export) == 1, "a scan away from the checkout still owes the operator's rules"


def test_a_denylist_beside_the_scanned_tree_is_also_honoured(tmp_path, monkeypatch):
    monkeypatch.setattr(cns, "REPO_ROOT", tmp_path / "no-such-checkout")
    tree = tmp_path / "bundle"
    tree.mkdir()
    (tree / ".ascep-denylist").write_text("forbidden-host\n", encoding="utf-8")
    (tree / "engine.log").write_text("connected to forbidden-host\n", encoding="utf-8")

    assert cns.scan(tree) == 1


def test_both_denylists_are_unioned_and_neither_is_read_twice(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".ascep-denylist").write_text("from-checkout\n", encoding="utf-8")
    monkeypatch.setattr(cns, "REPO_ROOT", checkout)

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / ".ascep-denylist").write_text("from-tree\n", encoding="utf-8")
    (tree / "a.md").write_text("from-checkout and from-tree\n", encoding="utf-8")

    assert cns.scan(tree) == 2, "one finding per rule, from both files"

    # Scanning the checkout itself must not load its own deny-list twice and double-count.
    (checkout / "b.md").write_text("from-checkout\n", encoding="utf-8")
    assert cns.scan(checkout / "b.md") == 1


def test_an_unreadable_denylist_regex_is_skipped_not_fatal(tmp_path, monkeypatch, capsys):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".ascep-denylist").write_text("valid-term\n[unclosed\n", encoding="utf-8")
    monkeypatch.setattr(cns, "REPO_ROOT", checkout)

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.md").write_text("valid-term\n", encoding="utf-8")

    assert cns.scan(tree) == 1
    assert "invalid regex" in capsys.readouterr().err


def test_every_pattern_that_should_fire_does(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".ascep-denylist").write_text(
        "(?i)codename-alpha\nsecret-cluster-[0-9]+\n", encoding="utf-8"
    )
    monkeypatch.setattr(cns, "REPO_ROOT", checkout)

    tree = tmp_path / "tree"
    tree.mkdir()
    # Every match here is assembled from pieces, for the same reason FAKE_PAT is. A fixture
    # that spells a pattern out in the source makes THIS file trip the gate, and a gate that
    # fails on its own tests every run is one an operator learns to skip -- which is the
    # false pass the whole module is about, arrived at from the other side.
    (tree / "a.md").write_text(
        "\n".join(
            [
                "a wholly ordinary line",
                f"a leaked {FAKE_PAT} token",
                "we ran it on Codename-Alpha",
                "reachable at 10.1" + ".2.3 from secret-cluster-42",
                "/home" + "/operator/ckpt and AKIA" + "B" * 16,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    raw = cns.GENERIC_PATTERNS + cns.load_denylist(tree)
    expected = sum(
        1
        for line in (tree / "a.md").read_text(encoding="utf-8").splitlines()
        for _, p, _ in raw
        if re.search(p, line)
    )
    assert expected >= 6, "the fixture has to trip several rules or this proves nothing"
    assert cns.scan(tree) == expected


def test_this_module_trips_none_of_the_patterns_it_exercises():
    """The fixtures above have to look like secrets without being findable in this file.

    Written as a rule rather than a one-off repair because the repair is invisible: a new
    fixture spelt out in full still passes its own test, and the only symptom is that
    ``python tools/check_no_secrets.py`` -- the command CI runs and the command the PR
    template asks a contributor to run before pushing -- now reports findings in the test
    suite forever. That is how a publication gate stops being read. Generic patterns only:
    the site deny-list is gitignored, so a checkout without one must reach the same verdict.
    """
    source = pathlib.Path(__file__).read_text(encoding="utf-8").splitlines()
    hits = [
        f"{name} on line {lineno}: {match.group(0)[:24]}"
        for lineno, line in enumerate(source, 1)
        for name, pattern, _ in cns.GENERIC_PATTERNS
        if (match := re.search(pattern, line))
    ]
    assert not hits, "assemble the fixture from pieces the way FAKE_PAT is:\n  " + "\n  ".join(hits)


def test_the_cache_skips_bytes_it_has_already_cleared(tmp_path, monkeypatch):
    monkeypatch.setattr(cns, "REPO_ROOT", tmp_path / "checkout")
    (tmp_path / "checkout").mkdir()
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.md").write_text("ordinary\n", encoding="utf-8")

    assert cns.scan(tree, use_cache=True) == 0
    cache = json.loads((tmp_path / "checkout" / cns.CACHE_NAME).read_text(encoding="utf-8"))
    assert len(cache["clean"]) == 1
    assert cns.scan(tree, use_cache=True) == 0


def test_the_cache_cannot_clear_a_file_it_has_not_seen(tmp_path, monkeypatch):
    """The whole risk of a cache in a safety gate: a false pass on bytes nobody read."""
    monkeypatch.setattr(cns, "REPO_ROOT", tmp_path / "checkout")
    (tmp_path / "checkout").mkdir()
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.md").write_text("ordinary\n", encoding="utf-8")
    assert cns.scan(tree, use_cache=True) == 0

    (tree / "b.md").write_text(f"{FAKE_PAT}\n", encoding="utf-8")
    assert cns.scan(tree, use_cache=True) == 1, "a new file is new bytes, and new bytes are read"

    # Editing a cleared file changes its digest, so its old verdict does not carry over.
    (tree / "a.md").write_text(f"{FAKE_PAT}\n", encoding="utf-8")
    assert cns.scan(tree, use_cache=True) == 2


def test_changing_the_rules_voids_every_cached_verdict(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(cns, "REPO_ROOT", checkout)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.md").write_text("built on codename-alpha\n", encoding="utf-8")

    assert cns.scan(tree, use_cache=True) == 0, "no rule covers it yet"

    (checkout / ".ascep-denylist").write_text("codename-alpha\n", encoding="utf-8")
    assert cns.scan(tree, use_cache=True) == 1, "a new rule must re-examine already-clean bytes"


def test_a_corrupt_cache_means_rescan_never_means_clean(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(cns, "REPO_ROOT", checkout)
    (checkout / cns.CACHE_NAME).write_text("{not json", encoding="utf-8")

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.md").write_text(f"{FAKE_PAT}\n", encoding="utf-8")

    assert cns.scan(tree, use_cache=True) == 1


def test_no_cache_is_written_when_the_cache_was_not_asked_for(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(cns, "REPO_ROOT", checkout)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.md").write_text("ordinary\n", encoding="utf-8")

    assert cns.scan(tree) == 0
    assert not (checkout / cns.CACHE_NAME).exists(), "CI must scan from scratch, leaving nothing"


def test_scan_defaults_to_one_job_so_a_dynamically_loaded_module_still_works():
    """This test file loads the tool by path, and so does tools/redact_bundle.py.

    A process pool re-imports the module by name in each child, which fails for a module that
    was never importable by name. Defaulting to a pool would therefore break every library
    caller while leaving the CLI green.
    """
    assert inspect.signature(cns.scan).parameters["jobs"].default == 1


def test_the_scanner_skips_the_files_that_hold_the_patterns(tmp_path, monkeypatch):
    monkeypatch.setattr(cns, "REPO_ROOT", tmp_path / "nowhere")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "check_no_secrets.py").write_text(f"{FAKE_PAT}\n", encoding="utf-8")
    (tree / ".ascep-denylist").write_text("nothing\n", encoding="utf-8")

    assert cns.scan(tree) == 0, "the scanner and its rule file would otherwise flag themselves"


def test_the_repo_itself_is_clean():
    """The gate, run over the tree it guards, with whatever rules this checkout carries."""
    assert cns.scan(ROOT / "protocol") == 0
    assert cns.scan(ROOT / "ascep") == 0
    assert cns.scan(ROOT / "schemas") == 0
