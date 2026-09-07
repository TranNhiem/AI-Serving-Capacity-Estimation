"""The README figure is generated, so the committed file must still match its generator.

A hero image is the first thing a reader trusts and the last thing anyone re-derives. The
failure this guards against is a silent divergence: someone edits ``tools/make_figure.py``
-- moves a row, changes a number -- and never re-runs it, so the README keeps showing the
old drawing while the source of truth says something else. Same discipline as
``test_report_conformance.py`` applies to the published examples: the artefact in the repo
is only trustworthy if a test regenerates it and compares bytes.

The second group is about what GitHub will actually render. GitHub sanitises SVG in
markdown: ``<script>``, ``<foreignObject>``, external fonts and remote images are stripped
or the whole image is refused. A figure that renders perfectly in a local browser and
shows as a broken-image icon on the project page is the exact outcome these check for --
including the one that already happened once, a font-family string whose nested double
quotes closed the XML attribute early and made the document unparseable.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import xml.dom.minidom

ROOT = pathlib.Path(__file__).resolve().parents[1]
SVG_PATH = ROOT / "assets" / "ascep-capacity-model.svg"

_spec = importlib.util.spec_from_file_location("make_figure", ROOT / "tools" / "make_figure.py")
make_figure = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(make_figure)


def test_the_committed_svg_is_what_the_generator_produces_today() -> None:
    assert SVG_PATH.read_text(encoding="utf-8") == make_figure.build_svg(), (
        "assets/ascep-capacity-model.svg is stale -- run `python tools/make_figure.py`"
    )


def test_the_generator_is_deterministic() -> None:
    # No timestamps, no set iteration, no randomness: two runs in one process must agree,
    # or every regeneration produces a spurious diff and the staleness test above becomes
    # noise that people learn to ignore.
    assert make_figure.build_svg() == make_figure.build_svg()


def test_the_svg_parses_as_xml() -> None:
    # A malformed attribute renders as a browser error page, not as a slightly-off figure,
    # and nothing about the file size or the exit code says so.
    xml.dom.minidom.parseString(SVG_PATH.read_text(encoding="utf-8"))


def test_the_svg_carries_nothing_github_strips() -> None:
    svg = SVG_PATH.read_text(encoding="utf-8")
    for banned in ("<script", "<foreignObject", "@import", "xlink:href", "<image"):
        assert banned not in svg, f"GitHub sanitises {banned} out of inline SVG"


def test_the_svg_references_no_remote_resource() -> None:
    # The XML namespace is the one legitimate URL in the document; anything else is a
    # fetch that will not happen behind GitHub's proxy, leaving a hole in the figure.
    urls = set(re.findall(r"https?://[^\"'\s>]+", SVG_PATH.read_text(encoding="utf-8")))
    assert urls == {"http://www.w3.org/2000/svg"}, urls


def test_the_root_element_is_sized_so_the_readme_can_scale_it() -> None:
    root = xml.dom.minidom.parse(str(SVG_PATH)).documentElement
    assert root.tagName == "svg"
    for attribute in ("width", "height", "viewBox"):
        assert root.getAttribute(attribute), f"missing {attribute} on <svg>"


def test_the_figure_states_the_rule_it_illustrates() -> None:
    # If the four floor names or the binding constraint ever drop out of the drawing, the
    # figure stops teaching the one thing it exists to teach (rule C5: name what binds).
    svg = SVG_PATH.read_text(encoding="utf-8")
    for token in ("Weights", "KV", "Prefill", "Throughput", "BINDING CONSTRAINT", "min("):
        assert token in svg
