"""Offline schema loading and validation.

The schemas carry absolute ``$id`` URIs under ``https://ascep.dev/schemas/`` because a schema
needs a stable global identity. But their cross-references resolve against that base, so a
naive ``jsonschema.validate`` tries to fetch them over the network and fails on any machine
that is offline, air-gapped, or simply behind a proxy — which describes most of the clusters
this protocol is meant to be run on.

This module builds a registry that maps those URIs to the copies shipped in ``schemas/``, so
validation never touches the network. Use :func:`validator_for` or :func:`validate` rather
than constructing a validator yourself.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator
from typing import Any


def _schema_dir() -> pathlib.Path:
    """Where the shipped schemas live, in both of the layouts this package exists in.

    In a wheel they are force-included at ``ascep/schemas`` (see pyproject); in a source
    checkout they sit at the repository root so they are browsable next to `protocol/`.
    Resolving only the checkout layout is a bug that no editable install can reveal: `pip
    install -e .` leaves the source tree in place, so CI stays green while `pip install ascep`
    gives every user a FileNotFoundError on their first `ascep validate`.
    """
    packaged = pathlib.Path(__file__).parent / "schemas"
    if packaged.is_dir():
        return packaged
    return pathlib.Path(__file__).parent.parent / "schemas"


SCHEMA_DIR = _schema_dir()

LAYERS = ("hardware", "model", "serving", "run", "workload", "capacity-report")


def schema_path(name: str) -> pathlib.Path:
    """Path to a shipped schema. ``name`` may be ``"model"`` or ``"model.schema.json"``."""
    fname = name if name.endswith(".schema.json") else f"{name}.schema.json"
    p = SCHEMA_DIR / fname
    if not p.exists():
        raise FileNotFoundError(f"no such schema: {fname}. Known: {', '.join(LAYERS)}")
    return p


def load_schema(name: str) -> dict:
    return json.loads(schema_path(name).read_text(encoding="utf-8"))


def _all_schemas() -> Iterator[dict]:
    for p in sorted(SCHEMA_DIR.glob("*.schema.json")):
        yield json.loads(p.read_text(encoding="utf-8"))


def build_registry():
    """A ``referencing.Registry`` resolving every ASCEP ``$id`` to its local copy.

    Registers each schema under both its absolute ``$id`` and its bare filename, so a report
    validates identically whether it was written against the published URI or a local path.
    """
    from referencing import Registry, Resource

    resources = []
    for schema in _all_schemas():
        resource = Resource.from_contents(schema)
        uri = schema.get("$id", "")
        if uri:
            resources.append((uri, resource))
            resources.append((uri.rsplit("/", 1)[-1], resource))
    return Registry().with_resources(resources)


def validator_for(name: str):
    """A Draft 2020-12 validator for one layer, wired for offline ``$ref`` resolution."""
    from jsonschema import Draft202012Validator

    return Draft202012Validator(load_schema(name), registry=build_registry())


def iter_errors(name: str, instance: Any):
    """Yield validation errors, sorted by location so output is stable across runs."""
    return sorted(validator_for(name).iter_errors(instance), key=lambda e: str(e.json_path))


def validate(name: str, instance: Any) -> list[str]:
    """Validate and return human-readable messages. Empty list means valid.

    Returns messages rather than raising because a conformance check wants *every* problem in
    one pass — fixing a declaration one exception at a time is how contributors give up.
    """
    return [f"{e.json_path}: {e.message}" for e in iter_errors(name, instance)]
