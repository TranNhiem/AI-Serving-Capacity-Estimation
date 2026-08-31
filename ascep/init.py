"""Generate a fillable report skeleton from the schemas.

The barrier to a first ASCEP report is not the measuring, it is that a report is a deeply
nested document validating against six schemas nobody reads first. This module walks the
schema and emits every field it requires, so the starting point is a file to fill in rather
than a specification to study.

It is driven entirely by the schema. A hand-written skeleton would rot the moment someone
added a required field, and would do it silently -- the contributor would produce a document
that validated a year ago. What it deliberately does NOT do is produce something conforming:
the placeholders are `null` and `TODO`, `ascep conformance` grades them as gaps, and working
through that verdict is how the rules get learned. Filling in a plausible number to make the
checker quiet is the failure the protocol exists to prevent, so the skeleton never supplies
one.

That rule has a consequence worth stating plainly, because it looks like a bug: **a fresh
skeleton does not pass `ascep validate`.** Two places in the schemas refuse a document that
leaves its sizing basis entirely unstated -- a workload must give either a concurrency or a
population, and a measurement point must give either a context length or an input length.
Neither has an honest placeholder, so :func:`decisions` reports them and `ascep init` prints
them, rather than the tool picking a number on the user's behalf. Every *other* validation
error on a fresh skeleton would be this module's fault, and :mod:`tests.test_init` holds that
line.
"""

from __future__ import annotations

import json
from typing import Any

from ascep import ASCEP_VERSION

from .validation import LAYERS, load_schema

# A $ref cycle would otherwise recurse forever. The real schemas nest about five deep, so
# hitting this cap means the schema changed shape, not that the cap is too low.
_MAX_DEPTH = 16

# Every generated placeholder carries this substring, so `grep TODO report.json` is a complete
# list of what is left to do, and a publish-time check can refuse a skeleton that was never
# filled in.
TODO = "TODO"

# Date-time fields get the same `TODO` as any other string, deliberately. The obvious
# alternative -- a parseable sentinel like the epoch -- was worse in the one way that matters:
# it validates, it survives `grep TODO`, and it reads as a real generation timestamp, so an
# untouched skeleton would publish a date nobody chose. `format` is an annotation the validator
# does not enforce, so `TODO` costs nothing here; if it ever is enforced, the resulting error
# points at an unfilled field, which is exactly what this module wants an error to do.

# Fields whose value is a fact about the document being generated rather than a claim about
# the user's system. Emitting the real protocol version is not a fabricated value, and it is
# the only way `ascep_version` can satisfy its `pattern` -- a "TODO" there fails validation
# for a reason that is entirely the tool's doing. This map stays at one entry as long as
# `ascep_version` remains the only patterned string in the schemas, which test_init asserts:
# a new `pattern` needs a real answer here, not another unmatchable placeholder.
_KNOWN = {"ascep_version": ASCEP_VERSION}


def _u_reason(field: str) -> str:
    return f"(U) {TODO}: state why {field} is unknown, or fill in {field} and delete this"


def _resolve(node: dict, root: dict) -> tuple[dict, dict]:
    """Follow a ``$ref``, returning the target and the document it lives in.

    Two forms appear: ``#/$defs/name`` within the current document, and a bare
    ``other.schema.json`` naming a sibling layer. The sibling case is why this returns the new
    root as well -- a ``$defs`` reference inside `model.schema.json` must resolve against that
    file, not against whoever pointed at it.
    """
    ref = node["$ref"]
    if ref.startswith("#/"):
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part]
        return target, root
    layer = ref.split("/")[-1].removesuffix(".schema.json")
    if layer not in LAYERS:
        raise ValueError(f"cannot resolve $ref {ref!r}: not one of {', '.join(LAYERS)}")
    other = load_schema(layer)
    return other, other


def _narrow(target: dict, siblings: dict) -> dict:
    """Overlay ``$ref`` siblings onto the resolved target, one level into ``properties``.

    Shallow everywhere else on purpose: the only narrowing the schemas do is pinning a
    property to a ``const``, and a general deep merge would quietly invent semantics
    (two ``required`` lists, two ``enum``s) that JSON Schema does not give it.
    """
    merged = dict(target)
    for key, value in siblings.items():
        if key == "properties" and isinstance(value, dict):
            props = dict(target.get("properties") or {})
            for name, override in value.items():
                props[name] = {**(props.get(name) or {}), **override}
            merged["properties"] = props
        elif key != "description":
            merged[key] = value
    return merged


def _merged(node: dict, root: dict) -> tuple[dict, list]:
    """Properties and required list for an object, flattening ``allOf``.

    Conditional branches (``if``/``then``, ``oneOf``, ``anyOf``) are ignored on purpose. Their
    requirements contradict each other by construction -- `model.schema.json` demands MLA
    geometry or GQA geometry, never both -- so a skeleton that tried to satisfy every branch
    would be a document no schema accepts.

    Only the unconditional fields are emitted, and only the ``anyOf``/``oneOf`` disjunctions
    reach :func:`decisions`. An ``if``/``then`` is keyed on a value the skeleton does not have
    yet: until ``attention_type`` says ``mla``, there is no sense in which ``kv_lora_rank`` is
    required, and listing every branch's fields up front would name a dozen the user will never
    need. They appear the moment they become real -- fill in the discriminator, run
    ``ascep validate``, and the branch's own requirements are the next errors.
    """
    props = dict(node.get("properties") or {})
    required = list(node.get("required") or [])
    for branch in node.get("allOf") or []:
        if "$ref" in branch:
            # Resolve into a local, never back into `root`. Rebinding it would make a later
            # `#/$defs/...` branch resolve against whichever file the previous branch pointed
            # at -- a KeyError if the name is absent there, and silently the wrong subschema
            # if it happens to exist.
            branch, _ = _resolve(branch, root)
        props.update(branch.get("properties") or {})
        required += [r for r in (branch.get("required") or []) if r not in required]
    return props, required


def _types(node: dict) -> list:
    t = node.get("type")
    if t is None:
        return []
    return list(t) if isinstance(t, list) else [t]


def _value(node: dict, root: dict, depth: int, path: str, notes: list) -> Any:
    """The placeholder for one leaf, in the precedence order the module docstring implies."""
    if depth > _MAX_DEPTH:
        # Raising rather than returning None: a null here is indistinguishable from an honest
        # unknown, so a $ref cycle would ship as a skeleton quietly missing a whole subtree.
        raise ValueError(
            f"schema nesting exceeded {_MAX_DEPTH} at {path or '(root)'}; this is a $ref cycle, "
            "not a deep schema -- the real ones nest five levels"
        )
    # A $ref may point at another $ref, so follow the chain rather than one hop: stopping
    # early leaves a node with no `type`, which falls through every branch below and returns
    # None -- a subtree silently replaced by an honest-looking unknown.
    for hop in range(_MAX_DEPTH + 1):
        if "$ref" not in node:
            break
        if hop == _MAX_DEPTH:
            raise ValueError(f"$ref cycle at {path or '(root)'}: {node['$ref']}")
        # Keywords sitting beside a $ref narrow it rather than being replaced by it, which is
        # how the four capacity rows share one shape while each pinning its own `tier` const.
        # Dropping the siblings would make the skeleton emit "TODO" for a value the schema
        # states outright.
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        node, root = _resolve(node, root)
        if siblings:
            node = _narrow(node, siblings)

    # A const or default is the schema stating the answer. Neither is a guess.
    if "const" in node:
        return node["const"]
    if "default" in node:
        return node["default"]

    types = _types(node)

    if "object" in types or "properties" in node or "allOf" in node:
        return _object(node, root, depth, path, notes)

    items = node.get("items")
    if "array" in types and isinstance(items, dict):
        shaped = "$ref" in items or items.get("required") or items.get("properties")
        # A list of sub-objects is worth emitting even where the field is nullable: `null`
        # tells the user nothing about what a row looks like, and the shape is the hard part.
        if shaped:
            count = max(1, int(node.get("minItems") or 1))
            return [_value(items, root, depth + 1, f"{path}[]", notes) for _ in range(count)]

    # An honest unknown is what C1 asks for, and the `_u_reason` companion beside it is where
    # the user says so. Anything else here is a value a reader could mistake for a declaration.
    if "null" in types:
        return None

    if "array" in types:
        # minItems on a scalar array still has to produce slots, or the emitted `[]` fails
        # validation for a reason the user cannot see in the file.
        count = int(node.get("minItems") or 0)
        return [_value(items, root, depth + 1, f"{path}[]", notes) for _ in range(count)]

    # A string field can hold a visible placeholder. Nothing else can, so everything else is
    # null -- including where the schema forbids null and a satisfying value was available.
    # `gpu_count: 1` would satisfy `minimum: 1` and validate silently, and it reads as a
    # declaration: a report left half-filled would claim a one-GPU deployment and nothing in
    # the toolchain would object. A null fails validation with "None is not of type integer",
    # which is the same information delivered as a task. Erring toward the error is the whole
    # posture of this protocol.
    if "string" in types and "null" not in types:
        return TODO
    return None


def _record_decision(node: dict, out: dict, path: str, notes: list) -> None:
    """Note a disjunction the skeleton cannot satisfy without inventing the user's data.

    An ``anyOf``/``oneOf`` of the form "give me A, or give me B and C" is the schema refusing
    a document that never states its basis at all. There is no honest placeholder for that --
    a zero is a claim -- so the choice is surfaced by name instead of guessed at. Branches
    already satisfied by the emitted fields are not reported: this must go quiet on its own if
    a future schema grows a disjunction the unconditional fields happen to cover.
    """
    for keyword in ("anyOf", "oneOf"):
        branches = [b.get("required") for b in node.get(keyword) or []]
        options = [list(r) for r in branches if r]
        if not options:
            continue
        if any(all(out.get(f) is not None for f in option) for option in options):
            continue
        entry = {"path": path or "(root)", "options": options}
        if entry not in notes:
            notes.append(entry)


def _object(node: dict, root: dict, depth: int, path: str, notes: list) -> dict:
    props, required = _merged(node, root)
    out: dict = {}
    for name in required:
        child = props.get(name)
        if name in _KNOWN:
            out[name] = _KNOWN[name]
            continue
        child_path = f"{path}.{name}" if path else name
        out[name] = None if child is None else _value(child, root, depth + 1, child_path, notes)
        # A null with no stated reason is exactly what C1 rejects, so the companion field is
        # emitted whenever the schema offers one -- including when it is not itself required.
        # Leaving the user to discover the convention from a conformance failure wastes the
        # one moment they are looking at the field.
        companion = f"{name}_u_reason"
        if out[name] is None and companion in props and companion not in out:
            out[companion] = _u_reason(name)
    _record_decision(node, out, path, notes)
    return out


def _build(layer: str) -> tuple[dict, list]:
    schema = load_schema(layer)
    notes: list = []
    return _object(schema, schema, 0, "", notes), notes


def skeleton(layer: str = "capacity-report") -> dict:
    """Every field ``layer``'s schema requires, with placeholders for the values."""
    return _build(layer)[0]


def decisions(layer: str = "capacity-report") -> list:
    """The choices a human must make before ``layer``'s skeleton will validate.

    Each entry is ``{"path": str, "options": [[field, ...], ...]}`` -- satisfying any one
    option is enough. Derived from the same walk that emits the skeleton, so the paths are
    guaranteed to be paths that exist in it.
    """
    return _build(layer)[1]


def render(layer: str = "capacity-report") -> str:
    """``skeleton`` as the JSON text to write to disk."""
    return json.dumps(skeleton(layer), indent=2) + "\n"
