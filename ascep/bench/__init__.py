"""The ASCEP benchmark harness: the empirical half of the protocol.

``ascep.capacity`` models what a configuration *should* do; this package measures what it
actually does, and emits records that populate ``run.schema.json``. The two halves are kept
apart on purpose. ``ascep.capacity`` is stdlib-only so the arithmetic can be audited on an
air-gapped login node, whereas a load generator that is not allowed an async HTTP client
becomes the bottleneck it is supposed to be measuring. The client lives behind the ``run``
extra (``pip install ascep[run]``) and nothing in the analytic half imports this package.

The whole analytic path through this package -- :mod:`ascep.bench.records`,
:mod:`ascep.bench.metrics`, :mod:`ascep.bench.ladder`, :mod:`ascep.bench.driver`,
:mod:`ascep.bench.workloads`, :mod:`ascep.bench.persist` and :mod:`ascep.bench.report` -- is
itself stdlib-only and stays that way: reading someone else's published records to re-derive
their percentiles, re-grade their rungs, re-rule their window boundaries, rebuild the prompts
those records came from, reassemble the report those rows belong to, and check the downloaded
bundle against its own digests must not require our HTTP client. Only the
adapters need it, which is why this initialiser deliberately re-exports nothing from them.
"""

from ascep.bench.records import Outcome, RequestRecord, read_records, write_records

__all__ = ["Outcome", "RequestRecord", "read_records", "write_records"]
