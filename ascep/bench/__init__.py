"""The ASCEP benchmark harness: the empirical half of the protocol.

``ascep.capacity`` models what a configuration *should* do; this package measures what it
actually does, and emits records that populate ``run.schema.json``. The two halves are kept
apart on purpose. ``ascep.capacity`` is stdlib-only so the arithmetic can be audited on an
air-gapped login node, whereas a load generator that is not allowed an async HTTP client
becomes the bottleneck it is supposed to be measuring. The client lives behind the ``run``
extra (``pip install ascep[run]``) and nothing in the analytic half imports this package.

:mod:`ascep.bench.records` is itself stdlib-only and stays that way: reading someone else's
published records to re-derive their percentiles must not require our HTTP client.
"""

from ascep.bench.records import Outcome, RequestRecord, read_records, write_records

__all__ = ["Outcome", "RequestRecord", "read_records", "write_records"]
