"""Framework adapters: the one place that knows a serving API's wire format.

Everything above this package works in ``RequestRecord`` terms, so supporting a framework
ASCEP has never seen means writing one adapter rather than touching the driver, the
reduction or the report.

This module deliberately does NOT import the concrete adapters. ``base`` is part of the
stdlib-only surface -- a third party implementing an adapter for their own framework must be
able to read the contract without installing our HTTP client -- and an eager re-export here
would drag ``httpx`` in through the package initialiser and quietly break that promise. Import
the concrete adapter by its own path::

    from ascep.bench.adapters.openai_compat import OpenAICompatAdapter
"""

from ascep.bench.adapters.base import Adapter, AdapterConfig, RequestSpec

__all__ = ["Adapter", "AdapterConfig", "RequestSpec"]
