"""Compatibility aliases for the model-context API.

Older examples and tests imported these helpers from ``context``. Model
registration moved to ``model_context``; keep the old imports working while
callers migrate.
"""

from sample_factory.algo.utils.model_context import global_model_factory, reset_global_model_context

reset_global_context = reset_global_model_context

__all__ = ["global_model_factory", "reset_global_context"]
