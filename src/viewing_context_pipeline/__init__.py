"""Thin orchestration package for the fixed Viewing Context pipeline."""

from .pipeline import STAGES
from .runtime import ConfigError, RunContext

__all__ = ["STAGES", "ConfigError", "RunContext"]
