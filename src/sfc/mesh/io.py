"""Native mesh I/O namespace.

Core mesh I/O intentionally excludes legacy solver importers. Optional external
format readers should live outside the solver core.
"""

from .topology import VolumeMesh

__all__ = ["VolumeMesh"]
