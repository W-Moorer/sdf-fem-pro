"""Analysis-backend interfaces and shared result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.sparse import csr_matrix


class AnalysisBackend(Protocol):
    """Protocol implemented by static, dynamic, and future nonlinear solvers."""

    name: str


@dataclass(frozen=True, slots=True)
class StaticSolveResult:
    """Result of a linear static solve."""

    u: np.ndarray
    reactions: np.ndarray
    stiffness: csr_matrix
    force: np.ndarray
    free_dofs: np.ndarray


@dataclass(frozen=True, slots=True)
class DynamicStepResult:
    """Result of one dynamic time-integration step."""

    u: np.ndarray
    v: np.ndarray
    a: np.ndarray
    stiffness: csr_matrix
    mass: csr_matrix
    damping: csr_matrix
    force: np.ndarray


_REGISTERED_ANALYSIS_BACKENDS: dict[str, AnalysisBackend] = {}


def register_analysis_backend(backend: AnalysisBackend) -> None:
    """Register an analysis backend by name."""

    _REGISTERED_ANALYSIS_BACKENDS[str(backend.name).lower()] = backend


def available_analysis_backends() -> tuple[str, ...]:
    """Return names of registered analysis backends."""

    return tuple(_REGISTERED_ANALYSIS_BACKENDS.keys())


def get_analysis_backend(name: str) -> AnalysisBackend:
    """Return a registered analysis backend."""

    key = str(name).lower()
    try:
        return _REGISTERED_ANALYSIS_BACKENDS[key]
    except KeyError as exc:
        available = ", ".join(available_analysis_backends()) or "none"
        raise ValueError(f"Unsupported analysis backend '{name}'. Registered backends: {available}") from exc
