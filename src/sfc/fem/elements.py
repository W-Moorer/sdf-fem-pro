"""Finite-element backend registry.

The registry keeps global assembly independent of a specific element formula.
Additional elements are added by registering another backend with the same
stiffness, mass, and volume interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .hex8 import hex8_mass, hex8_stiffness, hex8_volume
from .tet4 import tet4_mass, tet4_stiffness, tet4_volume

StiffnessFunction = Callable[[np.ndarray, float, float], np.ndarray]
MassFunction = Callable[[np.ndarray, float, str], np.ndarray]
VolumeFunction = Callable[[np.ndarray], float]


@dataclass(frozen=True, slots=True)
class ElementBackend:
    """Element-level operations required by sparse FEM assembly."""

    name: str
    aliases: tuple[str, ...]
    nodes_per_element: int
    stiffness: StiffnessFunction
    mass: MassFunction
    volume: VolumeFunction

    def nodal_body_force(self, Xe: np.ndarray, density: float, body_force: np.ndarray) -> np.ndarray:
        """Return the constant body-force vector assigned to each element node."""

        force = np.asarray(body_force, dtype=float)
        if force.shape != (3,):
            raise ValueError("body_force must have shape (3,)")
        return float(density) * self.volume(Xe) * force / float(self.nodes_per_element)


def _tet4_mass_backend(Xe: np.ndarray, rho: float, kind: str = "consistent") -> np.ndarray:
    return tet4_mass(Xe, rho, kind=kind)


C3D4_TET4_BACKEND = ElementBackend(
    name="tet4",
    aliases=("tet4", "c3d4"),
    nodes_per_element=4,
    stiffness=tet4_stiffness,
    mass=_tet4_mass_backend,
    volume=tet4_volume,
)

C3D8_HEX8_BACKEND = ElementBackend(
    name="hex8",
    aliases=("hex8", "c3d8"),
    nodes_per_element=8,
    stiffness=hex8_stiffness,
    mass=hex8_mass,
    volume=hex8_volume,
)

_REGISTERED_BACKENDS: dict[str, ElementBackend] = {}
_KNOWN_UNREGISTERED_ALIASES = {
    "tet10": "c3d10",
    "c3d10": "c3d10",
}


def register_element_backend(backend: ElementBackend) -> None:
    """Register an element backend under its canonical name and aliases."""

    keys = (backend.name, *backend.aliases)
    for key in keys:
        normalized = str(key).lower()
        if normalized in _KNOWN_UNREGISTERED_ALIASES:
            _KNOWN_UNREGISTERED_ALIASES.pop(normalized, None)
        _REGISTERED_BACKENDS[normalized] = backend


def available_element_backends() -> tuple[str, ...]:
    """Return canonical names of registered FEM element backends."""

    names: list[str] = []
    for backend in _REGISTERED_BACKENDS.values():
        if backend.name not in names:
            names.append(backend.name)
    return tuple(names)


def _available_alias_text() -> str:
    alias_groups: list[str] = []
    seen: set[str] = set()
    for backend in _REGISTERED_BACKENDS.values():
        if backend.name in seen:
            continue
        seen.add(backend.name)
        alias_groups.append("/".join(backend.aliases))
    return ", ".join(alias_groups) if alias_groups else "none"


def get_element_backend(element_type: str) -> ElementBackend:
    """Return the registered FEM backend for ``element_type``.

    ``tet4``/``C3D4`` and ``hex8``/``C3D8`` are registered aliases. C3D10 is
    intentionally not registered yet; callers get a clear error instead of
    silently falling back to a lower-order element.
    """

    key = str(element_type).lower()
    backend = _REGISTERED_BACKENDS.get(key)
    if backend is not None:
        return backend

    if key in _KNOWN_UNREGISTERED_ALIASES:
        target = _KNOWN_UNREGISTERED_ALIASES[key]
        raise ValueError(
            f"No FEM element backend registered for '{target}'. "
            f"Registered backends: {_available_alias_text()}"
        )

    raise ValueError(
        f"Unsupported FEM element type '{element_type}'. "
        f"Registered backends: {_available_alias_text()}"
    )


def canonical_fem_element_type(element_type: str) -> str:
    """Return the canonical registered FEM element type name."""

    return get_element_backend(element_type).name


register_element_backend(C3D4_TET4_BACKEND)
register_element_backend(C3D8_HEX8_BACKEND)
