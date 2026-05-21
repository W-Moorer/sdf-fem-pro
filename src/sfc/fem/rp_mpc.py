"""Abaqus-style reference-point hub MPC helpers.

The helpers in this module represent the common rigid hub/reference-point
kinematics used by Abaqus ``*MPC, BEAM`` gear examples in an eliminated form.
They do not introduce Abaqus as a dependency: the returned nodal displacement
values are ordinary Dirichlet data that can be consumed by the standalone SFC
finite-element solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_AXIS_TO_COMPONENT = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


@dataclass(frozen=True, slots=True)
class RigidHubMPC:
    """Rigid hub motion tied to a reference point.

    The translational displacement of node ``i`` is

    ``u_i = u_rp + theta x (X_i - X_rp)``.

    This is the small-rotation form of the Abaqus BEAM MPC kinematics and is
    the right representation for linearized implicit iterations and cropped
    gear submodels.  Large finite rotations should be supplied incrementally or
    handled by a higher-level rotation update before calling this helper.
    """

    node_ids: np.ndarray
    reference_nodes: np.ndarray
    reference_point: np.ndarray

    def __post_init__(self) -> None:
        node_ids = np.asarray(self.node_ids, dtype=np.int64).ravel()
        if node_ids.size == 0:
            raise ValueError("node_ids must not be empty")
        if np.any(node_ids < 0):
            raise ValueError("node_ids cannot contain negative entries")
        nodes = np.asarray(self.reference_nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("reference_nodes must have shape (n_nodes, 3)")
        if int(node_ids.max()) >= nodes.shape[0]:
            raise ValueError("node_ids reference a node outside reference_nodes")
        rp = np.asarray(self.reference_point, dtype=float).reshape(-1)
        if rp.shape != (3,):
            raise ValueError("reference_point must have shape (3,)")
        object.__setattr__(self, "node_ids", np.unique(node_ids))
        object.__setattr__(self, "reference_nodes", nodes)
        object.__setattr__(self, "reference_point", rp)

    def nodal_displacements(
        self,
        *,
        translation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        rotation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Return prescribed displacements for the hub nodes."""

        u_rp = np.asarray(translation, dtype=float).reshape(-1)
        theta = np.asarray(rotation, dtype=float).reshape(-1)
        if u_rp.shape != (3,) or theta.shape != (3,):
            raise ValueError("translation and rotation must have shape (3,)")
        lever = self.reference_nodes[self.node_ids] - self.reference_point[None, :]
        return u_rp[None, :] + np.cross(theta[None, :], lever)

    def nodal_velocities(
        self,
        *,
        translation_rate: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Return hub-node velocities induced by RP translational/rotational rates."""

        v_rp = np.asarray(translation_rate, dtype=float).reshape(-1)
        omega = np.asarray(angular_velocity, dtype=float).reshape(-1)
        if v_rp.shape != (3,) or omega.shape != (3,):
            raise ValueError("translation_rate and angular_velocity must have shape (3,)")
        lever = self.reference_nodes[self.node_ids] - self.reference_point[None, :]
        return v_rp[None, :] + np.cross(omega[None, :], lever)

    def dirichlet_dofs_and_values(
        self,
        *,
        translation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        rotation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        components: str | Sequence[str | int] = "xyz",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return global DOF indices and prescribed values for this hub."""

        comp = _normalize_components(components)
        disp = self.nodal_displacements(translation=translation, rotation=rotation)
        dofs = self.node_ids[:, None] * 3 + comp[None, :]
        values = disp[:, comp]
        order = np.argsort(dofs.ravel(), kind="stable")
        return dofs.ravel()[order].astype(np.int64), values.ravel()[order].astype(float)


def merge_dirichlet_conditions(*conditions: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Merge multiple ``(dofs, values)`` pairs, rejecting conflicts."""

    merged: dict[int, float] = {}
    for dofs, values in conditions:
        d = np.asarray(dofs, dtype=np.int64).ravel()
        v = np.asarray(values, dtype=float).ravel()
        if d.shape != v.shape:
            raise ValueError("each dofs array must match its values array")
        for dof, value in zip(d, v, strict=True):
            key = int(dof)
            val = float(value)
            if key in merged and not np.isclose(merged[key], val, rtol=1.0e-10, atol=1.0e-12):
                raise ValueError(f"conflicting Dirichlet values for dof {key}")
            merged[key] = val
    if not merged:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    dofs = np.asarray(sorted(merged), dtype=np.int64)
    values = np.asarray([merged[int(dof)] for dof in dofs], dtype=float)
    return dofs, values


def _normalize_components(components: str | Sequence[str | int]) -> np.ndarray:
    if isinstance(components, str):
        labels: Sequence[str | int]
        if components.lower() in {"all", "xyz"}:
            labels = ("x", "y", "z")
        else:
            labels = tuple(components.lower())
    else:
        labels = components
    try:
        comp = np.asarray([_AXIS_TO_COMPONENT[label] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("components must contain only x/y/z or 0/1/2") from exc
    if comp.size == 0:
        raise ValueError("components must not be empty")
    return np.unique(comp)
