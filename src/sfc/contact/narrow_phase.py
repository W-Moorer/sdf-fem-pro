"""Legacy projection-kernel narrow-phase contact constraints.

New field-contact assembly should use ``sfc.contact.field_contact``. This
module remains for validation references and backward-compatible tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from sfc.sdf import SurfaceSDFResult, dynamic_surface_sdf


def _node_ids(value: np.ndarray | Sequence[int], name: str) -> np.ndarray:
    ids = np.asarray(value, dtype=np.int64).ravel()
    if ids.size == 0:
        raise ValueError(f"{name} must contain at least one node")
    if np.any(ids < 0):
        raise ValueError(f"{name} cannot contain negative node ids")
    return ids


def _weights(value: np.ndarray | Sequence[float], size: int, name: str) -> np.ndarray:
    weights = np.asarray(value, dtype=float).ravel()
    if weights.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError(f"{name} must sum to 1")
    return weights


@dataclass(frozen=True, slots=True)
class SurfaceSample:
    """Slave surface quadrature/sample point definition."""

    node_ids: np.ndarray
    weights: np.ndarray
    candidate_face_ids: np.ndarray

    def __post_init__(self) -> None:
        node_ids = _node_ids(self.node_ids, "node_ids")
        weights = _weights(self.weights, node_ids.size, "weights")
        candidates = np.asarray(self.candidate_face_ids, dtype=np.int64).ravel()
        if np.any(candidates < 0):
            raise ValueError("candidate_face_ids cannot contain negative ids")

        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "candidate_face_ids", candidates)

    def point(self, x_current: np.ndarray) -> np.ndarray:
        X = np.asarray(x_current, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("x_current must have shape (n, 3)")
        if int(self.node_ids.max()) >= X.shape[0]:
            raise ValueError("sample references a node outside x_current")
        return self.weights @ X[self.node_ids]


@dataclass(frozen=True, slots=True)
class ContactConstraint:
    """One slave sample against one closest master surface feature."""

    g: float
    n: np.ndarray
    master_face_id: int
    master_node_ids: np.ndarray
    master_weights: np.ndarray
    slave_node_ids: np.ndarray
    slave_weights: np.ndarray
    x_slave: np.ndarray
    closest_point: np.ndarray

    @property
    def active(self) -> bool:
        return self.g < 0.0


def contact_constraint_from_sample(
    slave_x_current: np.ndarray,
    sample: SurfaceSample,
    master_x_current: np.ndarray,
    master_boundary_faces: np.ndarray,
) -> ContactConstraint:
    """Compute one contact constraint from a slave sample and master candidates."""

    if sample.candidate_face_ids.size == 0:
        raise ValueError("sample has no master candidate faces")

    x_slave = sample.point(slave_x_current)
    result: SurfaceSDFResult = dynamic_surface_sdf(
        x_slave,
        master_x_current,
        master_boundary_faces,
        sample.candidate_face_ids,
    )
    faces = np.asarray(master_boundary_faces, dtype=np.int64)
    master_node_ids = faces[result.face_id]
    return ContactConstraint(
        g=float(result.g),
        n=np.asarray(result.n, dtype=float),
        master_face_id=int(result.face_id),
        master_node_ids=master_node_ids.copy(),
        master_weights=np.asarray(result.w, dtype=float),
        slave_node_ids=sample.node_ids.copy(),
        slave_weights=sample.weights.copy(),
        x_slave=x_slave,
        closest_point=np.asarray(result.p, dtype=float),
    )


def compute_contact_constraints(
    slave_x_current: np.ndarray,
    samples: Iterable[SurfaceSample],
    master_x_current: np.ndarray,
    master_boundary_faces: np.ndarray,
) -> list[ContactConstraint]:
    """Compute constraints for all samples that have supplied candidates."""

    constraints: list[ContactConstraint] = []
    for sample in samples:
        if sample.candidate_face_ids.size == 0:
            continue
        constraints.append(
            contact_constraint_from_sample(
                slave_x_current,
                sample,
                master_x_current,
                master_boundary_faces,
            )
        )
    return constraints
