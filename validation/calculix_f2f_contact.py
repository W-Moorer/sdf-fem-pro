"""Validation-only CalculiX-style C3D4 face-to-face contact samples.

This module is a clean-room alignment layer for external comparisons.  It does
not copy or import CalculiX code.  The goal is to match the scoped C3D4
face-to-face discretization choices observed in the local CalculiX source:

- one integration point at the centroid of each linear triangular slave face;
- current slave-face area as the spring area;
- hard linear pressure-overclosure response;
- a contact-spring-element style diagnostic record.

The mechanics backend still consumes generic ``ContactSample`` objects, so the
same enforcement path can be driven by an analytic plane query or by the SFC
dynamic surface-SDF query.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from sfc.fem.calculix_aligned import ContactSample
from sfc.sdf.dynamic_surface_sdf import dynamic_surface_sdf

CandidateProvider = Callable[[np.ndarray], np.ndarray]

C3D4_FACE_CENTROID_WEIGHTS = np.full(3, 1.0 / 3.0, dtype=float)


@dataclass(frozen=True, slots=True)
class CalculixF2FContactSpring:
    """Diagnostic state for one CalculiX-style contact spring element."""

    slave_face_index: int
    slave_nodes: np.ndarray
    slave_weights: np.ndarray
    master_face_index: int
    master_weights: np.ndarray
    normal: np.ndarray
    spring_area: float
    clearance: float

    @property
    def active(self) -> bool:
        """Return whether the dynamic contact criterion creates the spring."""

        return self.clearance <= 0.0


@dataclass(frozen=True, slots=True)
class CalculixC3D4FaceToFacePlaneContactGeometry:
    """Rigid-plane C3D4 face-to-face contact with one point per slave face."""

    faces: np.ndarray
    plane_z: float
    stiffness: float
    normal: np.ndarray | None = None

    def contact_springs(self, x_current: np.ndarray) -> list[CalculixF2FContactSpring]:
        """Return the current CalculiX-style spring diagnostics."""

        x = _validate_points(x_current)
        faces = _validate_faces(self.faces, x.shape[0])
        normal = _unit_normal(np.asarray([0.0, 0.0, 1.0] if self.normal is None else self.normal, dtype=float))
        plane_point = np.asarray([0.0, 0.0, float(self.plane_z)], dtype=float)
        springs: list[CalculixF2FContactSpring] = []
        for face_index, face in enumerate(faces):
            tri = x[face]
            area = _triangle_area(tri)
            if area <= 0.0:
                continue
            point = C3D4_FACE_CENTROID_WEIGHTS @ tri
            clearance = float((point - plane_point) @ normal)
            springs.append(
                CalculixF2FContactSpring(
                    slave_face_index=face_index,
                    slave_nodes=np.asarray(face, dtype=np.int64),
                    slave_weights=C3D4_FACE_CENTROID_WEIGHTS.copy(),
                    master_face_index=-1,
                    master_weights=np.zeros(0, dtype=float),
                    normal=normal.copy(),
                    spring_area=area,
                    clearance=clearance,
                )
            )
        return springs

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        """Yield one hard-linear penalty sample per slave face centroid."""

        for spring in self.contact_springs(x_current):
            yield _sample_from_spring(spring, self.stiffness)


@dataclass(frozen=True, slots=True)
class CalculixC3D4FaceToFaceSDFContactGeometry:
    """C3D4 one-point slave contact whose master query is dynamic SDF."""

    slave_faces: np.ndarray
    master_x_current: np.ndarray
    master_faces: np.ndarray
    candidate_provider: CandidateProvider
    stiffness: float

    def contact_springs(self, x_current: np.ndarray) -> list[CalculixF2FContactSpring]:
        """Return current spring diagnostics using SDF closest-point queries."""

        slave_x = _validate_points(x_current)
        slave_faces = _validate_faces(self.slave_faces, slave_x.shape[0])
        master_x = _validate_points(self.master_x_current)
        master_faces = _validate_faces(self.master_faces, master_x.shape[0])
        springs: list[CalculixF2FContactSpring] = []
        for face_index, face in enumerate(slave_faces):
            tri = slave_x[face]
            area = _triangle_area(tri)
            if area <= 0.0:
                continue
            point = C3D4_FACE_CENTROID_WEIGHTS @ tri
            candidates = np.asarray(self.candidate_provider(point), dtype=np.int64).ravel()
            result = dynamic_surface_sdf(point, master_x, master_faces, candidates)
            springs.append(
                CalculixF2FContactSpring(
                    slave_face_index=face_index,
                    slave_nodes=np.asarray(face, dtype=np.int64),
                    slave_weights=C3D4_FACE_CENTROID_WEIGHTS.copy(),
                    master_face_index=int(result.face_id),
                    master_weights=np.asarray(result.w, dtype=float),
                    normal=_unit_normal(np.asarray(result.n, dtype=float)),
                    spring_area=area,
                    clearance=float(result.g),
                )
            )
        return springs

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        """Yield one hard-linear penalty sample per slave face centroid."""

        for spring in self.contact_springs(x_current):
            yield _sample_from_spring(spring, self.stiffness)


def active_contact_spring_count(geometry: object, x_current: np.ndarray) -> int:
    """Return a CalculiX-like active contact spring count for diagnostics."""

    if not hasattr(geometry, "contact_springs"):
        return -1
    springs = getattr(geometry, "contact_springs")(x_current)
    return int(sum(1 for spring in springs if spring.active))


def _sample_from_spring(spring: CalculixF2FContactSpring, stiffness: float) -> ContactSample:
    return ContactSample(
        node_ids=np.asarray(spring.slave_nodes, dtype=np.int64),
        shape_weights=np.asarray(spring.slave_weights, dtype=float),
        gap=float(spring.clearance),
        normal=np.asarray(spring.normal, dtype=float),
        area=float(spring.spring_area),
        stiffness=float(stiffness),
    )


def _validate_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (n, 3)")
    return arr


def _validate_faces(faces: np.ndarray, n_nodes: int) -> np.ndarray:
    arr = np.asarray(faces, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("faces must have shape (m, 3)")
    if np.any(arr < 0) or (arr.size and int(arr.max()) >= int(n_nodes)):
        raise ValueError("faces reference nodes outside the coordinate array")
    return arr


def _triangle_area(tri: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))


def _unit_normal(normal: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("contact normal must be nonzero")
    return np.asarray(normal, dtype=float) / norm
