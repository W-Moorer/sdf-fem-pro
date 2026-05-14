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
from scipy.sparse import coo_matrix, csr_matrix

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
    master_nodes: np.ndarray
    master_weights: np.ndarray
    normal: np.ndarray
    spring_area: float
    clearance: float
    master_element_node_count: int | None = None

    @property
    def active(self) -> bool:
        """Return whether the dynamic contact criterion creates the spring."""

        return self.clearance <= 0.0

    @property
    def calculix_cnum_weight(self) -> int:
        """Return the CalculiX ``CNUM`` contribution for this spring record.

        CalculiX reports ``CNUM`` from its generated contact-element storage,
        not from the number of force-producing integration points.  For the
        scoped C3D4 slave face against an S4 rigid-plane master used in the
        validation input, one force-producing record contributes
        ``nopes + nopem = 3 + 4`` to the printed contact-element count.
        """

        slave_count = int(np.asarray(self.slave_nodes, dtype=np.int64).size)
        if self.master_element_node_count is not None:
            master_count = int(self.master_element_node_count)
        else:
            master_nodes = np.asarray(self.master_nodes, dtype=np.int64)
            master_count = int(master_nodes.size) if master_nodes.size else 4
        return slave_count + master_count


@dataclass(frozen=True, slots=True)
class CalculixF2FLifecycleEvent:
    """One spring lifecycle event for diagnostics."""

    slave_face_index: int
    status: str
    clearance: float
    was_active: bool
    is_active: bool


@dataclass(frozen=True, slots=True)
class ContactConvergenceRecord:
    """Contact active-set convergence diagnostic for one iteration."""

    iteration: int
    active_count: int
    residual_norm: float
    recommended_cutback: bool
    reason: str


@dataclass(slots=True)
class CalculixF2FContactLifecycle:
    """Clean-room persistent contact spring lifecycle for validation.

    The lifecycle stores contact spring elements by slave face.  It approximates
    the bookkeeping concept used by CalculiX contact elements without copying
    source-level branch logic.  A previously active spring may be retained up to
    ``release_tolerance`` so validation code can distinguish generated contact
    elements from force-producing negative-clearance samples.
    """

    activation_tolerance: float = 0.0
    release_tolerance: float = 0.0
    keep_previous_on_cutback: bool = True
    active_springs: dict[int, CalculixF2FContactSpring] | None = None
    events: list[CalculixF2FLifecycleEvent] | None = None

    def __post_init__(self) -> None:
        if self.active_springs is None:
            self.active_springs = {}
        if self.events is None:
            self.events = []

    def update(
        self,
        candidates: Iterable[CalculixF2FContactSpring],
        *,
        cutback: bool = False,
    ) -> list[CalculixF2FContactSpring]:
        """Update generated contact springs and return current generated set."""

        previous = dict(self.active_springs or {})
        next_active: dict[int, CalculixF2FContactSpring] = {}
        seen: set[int] = set()
        events: list[CalculixF2FLifecycleEvent] = []
        for spring in candidates:
            key = int(spring.slave_face_index)
            seen.add(key)
            was_active = key in previous
            penetrates = spring.clearance <= float(self.activation_tolerance)
            persists = was_active and spring.clearance <= float(self.release_tolerance)
            cutback_persists = bool(cutback and self.keep_previous_on_cutback and was_active)
            is_active = bool(penetrates or persists or cutback_persists)
            if is_active:
                next_active[key] = spring
                if penetrates and not was_active:
                    status = "generated"
                elif cutback_persists and not penetrates and not persists:
                    status = "cutback_persisted"
                elif was_active:
                    status = "persisted"
                else:
                    status = "generated"
            else:
                status = "released" if was_active else "inactive"
            events.append(
                CalculixF2FLifecycleEvent(
                    slave_face_index=key,
                    status=status,
                    clearance=float(spring.clearance),
                    was_active=was_active,
                    is_active=is_active,
                )
            )

        for key, spring in previous.items():
            if key in seen:
                continue
            events.append(
                CalculixF2FLifecycleEvent(
                    slave_face_index=key,
                    status="lost_candidate_released",
                    clearance=float(spring.clearance),
                    was_active=True,
                    is_active=False,
                )
            )

        self.active_springs = next_active
        self.events = events
        return list(next_active.values())

    def snapshot(self) -> tuple[dict[int, CalculixF2FContactSpring], list[CalculixF2FLifecycleEvent]]:
        """Return a rollback snapshot of generated springs and events."""

        return dict(self.active_springs or {}), list(self.events or [])

    def restore(self, snapshot: tuple[dict[int, CalculixF2FContactSpring], list[CalculixF2FLifecycleEvent]]) -> None:
        """Restore a rollback snapshot."""

        springs, events = snapshot
        self.active_springs = dict(springs)
        self.events = list(events)

    @property
    def generated_count(self) -> int:
        """Return number of currently generated force spring records."""

        return len(self.active_springs or {})

    @property
    def calculix_contact_element_count(self) -> int:
        """Return the CalculiX ``CNUM``-equivalent generated count."""

        return int(sum(spring.calculix_cnum_weight for spring in (self.active_springs or {}).values()))

    @property
    def penetrating_count(self) -> int:
        """Return number of generated springs with negative clearance."""

        return int(sum(1 for spring in (self.active_springs or {}).values() if spring.clearance < 0.0))


@dataclass(slots=True)
class CalculixContactConvergenceHeuristic:
    """Small contact active-set convergence/cutback diagnostic.

    This is not a clone of CalculiX's full convergence code.  It captures the
    behavior needed by the validation runner: stable active sets with decreasing
    residuals are accepted, while active-set oscillation or increasing residuals
    request a cutback/retry.
    """

    residual_growth_factor: float = 1.05
    history: list[ContactConvergenceRecord] | None = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []

    def update(self, *, iteration: int, active_count: int, residual_norm: float) -> ContactConvergenceRecord:
        """Record an iteration and return the cutback recommendation."""

        residual = float(residual_norm)
        count = int(active_count)
        reason = "stable_or_improving"
        cutback = False
        previous = self.history[-1] if self.history else None
        if (
            previous is not None
            and float(previous.residual_norm) > 1.0e-14
            and residual > float(previous.residual_norm) * float(self.residual_growth_factor)
        ):
            cutback = True
            reason = "residual_growth"
        if len(self.history) >= 2:
            a = self.history[-2].active_count
            b = self.history[-1].active_count
            if a == count and b != count:
                cutback = True
                reason = "active_set_oscillation"
        record = ContactConvergenceRecord(
            iteration=int(iteration),
            active_count=count,
            residual_norm=residual,
            recommended_cutback=cutback,
            reason=reason,
        )
        self.history.append(record)
        return record


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
                    master_nodes=np.zeros(0, dtype=np.int64),
                    master_weights=np.zeros(0, dtype=float),
                    normal=normal.copy(),
                    spring_area=area,
                    clearance=clearance,
                    master_element_node_count=4,
                )
            )
        return springs

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        """Yield one hard-linear penalty sample per slave face centroid."""

        for spring in self.contact_springs(x_current):
            yield _sample_from_spring(spring, self.stiffness)


@dataclass(slots=True)
class PersistentCalculixC3D4FaceToFacePlaneContactGeometry:
    """Stateful rigid-plane F2F contact with persistent spring bookkeeping."""

    faces: np.ndarray
    plane_z: float
    stiffness: float
    normal: np.ndarray | None = None
    lifecycle: CalculixF2FContactLifecycle | None = None
    cutback_retry: bool = False

    def __post_init__(self) -> None:
        if self.lifecycle is None:
            self.lifecycle = CalculixF2FContactLifecycle()

    @property
    def generated_contact_count(self) -> int:
        """Return CalculiX-style generated contact spring count."""

        return int(self.lifecycle.generated_count if self.lifecycle is not None else 0)

    @property
    def calculix_contact_element_count(self) -> int:
        """Return the ``CNUM``-equivalent generated contact-element count."""

        return int(self.lifecycle.calculix_contact_element_count if self.lifecycle is not None else 0)

    def contact_springs(self, x_current: np.ndarray) -> list[CalculixF2FContactSpring]:
        """Update lifecycle and return generated springs."""

        stateless = CalculixC3D4FaceToFacePlaneContactGeometry(
            self.faces,
            self.plane_z,
            self.stiffness,
            self.normal,
        )
        assert self.lifecycle is not None
        return self.lifecycle.update(stateless.contact_springs(x_current), cutback=self.cutback_retry)

    def set_cutback_retry(self, value: bool) -> None:
        """Set whether this evaluation is part of a cutback retry."""

        self.cutback_retry = bool(value)

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        """Yield samples from generated springs."""

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
    master_element_node_count: int | None = None

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
                    master_nodes=np.asarray(master_faces[int(result.face_id)], dtype=np.int64),
                    master_weights=np.asarray(result.w, dtype=float),
                    normal=_unit_normal(np.asarray(result.n, dtype=float)),
                    spring_area=area,
                    clearance=float(result.g),
                    master_element_node_count=self.master_element_node_count,
                )
            )
        return springs

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        """Yield one hard-linear penalty sample per slave face centroid."""

        for spring in self.contact_springs(x_current):
            yield _sample_from_spring(spring, self.stiffness)


@dataclass(slots=True)
class PersistentCalculixC3D4FaceToFaceSDFContactGeometry:
    """Stateful dynamic-SDF F2F contact with persistent spring bookkeeping."""

    slave_faces: np.ndarray
    master_x_current: np.ndarray
    master_faces: np.ndarray
    candidate_provider: CandidateProvider
    stiffness: float
    master_element_node_count: int | None = None
    lifecycle: CalculixF2FContactLifecycle | None = None
    cutback_retry: bool = False

    def __post_init__(self) -> None:
        if self.lifecycle is None:
            self.lifecycle = CalculixF2FContactLifecycle()

    @property
    def generated_contact_count(self) -> int:
        """Return CalculiX-style generated contact spring count."""

        return int(self.lifecycle.generated_count if self.lifecycle is not None else 0)

    @property
    def calculix_contact_element_count(self) -> int:
        """Return the ``CNUM``-equivalent generated contact-element count."""

        return int(self.lifecycle.calculix_contact_element_count if self.lifecycle is not None else 0)

    def contact_springs(self, x_current: np.ndarray) -> list[CalculixF2FContactSpring]:
        """Update lifecycle and return generated springs."""

        stateless = CalculixC3D4FaceToFaceSDFContactGeometry(
            self.slave_faces,
            self.master_x_current,
            self.master_faces,
            self.candidate_provider,
            self.stiffness,
            self.master_element_node_count,
        )
        candidates = stateless.contact_springs(x_current)
        assert self.lifecycle is not None
        previous = self.lifecycle.active_springs or {}
        if previous:
            candidate_by_face = {int(spring.slave_face_index): spring for spring in candidates}
            stored_candidates: list[CalculixF2FContactSpring] = []
            for face_index, spring in candidate_by_face.items():
                previous_spring = previous.get(face_index)
                if previous_spring is None or previous_spring.master_face_index < 0:
                    stored_candidates.append(spring)
                else:
                    stored_candidates.append(
                        _sdf_spring_from_stored_master_projection(
                            previous_spring,
                            x_current,
                            self.slave_faces,
                            self.master_x_current,
                        )
                    )
            candidates = stored_candidates
        return self.lifecycle.update(candidates, cutback=self.cutback_retry)

    def set_cutback_retry(self, value: bool) -> None:
        """Set whether this evaluation is part of a cutback retry."""

        self.cutback_retry = bool(value)

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        """Yield samples from generated springs."""

        for spring in self.contact_springs(x_current):
            yield _sample_from_spring(spring, self.stiffness)


@dataclass(frozen=True, slots=True)
class DeformableF2FContactResponse:
    """Master/slave force and stiffness for deformable F2F contact springs."""

    force: np.ndarray
    stiffness: csr_matrix
    active_count: int
    generated_count: int
    normal_force: float
    energy: float
    max_penetration: float


def assemble_deformable_f2f_contact_response(
    springs: Iterable[CalculixF2FContactSpring],
    *,
    stiffness: float,
    n_slave_nodes: int,
    n_master_nodes: int,
    slave_dof_offset: int = 0,
    master_dof_offset: int | None = None,
) -> DeformableF2FContactResponse:
    """Assemble deformable-deformable F2F ``J^T lambda`` and ``J^T J``.

    The gap row follows the project convention:

    ``dg/dx_slave = N_slave n^T`` and
    ``dg/dx_master = -N_master n^T``.
    """

    k = float(stiffness)
    if k <= 0.0:
        raise ValueError("stiffness must be positive")
    slave_offset = int(slave_dof_offset)
    master_offset = 3 * int(n_slave_nodes) if master_dof_offset is None else int(master_dof_offset)
    n_total = max(slave_offset + 3 * int(n_slave_nodes), master_offset + 3 * int(n_master_nodes))
    force = np.zeros(n_total, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    active_count = 0
    generated_count = 0
    normal_force = 0.0
    energy = 0.0
    max_pen = 0.0

    for spring in springs:
        generated_count += 1
        penetration = max(-float(spring.clearance), 0.0)
        if penetration <= 0.0:
            continue
        cols_i, vals_i = _spring_jacobian_entries(
            spring,
            slave_dof_offset=slave_offset,
            master_dof_offset=master_offset,
        )
        lam = k * float(spring.spring_area) * penetration
        tangent = k * float(spring.spring_area)
        active_count += 1
        normal_force += lam
        energy += 0.5 * tangent * penetration * penetration
        max_pen = max(max_pen, penetration)
        force[cols_i] += vals_i * lam
        for row_col, row_val in zip(cols_i, vals_i, strict=True):
            for col_col, col_val in zip(cols_i, vals_i, strict=True):
                rows.append(int(row_col))
                cols.append(int(col_col))
                data.append(float(tangent * row_val * col_val))

    K = coo_matrix((data, (rows, cols)), shape=(n_total, n_total)).tocsr()
    return DeformableF2FContactResponse(
        force=force,
        stiffness=K,
        active_count=active_count,
        generated_count=generated_count,
        normal_force=float(normal_force),
        energy=float(energy),
        max_penetration=float(max_pen),
    )


def active_contact_spring_count(geometry: object, x_current: np.ndarray) -> int:
    """Return a CalculiX-like active contact spring count for diagnostics."""

    if not hasattr(geometry, "contact_springs"):
        return -1
    springs = getattr(geometry, "contact_springs")(x_current)
    return int(sum(1 for spring in springs if spring.active))


def generated_contact_spring_count(geometry: object, x_current: np.ndarray | None = None) -> int:
    """Return generated spring count, updating from coordinates if supplied."""

    if hasattr(geometry, "generated_contact_count"):
        return int(getattr(geometry, "generated_contact_count"))
    if x_current is None or not hasattr(geometry, "contact_springs"):
        return -1
    return len(getattr(geometry, "contact_springs")(x_current))


def calculix_equivalent_contact_element_count(geometry: object, x_current: np.ndarray | None = None) -> int:
    """Return a ``CNUM``-equivalent count for validation diagnostics."""

    if hasattr(geometry, "calculix_contact_element_count"):
        return int(getattr(geometry, "calculix_contact_element_count"))
    if x_current is None or not hasattr(geometry, "contact_springs"):
        return -1
    springs = getattr(geometry, "contact_springs")(x_current)
    return int(sum(spring.calculix_cnum_weight for spring in springs if spring.active))


def _sample_from_spring(spring: CalculixF2FContactSpring, stiffness: float) -> ContactSample:
    return ContactSample(
        node_ids=np.asarray(spring.slave_nodes, dtype=np.int64),
        shape_weights=np.asarray(spring.slave_weights, dtype=float),
        gap=float(spring.clearance),
        normal=np.asarray(spring.normal, dtype=float),
        area=float(spring.spring_area),
        stiffness=float(stiffness),
    )


def _spring_jacobian_entries(
    spring: CalculixF2FContactSpring,
    *,
    slave_dof_offset: int,
    master_dof_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    normal = _unit_normal(np.asarray(spring.normal, dtype=float))
    cols: list[int] = []
    vals: list[float] = []
    for node, weight in zip(spring.slave_nodes, spring.slave_weights, strict=True):
        base = int(slave_dof_offset) + 3 * int(node)
        for component in range(3):
            cols.append(base + component)
            vals.append(float(weight) * normal[component])
    master_nodes = np.asarray(spring.master_nodes, dtype=np.int64)
    master_weights = np.asarray(spring.master_weights, dtype=float)
    if spring.master_face_index >= 0 and master_nodes.size and master_weights.size:
        for node, weight in zip(master_nodes, master_weights, strict=True):
            base = int(master_dof_offset) + 3 * int(node)
            for component in range(3):
                cols.append(base + component)
                vals.append(-float(weight) * normal[component])
    return np.asarray(cols, dtype=np.int64), np.asarray(vals, dtype=float)


def _sdf_spring_from_stored_master_projection(
    previous: CalculixF2FContactSpring,
    x_current: np.ndarray,
    slave_faces: np.ndarray,
    master_x_current: np.ndarray,
) -> CalculixF2FContactSpring:
    """Update a persistent SDF spring using CalculiX-style stored projection data.

    CalculiX F2F contact stores the master local coordinates and normal when a
    contact spring is generated, then recomputes clearance from current nodal
    coordinates.  This helper mirrors that behavior for the validation-only
    dynamic-SDF F2F mode: the stored master face/weights/normal are kept, while
    the slave centroid and master projection positions are updated from the
    current coordinates.
    """

    slave_x = _validate_points(x_current)
    faces = _validate_faces(slave_faces, slave_x.shape[0])
    master_x = _validate_points(master_x_current)
    face = faces[int(previous.slave_face_index)]
    tri = slave_x[face]
    area = _triangle_area(tri)
    point = C3D4_FACE_CENTROID_WEIGHTS @ tri
    master_nodes = np.asarray(previous.master_nodes, dtype=np.int64)
    master_weights = np.asarray(previous.master_weights, dtype=float)
    if master_nodes.size == 0 or master_weights.size == 0:
        projection = point.copy()
    else:
        projection = master_weights @ master_x[master_nodes]
    normal = _unit_normal(np.asarray(previous.normal, dtype=float))
    return CalculixF2FContactSpring(
        slave_face_index=int(previous.slave_face_index),
        slave_nodes=np.asarray(face, dtype=np.int64),
        slave_weights=C3D4_FACE_CENTROID_WEIGHTS.copy(),
        master_face_index=int(previous.master_face_index),
        master_nodes=master_nodes.copy(),
        master_weights=master_weights.copy(),
        normal=normal,
        spring_area=float(area),
        clearance=float((point - projection) @ normal),
        master_element_node_count=previous.master_element_node_count,
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
