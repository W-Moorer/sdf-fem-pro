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
from scipy.sparse import coo_matrix, csr_matrix, issparse


_AXIS_TO_COMPONENT = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


def rotation_matrix_from_vector(rotation: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return the finite-rotation matrix for an axis-angle vector.

    ``rotation`` is interpreted as the rotation vector whose direction is the
    rotation axis and whose norm is the angle in radians.  The implementation
    uses Rodrigues' formula and is independent of Abaqus; it is the finite
    counterpart to the small-rotation hub helper below.
    """

    theta = np.asarray(rotation, dtype=float).reshape(-1)
    if theta.shape != (3,):
        raise ValueError("rotation must have shape (3,)")
    angle = float(np.linalg.norm(theta))
    if angle <= 0.0:
        return np.eye(3, dtype=float)
    axis = theta / angle
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


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


@dataclass(frozen=True, slots=True)
class FiniteRotationRigidHubMPC:
    """Finite-rotation rigid hub motion tied to a reference point.

    The current hub-node position is

    ``x_i = X_rp + u_rp + R(theta) (X_i - X_rp)``.

    This helper is the kinematic foundation needed for full gear-like hub/RP
    motion.  It returns ordinary nodal displacement values so existing
    eliminated-Dirichlet solvers can use it immediately; a reduced RP-DOF
    dynamics layer can later reuse the same transformation.
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

    def current_positions(
        self,
        *,
        translation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        rotation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Return finite-rotation current positions for the hub nodes."""

        u_rp = np.asarray(translation, dtype=float).reshape(-1)
        if u_rp.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        R = rotation_matrix_from_vector(rotation)
        lever = self.reference_nodes[self.node_ids] - self.reference_point[None, :]
        return self.reference_point[None, :] + u_rp[None, :] + lever @ R.T

    def nodal_displacements(
        self,
        *,
        translation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        rotation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Return finite-rotation prescribed displacements for hub nodes."""

        return self.current_positions(translation=translation, rotation=rotation) - self.reference_nodes[self.node_ids]

    def nodal_velocities(
        self,
        *,
        translation_rate: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        rotation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Return hub-node velocities for the current finite orientation."""

        v_rp = np.asarray(translation_rate, dtype=float).reshape(-1)
        omega = np.asarray(angular_velocity, dtype=float).reshape(-1)
        if v_rp.shape != (3,) or omega.shape != (3,):
            raise ValueError("translation_rate and angular_velocity must have shape (3,)")
        current_lever = self.current_positions(translation=(0.0, 0.0, 0.0), rotation=rotation) - self.reference_point[None, :]
        return v_rp[None, :] + np.cross(omega[None, :], current_lever)

    def dirichlet_dofs_and_values(
        self,
        *,
        translation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        rotation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        components: str | Sequence[str | int] = "xyz",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return global DOF indices and finite-rotation prescribed values."""

        comp = _normalize_components(components)
        disp = self.nodal_displacements(translation=translation, rotation=rotation)
        dofs = self.node_ids[:, None] * 3 + comp[None, :]
        values = disp[:, comp]
        order = np.argsort(dofs.ravel(), kind="stable")
        return dofs.ravel()[order].astype(np.int64), values.ravel()[order].astype(float)


@dataclass(frozen=True, slots=True)
class RigidHubReducedAssembly:
    """Sparse reduced-DOF assembly map for Abaqus-style BEAM hub MPCs.

    The transformation ``u_full = T q_reduced`` keeps ordinary translational
    DOFs for non-hub nodes and replaces each rigid hub by six RP DOFs:

    ``u_i = u_rp + theta x (X_i - X_rp)``.

    It is linearized about the reference configuration, matching the
    small-rotation BEAM-MPC form used for global matrix projection.  Large
    rotations can still be advanced incrementally by rebuilding or updating the
    tangent frame at a higher level.
    """

    reference_nodes: np.ndarray
    hubs: tuple[RigidHubMPC, ...]
    free_node_ids: np.ndarray
    hub_column_offsets: np.ndarray
    transformation: csr_matrix

    @property
    def n_full_dofs(self) -> int:
        """Number of full translational nodal DOFs."""

        return int(self.transformation.shape[0])

    @property
    def n_reduced_dofs(self) -> int:
        """Number of reduced DOFs."""

        return int(self.transformation.shape[1])

    def hub_slice(self, hub_index: int = 0) -> slice:
        """Return the six reduced columns for one hub RP."""

        index = int(hub_index)
        if index < 0 or index >= len(self.hubs):
            raise IndexError("hub_index out of range")
        start = int(self.hub_column_offsets[index])
        return slice(start, start + 6)

    def expand_vector(self, reduced_vector: np.ndarray | Sequence[float]) -> np.ndarray:
        """Expand a reduced vector to full flattened nodal DOFs."""

        q = np.asarray(reduced_vector, dtype=float).reshape(-1)
        if q.shape != (self.n_reduced_dofs,):
            raise ValueError("reduced_vector length must match n_reduced_dofs")
        return np.asarray(self.transformation @ q, dtype=float).reshape(-1)

    def expand_displacements(self, reduced_vector: np.ndarray | Sequence[float]) -> np.ndarray:
        """Expand a reduced displacement vector to ``(n_nodes, 3)``."""

        return self.expand_vector(reduced_vector).reshape((-1, 3))

    def current_positions(self, reduced_vector: np.ndarray | Sequence[float]) -> np.ndarray:
        """Return current nodal positions induced by a reduced displacement."""

        return self.reference_nodes + self.expand_displacements(reduced_vector)

    def reduce_vector(self, full_vector: np.ndarray | Sequence[float]) -> np.ndarray:
        """Project a full vector into reduced coordinates via ``T.T f``."""

        f = np.asarray(full_vector, dtype=float)
        if f.shape == self.reference_nodes.shape:
            f = f.reshape(-1)
        else:
            f = f.reshape(-1)
        if f.shape != (self.n_full_dofs,):
            raise ValueError("full_vector length must match n_full_dofs")
        return np.asarray(self.transformation.T @ f, dtype=float).reshape(-1)

    def reduce_matrix(self, full_matrix) -> csr_matrix:
        """Project a full matrix into reduced coordinates via ``T.T A T``."""

        if getattr(full_matrix, "shape", None) != (self.n_full_dofs, self.n_full_dofs):
            raise ValueError("full_matrix must have shape (n_full_dofs, n_full_dofs)")
        A = full_matrix.tocsr() if issparse(full_matrix) else csr_matrix(np.asarray(full_matrix, dtype=float))
        return (self.transformation.T @ A @ self.transformation).tocsr()


def build_rigid_hub_reduced_assembly(
    reference_nodes: np.ndarray,
    hubs: Sequence[RigidHubMPC],
    *,
    include_free_nodes: bool = True,
) -> RigidHubReducedAssembly:
    """Build a sparse global reduced assembly for one or more rigid hubs."""

    nodes = np.asarray(reference_nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("reference_nodes must have shape (n_nodes, 3)")
    hub_tuple = tuple(hubs)
    if not hub_tuple:
        raise ValueError("at least one hub is required")
    occupied: dict[int, int] = {}
    normalized_hubs: list[RigidHubMPC] = []
    for hub_index, hub in enumerate(hub_tuple):
        normalized = RigidHubMPC(hub.node_ids, nodes, hub.reference_point)
        for node in normalized.node_ids:
            key = int(node)
            if key in occupied:
                raise ValueError(f"node {key} belongs to multiple rigid hubs")
            occupied[key] = hub_index
        normalized_hubs.append(normalized)
    all_nodes = np.arange(nodes.shape[0], dtype=np.int64)
    if include_free_nodes:
        free_nodes = np.asarray([node for node in all_nodes if int(node) not in occupied], dtype=np.int64)
    else:
        free_nodes = np.empty(0, dtype=np.int64)
    n_full = 3 * nodes.shape[0]
    free_cols = 3 * free_nodes.size
    hub_offsets = free_cols + 6 * np.arange(len(normalized_hubs), dtype=np.int64)
    n_reduced = free_cols + 6 * len(normalized_hubs)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for free_index, node in enumerate(free_nodes):
        for axis in range(3):
            rows.append(3 * int(node) + axis)
            cols.append(3 * free_index + axis)
            data.append(1.0)
    for hub_index, hub in enumerate(normalized_hubs):
        col0 = int(hub_offsets[hub_index])
        for node in hub.node_ids:
            lever = nodes[int(node)] - hub.reference_point
            block = _small_rotation_hub_block(lever)
            for axis in range(3):
                full_row = 3 * int(node) + axis
                for local_col in range(6):
                    value = float(block[axis, local_col])
                    if value != 0.0:
                        rows.append(full_row)
                        cols.append(col0 + local_col)
                        data.append(value)
    T = coo_matrix((data, (rows, cols)), shape=(n_full, n_reduced), dtype=float).tocsr()
    return RigidHubReducedAssembly(
        reference_nodes=nodes,
        hubs=tuple(normalized_hubs),
        free_node_ids=free_nodes,
        hub_column_offsets=hub_offsets,
        transformation=T,
    )


def reduced_hub_rotational_inertia(
    node_ids: np.ndarray,
    reference_nodes: np.ndarray,
    reference_point: np.ndarray,
    mass_matrix,
) -> np.ndarray:
    """Return the 3x3 RP rotational inertia induced by translational mass.

    The generalized rotational inertia is computed from kinetic energy:

    ``I_ij = v_i^T M v_j``

    where ``v_i`` is the nodal velocity field generated by a unit angular
    velocity about RP axis ``i``.  This works for diagonal, consistent, and
    sparse mass matrices and is the reduced-DOF counterpart needed for RP
    torque loading.
    """

    ids = np.asarray(node_ids, dtype=np.int64).ravel()
    nodes = np.asarray(reference_nodes, dtype=float)
    rp = np.asarray(reference_point, dtype=float).reshape(-1)
    if ids.size == 0:
        raise ValueError("node_ids must not be empty")
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("reference_nodes must have shape (n_nodes, 3)")
    if rp.shape != (3,):
        raise ValueError("reference_point must have shape (3,)")
    if np.any(ids < 0) or int(ids.max()) >= nodes.shape[0]:
        raise ValueError("node_ids reference a node outside reference_nodes")
    n_dofs = 3 * nodes.shape[0]
    if getattr(mass_matrix, "shape", None) != (n_dofs, n_dofs):
        raise ValueError("mass_matrix must have shape (3*n_nodes, 3*n_nodes)")
    basis = np.eye(3, dtype=float)
    velocity_fields = []
    lever = nodes[ids] - rp[None, :]
    for axis in basis:
        velocity = np.zeros((nodes.shape[0], 3), dtype=float)
        velocity[ids] = np.cross(axis[None, :], lever)
        velocity_fields.append(velocity.reshape(-1))
    mass_array = None if issparse(mass_matrix) else np.asarray(mass_matrix, dtype=float)
    mass_velocity_fields = [
        mass_matrix @ velocity if mass_array is None else mass_array @ velocity
        for velocity in velocity_fields
    ]
    inertia = np.empty((3, 3), dtype=float)
    for i, vi in enumerate(velocity_fields):
        for j, Mvj in enumerate(mass_velocity_fields):
            inertia[i, j] = float(vi @ Mvj)
    return 0.5 * (inertia + inertia.T)


def constant_torque_rotation_history(
    inertia: np.ndarray,
    torque: np.ndarray | Sequence[float],
    *,
    duration: float,
    dt: float,
    active_axes: Sequence[int] | np.ndarray = (0, 1, 2),
    initial_rotation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    initial_angular_velocity: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate free RP rotation under constant torque exactly in time.

    This is the no-stiffness reduced rotational DOF response used for the
    torque-coupling alignment layer.  It preserves the requested time
    increments and returns ``(times, rotations, angular_velocities)``.
    """

    I = np.asarray(inertia, dtype=float)
    tau = np.asarray(torque, dtype=float).reshape(-1)
    theta0 = np.asarray(initial_rotation, dtype=float).reshape(-1)
    omega0 = np.asarray(initial_angular_velocity, dtype=float).reshape(-1)
    if I.shape != (3, 3):
        raise ValueError("inertia must have shape (3, 3)")
    if tau.shape != (3,) or theta0.shape != (3,) or omega0.shape != (3,):
        raise ValueError("torque, initial_rotation, and initial_angular_velocity must have shape (3,)")
    if float(duration) < 0.0 or float(dt) <= 0.0:
        raise ValueError("duration must be non-negative and dt must be positive")
    axes = np.asarray(active_axes, dtype=np.int64).ravel()
    if axes.size == 0 or np.any(axes < 0) or np.any(axes > 2):
        raise ValueError("active_axes must contain one or more axes in {0, 1, 2}")
    axes = np.unique(axes)
    steps = int(round(float(duration) / float(dt)))
    if not np.isclose(steps * float(dt), float(duration), rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("duration must be an integer multiple of dt")
    alpha = np.zeros(3, dtype=float)
    alpha_active = np.linalg.solve(I[axes[:, None], axes], tau[axes])
    alpha[axes] = alpha_active
    times = np.linspace(0.0, float(duration), steps + 1)
    rotations = theta0[None, :] + times[:, None] * omega0[None, :] + 0.5 * (times[:, None] ** 2) * alpha[None, :]
    angular_velocities = omega0[None, :] + times[:, None] * alpha[None, :]
    return times, rotations, angular_velocities


def constant_angular_velocity_rotation_history(
    angular_velocity: np.ndarray | Sequence[float],
    *,
    duration: float,
    dt: float,
    initial_rotation: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the RP rotation history for prescribed constant angular velocity."""

    omega = np.asarray(angular_velocity, dtype=float).reshape(-1)
    theta0 = np.asarray(initial_rotation, dtype=float).reshape(-1)
    if omega.shape != (3,) or theta0.shape != (3,):
        raise ValueError("angular_velocity and initial_rotation must have shape (3,)")
    if float(duration) < 0.0 or float(dt) <= 0.0:
        raise ValueError("duration must be non-negative and dt must be positive")
    steps = int(round(float(duration) / float(dt)))
    if not np.isclose(steps * float(dt), float(duration), rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("duration must be an integer multiple of dt")
    times = np.linspace(0.0, float(duration), steps + 1)
    rotations = theta0[None, :] + times[:, None] * omega[None, :]
    angular_velocities = np.repeat(omega[None, :], steps + 1, axis=0)
    return times, rotations, angular_velocities


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


def _small_rotation_hub_block(lever: np.ndarray) -> np.ndarray:
    r = np.asarray(lever, dtype=float).reshape(-1)
    if r.shape != (3,):
        raise ValueError("lever must have shape (3,)")
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, r[2], -r[1]],
            [0.0, 1.0, 0.0, -r[2], 0.0, r[0]],
            [0.0, 0.0, 1.0, r[1], -r[0], 0.0],
        ],
        dtype=float,
    )


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
