"""Surface-contact geometry backed by the Lagrangian SDF oracle."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sfc.contact.lagrangian_sdf_oracle import LagrangianSDFContactOracle
from sfc.fem.calculix_aligned import ContactSample
from sfc.sdf.material_sdf import MaterialSDF


_TRI3_BARY = np.asarray(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=float,
)


@dataclass(slots=True)
class LagrangianSDFSurfaceContactGeometry:
    """TET4 triangular surface quadrature against a Lagrangian-SDF master.

    The geometry consumes a concatenated two-body current coordinate array.  The
    slave samples are evaluated on triangular boundary faces; master-side
    sensitivity is supplied by the closest-feature payload returned by
    :class:`LagrangianSDFContactOracle`.
    """

    slave_faces: np.ndarray
    master_material: MaterialSDF
    master_reference_nodes: np.ndarray
    pressure_stiffness: float
    slave_node_offset: int = 0
    master_node_offset: int = 0
    quadrature: str = "tri3"
    search_radius: float | None = None
    patch_cell_size: float | None = None
    _oracle: LagrangianSDFContactOracle = field(init=False, repr=False)

    def __post_init__(self) -> None:
        faces = np.asarray(self.slave_faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("slave_faces must have shape (n_faces, 3)")
        if np.any(faces < 0):
            raise ValueError("slave_faces cannot contain negative node ids")
        master_nodes = np.asarray(self.master_reference_nodes, dtype=float)
        if master_nodes.ndim != 2 or master_nodes.shape[1] != 3:
            raise ValueError("master_reference_nodes must have shape (n_nodes, 3)")
        if float(self.pressure_stiffness) <= 0.0:
            raise ValueError("pressure_stiffness must be positive")
        if self.quadrature not in {"centroid", "tri3"}:
            raise ValueError("quadrature must be 'centroid' or 'tri3'")
        self.slave_faces = faces
        self.master_reference_nodes = master_nodes
        self._oracle = LagrangianSDFContactOracle(
            self.master_material,
            master_nodes.copy(),
            search_radius=self.search_radius,
            patch_cell_size=self.patch_cell_size,
        )

    def samples(self, x_current: np.ndarray):
        X = np.asarray(x_current, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("x_current must have shape (n_nodes, 3)")
        master_start = int(self.master_node_offset)
        master_stop = master_start + self.master_reference_nodes.shape[0]
        if master_stop > X.shape[0]:
            raise ValueError("master_node_offset places master nodes outside x_current")
        master_x = X[master_start:master_stop]
        self._oracle.refit(master_x)
        barycentric = np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        weight_scale = np.asarray([1.0], dtype=float)
        if self.quadrature == "tri3":
            barycentric = _TRI3_BARY
            weight_scale = np.full(3, 1.0 / 3.0, dtype=float)

        for face_id, face in enumerate(self.slave_faces):
            global_face = face + int(self.slave_node_offset)
            tri = X[global_face]
            area = _triangle_area(tri)
            if area <= 0.0:
                continue
            for qp, (weights, scale) in enumerate(zip(barycentric, weight_scale, strict=True)):
                point = weights @ tri
                query = self._oracle.query(point, cache_key=(int(face_id), int(qp)))
                yield ContactSample(
                    node_ids=global_face.copy(),
                    shape_weights=weights.copy(),
                    gap=float(query.gap),
                    normal=query.normal.copy(),
                    area=float(area * scale),
                    stiffness=float(self.pressure_stiffness),
                    master_node_ids=query.master_node_ids.astype(np.int64) + master_start,
                    master_shape_weights=query.master_weights.copy(),
                )


@dataclass(slots=True)
class LagrangianSDFQuadrilateralSurfaceContactGeometry:
    """Quadrilateral slave-surface quadrature against a Lagrangian-SDF master.

    This is the C3D8/H8 surface-to-surface counterpart of
    :class:`LagrangianSDFSurfaceContactGeometry`.  Slave points are evaluated
    with tensor-product Gauss quadrature on Q4 faces; each point queries the
    master Lagrangian SDF oracle for gap, normal, and closest-feature payload.
    """

    slave_quads: np.ndarray
    master_material: MaterialSDF
    master_reference_nodes: np.ndarray
    pressure_stiffness: float
    slave_node_offset: int = 0
    master_node_offset: int = 0
    quadrature_order: int = 2
    search_radius: float | None = None
    patch_cell_size: float | None = None
    _oracle: LagrangianSDFContactOracle = field(init=False, repr=False)

    def __post_init__(self) -> None:
        quads = np.asarray(self.slave_quads, dtype=np.int64)
        if quads.ndim != 2 or quads.shape[1] != 4:
            raise ValueError("slave_quads must have shape (n_faces, 4)")
        if np.any(quads < 0):
            raise ValueError("slave_quads cannot contain negative node ids")
        master_nodes = np.asarray(self.master_reference_nodes, dtype=float)
        if master_nodes.ndim != 2 or master_nodes.shape[1] != 3:
            raise ValueError("master_reference_nodes must have shape (n_nodes, 3)")
        if float(self.pressure_stiffness) <= 0.0:
            raise ValueError("pressure_stiffness must be positive")
        if int(self.quadrature_order) not in {1, 2, 3}:
            raise ValueError("quadrature_order must be 1, 2, or 3")
        self.slave_quads = quads
        self.master_reference_nodes = master_nodes
        self._oracle = LagrangianSDFContactOracle(
            self.master_material,
            master_nodes.copy(),
            search_radius=self.search_radius,
            patch_cell_size=self.patch_cell_size,
        )

    def samples(self, x_current: np.ndarray):
        X = np.asarray(x_current, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("x_current must have shape (n_nodes, 3)")
        master_start = int(self.master_node_offset)
        master_stop = master_start + self.master_reference_nodes.shape[0]
        if master_stop > X.shape[0]:
            raise ValueError("master_node_offset places master nodes outside x_current")
        master_x = X[master_start:master_stop]
        self._oracle.refit(master_x)
        points, weights = np.polynomial.legendre.leggauss(int(self.quadrature_order))
        for face_id, quad in enumerate(self.slave_quads):
            global_quad = quad + int(self.slave_node_offset)
            qx = X[global_quad]
            for a, xi in enumerate(points):
                for b, eta in enumerate(points):
                    shape = _q4_shape_functions(float(xi), float(eta))
                    dshape = _q4_shape_derivatives(float(xi), float(eta))
                    tangent_xi = dshape[:, 0] @ qx
                    tangent_eta = dshape[:, 1] @ qx
                    jac = float(np.linalg.norm(np.cross(tangent_xi, tangent_eta)))
                    if jac <= 0.0:
                        continue
                    point = shape @ qx
                    query = self._oracle.query(point, cache_key=(int(face_id), int(a), int(b)))
                    yield ContactSample(
                        node_ids=global_quad.copy(),
                        shape_weights=shape.copy(),
                        gap=float(query.gap),
                        normal=query.normal.copy(),
                        area=float(jac * float(weights[a]) * float(weights[b])),
                        stiffness=float(self.pressure_stiffness),
                        master_node_ids=query.master_node_ids.astype(np.int64) + master_start,
                        master_shape_weights=query.master_weights.copy(),
                    )


def _triangle_area(tri: np.ndarray) -> float:
    T = np.asarray(tri, dtype=float)
    if T.shape != (3, 3):
        raise ValueError("tri must have shape (3, 3)")
    return 0.5 * float(np.linalg.norm(np.cross(T[1] - T[0], T[2] - T[0])))


def _q4_shape_functions(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.asarray(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def _q4_shape_derivatives(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.asarray(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ],
        dtype=float,
    )
