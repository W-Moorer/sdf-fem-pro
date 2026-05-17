"""Legacy projection-backed contact-geometry adapter.

The paper-facing contact path is the dynamic narrow-band SDF field in
``sfc.contact.field_contact``. This adapter is retained for validation and
backward-compatible projection-query comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from sfc.fem.calculix_aligned import (
    TRIANGLE_QUADRATURE_BARYCENTRIC,
    TRIANGLE_QUADRATURE_WEIGHTS,
    ContactSample,
)
from sfc.sdf.dynamic_surface_sdf import dynamic_surface_sdf

CandidateProvider = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class DynamicSurfaceSDFContactGeometry:
    """Slave-face quadrature contact geometry using projection-kernel queries.

    The mechanics backend consumes only :class:`ContactSample` objects. This
    adapter keeps older validation paths available: final gap and normal
    evaluation are computed from the current master FEM surface by local
    projection. It is not the main dynamic SDF field-contact API.

    ``candidate_provider`` is required. It should be supplied by the broad
    phase and must return candidate master face ids for the query point. The
    adapter intentionally does not fall back to a global all-face search.
    """

    slave_faces: np.ndarray
    master_x_current: np.ndarray
    master_faces: np.ndarray
    candidate_provider: CandidateProvider
    stiffness: float

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        """Yield active/inactive slave quadrature contact samples."""

        slave_x = np.asarray(x_current, dtype=float)
        if slave_x.ndim != 2 or slave_x.shape[1] != 3:
            raise ValueError("x_current must have shape (n, 3)")
        slave_faces = np.asarray(self.slave_faces, dtype=np.int64)
        if slave_faces.ndim != 2 or slave_faces.shape[1] != 3:
            raise ValueError("slave_faces must have shape (m, 3)")
        if np.any(slave_faces < 0) or (slave_faces.size and int(slave_faces.max()) >= slave_x.shape[0]):
            raise ValueError("slave_faces reference nodes outside x_current")

        master_x = np.asarray(self.master_x_current, dtype=float)
        master_faces = np.asarray(self.master_faces, dtype=np.int64)
        bary = TRIANGLE_QUADRATURE_BARYCENTRIC
        weights = TRIANGLE_QUADRATURE_WEIGHTS
        for face in slave_faces:
            tri = slave_x[face]
            area = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
            if area <= 0.0:
                continue
            for shape, q_weight in zip(bary, weights, strict=True):
                point = shape @ tri
                candidates = np.asarray(self.candidate_provider(point), dtype=np.int64).ravel()
                result = dynamic_surface_sdf(point, master_x, master_faces, candidates)
                yield ContactSample(
                    node_ids=np.asarray(face, dtype=np.int64),
                    shape_weights=np.asarray(shape, dtype=float),
                    gap=float(result.g),
                    normal=np.asarray(result.n, dtype=float),
                    area=area * float(q_weight),
                    stiffness=float(self.stiffness),
                )
