"""Sparse global assembly for finite-element bodies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .body import DeformableBody
from .elements import ElementBackend, get_element_backend


def _material_parameters(material: Any) -> tuple[float, float]:
    if isinstance(material, Mapping):
        return float(material["E"]), float(material["nu"])
    if isinstance(material, tuple) and len(material) == 2:
        return float(material[0]), float(material[1])

    try:
        return float(material.E), float(material.nu)
    except AttributeError as exc:
        raise TypeError("material must provide E and nu") from exc


def _element_dofs(element: np.ndarray) -> np.ndarray:
    conn = np.asarray(element, dtype=np.int64)
    return np.repeat(conn, 3) * 3 + np.tile(np.arange(3, dtype=np.int64), conn.size)


def _element_backend(body: DeformableBody) -> ElementBackend:
    return get_element_backend(getattr(body.mesh, "element_type", "tet4"))


def _assemble_element_matrices(
    body: DeformableBody,
    local_matrix,
) -> csr_matrix:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    for element in body.mesh.elements:
        dofs = _element_dofs(element)
        Ke = local_matrix(element)
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(rr.ravel())
        cols.append(cc.ravel())
        data.append(Ke.ravel())

    if not data:
        return csr_matrix((body.n_dofs, body.n_dofs), dtype=float)

    return coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(body.n_dofs, body.n_dofs),
    ).tocsr()


def assemble_stiffness_matrix(body: DeformableBody) -> csr_matrix:
    """Assemble the sparse global linear elastic stiffness matrix."""

    backend = _element_backend(body)
    E, nu = _material_parameters(body.material)
    X = body.mesh.X

    def local_matrix(element: np.ndarray) -> np.ndarray:
        return backend.stiffness(X[element], E, nu)

    return _assemble_element_matrices(body, local_matrix)


def assemble_mass_matrix(body: DeformableBody, *, kind: str = "consistent") -> csr_matrix:
    """Assemble the sparse global mass matrix."""

    backend = _element_backend(body)
    X = body.mesh.X

    def local_matrix(element: np.ndarray) -> np.ndarray:
        return backend.mass(X[element], body.density, kind)

    return _assemble_element_matrices(body, local_matrix)


def assemble_gravity_force(
    body: DeformableBody,
    gravity: np.ndarray | tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> np.ndarray:
    """Assemble the global body-force vector for constant gravity."""

    g = np.asarray(gravity, dtype=float)
    if g.shape != (3,):
        raise ValueError("gravity must have shape (3,)")

    F = np.zeros(body.n_dofs, dtype=float)
    backend = _element_backend(body)
    X = body.mesh.X

    for element in body.mesh.elements:
        nodal_force = backend.nodal_body_force(X[element], body.density, g)
        for node in element:
            i = int(node) * 3
            F[i : i + 3] += nodal_force

    return F
