"""Core mesh topology data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def signed_tet4_jacobian_determinants(X: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Return signed Jacobian determinants for TET4 elements."""

    coords = X[elements]
    jacs = np.stack(
        (
            coords[:, 1] - coords[:, 0],
            coords[:, 2] - coords[:, 0],
            coords[:, 3] - coords[:, 0],
        ),
        axis=2,
    )
    return np.linalg.det(jacs)


def orient_tet4_connectivity(X: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Return connectivity with all TET4 elements in positive orientation."""

    oriented = np.asarray(elements, dtype=np.int64).copy()
    if oriented.size == 0:
        return oriented

    dets = signed_tet4_jacobian_determinants(X, oriented)
    if np.any(np.isclose(dets, 0.0)):
        raise ValueError("tet4 element has non-positive volume")

    negative = dets < 0.0
    if np.any(negative):
        oriented[negative] = oriented[negative][:, [0, 2, 1, 3]]
    return oriented


@dataclass(slots=True)
class VolumeMesh:
    """Reference-domain volume mesh.

    The FEM assembler currently supports only ``tet4``. The mesh container also
    accepts ``hex8`` so validation-only current-surface distance queries can be
    built from C3D8/HEX8 element boundaries without tying the dynamic SDF to
    tetrahedral topology. Node and face sets store integer indices into the
    mesh arrays and are copied to NumPy arrays on construction.
    """

    X: np.ndarray
    elements: np.ndarray
    element_type: str = "tet4"
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)
    face_sets: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        element_type = self.element_type.lower()
        if element_type not in {"tet4", "hex8"}:
            raise ValueError("element_type must be 'tet4' or 'hex8'")

        X = np.asarray(self.X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("X must have shape (n, 3)")

        elements = np.asarray(self.elements, dtype=np.int64)
        nodes_per_element = 4 if element_type == "tet4" else 8
        if elements.ndim != 2 or elements.shape[1] != nodes_per_element:
            raise ValueError(f"elements must have shape (m, {nodes_per_element}) for {element_type} meshes")
        if np.any(elements < 0):
            raise ValueError("elements cannot contain negative node indices")
        if elements.size and int(elements.max()) >= X.shape[0]:
            raise ValueError("elements reference node indices outside X")
        if elements.size and np.any(np.diff(np.sort(elements, axis=1), axis=1) == 0):
            raise ValueError(f"each {element_type} element must reference {nodes_per_element} distinct nodes")

        self.X = X
        self.elements = orient_tet4_connectivity(X, elements) if element_type == "tet4" else elements
        self.element_type = element_type
        self.node_sets = {
            str(name): np.asarray(indices, dtype=np.int64)
            for name, indices in self.node_sets.items()
        }
        self.face_sets = {
            str(name): np.asarray(indices, dtype=np.int64)
            for name, indices in self.face_sets.items()
        }
