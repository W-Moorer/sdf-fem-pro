"""Finite-element deformation maps for material SDF queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfc.mesh.topology import VolumeMesh

from .hex8 import hex8_natural_gradients, hex8_shape_functions
from .tet4 import tet4_shape_function_gradients


@dataclass(frozen=True, slots=True)
class DeformationMapEvaluation:
    """Evaluation of ``x = chi(X,t)`` at one material point."""

    element_id: int
    node_ids: np.ndarray
    shape_values: np.ndarray
    material_point: np.ndarray
    current_point: np.ndarray
    deformation_gradient: np.ndarray
    natural_coordinates: np.ndarray


@dataclass(frozen=True, slots=True)
class FEMDeformationMap:
    """Isoparametric FEM map from material to current coordinates."""

    mesh: VolumeMesh
    x_current: np.ndarray

    def __post_init__(self) -> None:
        X = np.asarray(self.x_current, dtype=float)
        if X.shape != self.mesh.X.shape:
            raise ValueError("x_current must match mesh.X shape")
        object.__setattr__(self, "x_current", X.copy())

    def with_current(self, x_current: np.ndarray) -> "FEMDeformationMap":
        """Return the same reference mesh with updated current nodal positions."""

        return type(self)(self.mesh, x_current)

    def element_for_face(self, face_node_ids: np.ndarray) -> int | None:
        """Return an element containing all face nodes, if one exists."""

        face = set(int(v) for v in np.asarray(face_node_ids, dtype=np.int64).ravel())
        for element_id, element in enumerate(self.mesh.elements):
            if face.issubset(set(int(v) for v in element)):
                return int(element_id)
        return None

    def evaluate(
        self,
        material_point: np.ndarray,
        *,
        preferred_element: int | None = None,
    ) -> DeformationMapEvaluation:
        """Evaluate ``chi``, shape values, and deformation gradient."""

        X = _as_point(material_point, "material_point")
        element_id = self._find_element(X, preferred_element=preferred_element)
        if self.mesh.element_type == "tet4":
            return self._evaluate_tet4(element_id, X)
        if self.mesh.element_type == "hex8":
            return self._evaluate_hex8(element_id, X)
        raise ValueError(f"unsupported element type {self.mesh.element_type!r}")

    def pull_back_current_point(
        self,
        current_point: np.ndarray,
        *,
        preferred_element: int | None = None,
        initial_material_point: np.ndarray | None = None,
        max_iterations: int = 12,
        tolerance: float = 1.0e-12,
    ) -> DeformationMapEvaluation:
        """Solve ``chi(X,t) ~= current_point`` in a preferred element."""

        x = _as_point(current_point, "current_point")
        if preferred_element is None:
            if initial_material_point is not None:
                preferred_element = self._find_element(_as_point(initial_material_point, "initial_material_point"))
            else:
                preferred_element = 0
        element_id = int(preferred_element)
        if self.mesh.element_type == "tet4":
            return self._pull_back_tet4(element_id, x)
        if self.mesh.element_type == "hex8":
            return self._pull_back_hex8(
                element_id,
                x,
                initial_material_point=initial_material_point,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
        raise ValueError(f"unsupported element type {self.mesh.element_type!r}")

    def _find_element(self, material_point: np.ndarray, *, preferred_element: int | None = None) -> int:
        if preferred_element is not None:
            element_id = int(preferred_element)
            if element_id < 0 or element_id >= self.mesh.elements.shape[0]:
                raise ValueError("preferred_element is outside the mesh")
            if self._element_contains(element_id, material_point, tol=1.0e-7):
                return element_id
        best_id = 0
        best_score = np.inf
        for element_id in range(self.mesh.elements.shape[0]):
            score = self._element_outside_score(element_id, material_point)
            if score < best_score:
                best_score = score
                best_id = int(element_id)
                if score <= 1.0e-12:
                    break
        return best_id

    def _element_contains(self, element_id: int, material_point: np.ndarray, *, tol: float) -> bool:
        return self._element_outside_score(element_id, material_point) <= float(tol)

    def _element_outside_score(self, element_id: int, material_point: np.ndarray) -> float:
        element = self.mesh.elements[int(element_id)]
        X_ref = self.mesh.X[element]
        if self.mesh.element_type == "tet4":
            shape = _tet4_shape_values(X_ref, material_point)
            return float(np.sum(np.maximum(-shape, 0.0)) + max(float(np.sum(shape) - 1.0), 0.0))
        natural = _hex8_reference_inverse(X_ref, material_point)
        return float(np.sum(np.maximum(np.abs(natural) - 1.0, 0.0)))

    def _evaluate_tet4(self, element_id: int, material_point: np.ndarray) -> DeformationMapEvaluation:
        nodes = self.mesh.elements[int(element_id)]
        X_ref = self.mesh.X[nodes]
        x_cur = self.x_current[nodes]
        shape = _tet4_shape_values(X_ref, material_point)
        gradients = tet4_shape_function_gradients(X_ref)
        current = shape @ x_cur
        F = x_cur.T @ gradients
        return DeformationMapEvaluation(
            element_id=int(element_id),
            node_ids=nodes.copy(),
            shape_values=shape,
            material_point=material_point.copy(),
            current_point=current,
            deformation_gradient=F,
            natural_coordinates=shape[1:].copy(),
        )

    def _evaluate_hex8(self, element_id: int, material_point: np.ndarray) -> DeformationMapEvaluation:
        nodes = self.mesh.elements[int(element_id)]
        X_ref = self.mesh.X[nodes]
        x_cur = self.x_current[nodes]
        natural = _hex8_reference_inverse(X_ref, material_point)
        xi, eta, zeta = (float(v) for v in natural)
        shape = hex8_shape_functions(xi, eta, zeta)
        natural_gradients = hex8_natural_gradients(xi, eta, zeta)
        J_ref = X_ref.T @ natural_gradients
        grad_ref = natural_gradients @ np.linalg.inv(J_ref)
        current = shape @ x_cur
        F = x_cur.T @ grad_ref
        return DeformationMapEvaluation(
            element_id=int(element_id),
            node_ids=nodes.copy(),
            shape_values=shape,
            material_point=material_point.copy(),
            current_point=current,
            deformation_gradient=F,
            natural_coordinates=natural,
        )

    def _pull_back_tet4(self, element_id: int, current_point: np.ndarray) -> DeformationMapEvaluation:
        nodes = self.mesh.elements[int(element_id)]
        x_cur = self.x_current[nodes]
        shape = _tet4_shape_values(x_cur, current_point)
        material = shape @ self.mesh.X[nodes]
        return self._evaluate_tet4(element_id, material)

    def _pull_back_hex8(
        self,
        element_id: int,
        current_point: np.ndarray,
        *,
        initial_material_point: np.ndarray | None,
        max_iterations: int,
        tolerance: float,
    ) -> DeformationMapEvaluation:
        nodes = self.mesh.elements[int(element_id)]
        X_ref = self.mesh.X[nodes]
        x_cur = self.x_current[nodes]
        if initial_material_point is None:
            natural = np.zeros(3, dtype=float)
        else:
            natural = _hex8_reference_inverse(X_ref, _as_point(initial_material_point, "initial_material_point"))
        for _ in range(max(1, int(max_iterations))):
            shape = hex8_shape_functions(float(natural[0]), float(natural[1]), float(natural[2]))
            gradients = hex8_natural_gradients(float(natural[0]), float(natural[1]), float(natural[2]))
            residual = shape @ x_cur - current_point
            if float(np.linalg.norm(residual)) <= float(tolerance):
                break
            J_cur = x_cur.T @ gradients
            try:
                delta = np.linalg.solve(J_cur, -residual)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(J_cur, -residual, rcond=None)
            natural = np.clip(natural + delta, -1.5, 1.5)
            if float(np.linalg.norm(delta)) <= float(tolerance):
                break
        material = hex8_shape_functions(float(natural[0]), float(natural[1]), float(natural[2])) @ X_ref
        return self._evaluate_hex8(element_id, material)


def _tet4_shape_values(Xe: np.ndarray, point: np.ndarray) -> np.ndarray:
    X = np.asarray(Xe, dtype=float)
    p = _as_point(point, "point")
    if X.shape != (4, 3):
        raise ValueError("tet4 coordinates must have shape (4, 3)")
    A = np.column_stack((X[1] - X[0], X[2] - X[0], X[3] - X[0]))
    try:
        local = np.linalg.solve(A, p - X[0])
    except np.linalg.LinAlgError:
        local, *_ = np.linalg.lstsq(A, p - X[0], rcond=None)
    return np.asarray([1.0 - float(np.sum(local)), local[0], local[1], local[2]], dtype=float)


def _hex8_reference_inverse(Xe: np.ndarray, material_point: np.ndarray) -> np.ndarray:
    X = np.asarray(Xe, dtype=float)
    p = _as_point(material_point, "material_point")
    if X.shape != (8, 3):
        raise ValueError("hex8 coordinates must have shape (8, 3)")
    natural = np.zeros(3, dtype=float)
    for _ in range(16):
        shape = hex8_shape_functions(float(natural[0]), float(natural[1]), float(natural[2]))
        gradients = hex8_natural_gradients(float(natural[0]), float(natural[1]), float(natural[2]))
        residual = shape @ X - p
        if float(np.linalg.norm(residual)) <= 1.0e-12:
            break
        J = X.T @ gradients
        try:
            delta = np.linalg.solve(J, -residual)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(J, -residual, rcond=None)
        natural = np.clip(natural + delta, -1.5, 1.5)
        if float(np.linalg.norm(delta)) <= 1.0e-12:
            break
    return natural


def _as_point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return point
