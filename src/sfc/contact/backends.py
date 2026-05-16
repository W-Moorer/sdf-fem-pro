"""Contact backend adapters.

These adapters separate contact enforcement from analysis time integration.
The current registered-style backend wraps the existing frictionless penalty
normal contact path without changing the gap, Jacobian, force, or stiffness
formulas.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .narrow_phase import ContactConstraint, SurfaceSample, compute_contact_constraints
from .penalty import penalty_contact_response


@dataclass(frozen=True, slots=True)
class ContactBackendResult:
    """Contact constraints plus assembled force and tangent contribution."""

    constraints: tuple[ContactConstraint, ...]
    force: np.ndarray
    stiffness: csr_matrix


@dataclass(frozen=True, slots=True)
class PenaltyContactBackend:
    """Frictionless penalty contact backend."""

    stiffness: float
    name: str = "penalty"

    def __post_init__(self) -> None:
        if float(self.stiffness) <= 0.0:
            raise ValueError("stiffness must be positive")

    def assemble_response(
        self,
        constraints: Sequence[ContactConstraint],
        *,
        n_total_dofs: int,
        slave_dof_offset: int = 0,
        master_dof_offset: int = 0,
    ) -> ContactBackendResult:
        """Assemble penalty force and Gauss-Newton contact stiffness."""

        force, K = penalty_contact_response(
            constraints,
            stiffness=float(self.stiffness),
            n_total_dofs=n_total_dofs,
            slave_dof_offset=slave_dof_offset,
            master_dof_offset=master_dof_offset,
        )
        return ContactBackendResult(constraints=tuple(constraints), force=force, stiffness=K)

    def evaluate(
        self,
        slave_x_current: np.ndarray,
        samples: Iterable[SurfaceSample],
        master_x_current: np.ndarray,
        master_boundary_faces: np.ndarray,
        *,
        n_total_dofs: int,
        slave_dof_offset: int = 0,
        master_dof_offset: int = 0,
    ) -> ContactBackendResult:
        """Compute dynamic-SDF constraints from supplied samples and assemble response."""

        constraints = tuple(
            compute_contact_constraints(
                slave_x_current,
                samples,
                master_x_current,
                master_boundary_faces,
            )
        )
        return self.assemble_response(
            constraints,
            n_total_dofs=n_total_dofs,
            slave_dof_offset=slave_dof_offset,
            master_dof_offset=master_dof_offset,
        )
