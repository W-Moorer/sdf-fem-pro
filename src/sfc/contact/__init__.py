"""Contact detection and response utilities."""

from .broad_phase import UniformTriangleAABBHash, triangle_aabbs
from .backends import ContactBackendResult, PenaltyContactBackend
from .field_contact import (
    FieldContactConstraint,
    assemble_field_contact_jacobian,
    compute_field_contact_constraints,
    field_contact_constraint_from_sample,
    field_contact_jacobian_entries,
    field_contact_jacobian_row,
    field_penalty_contact_response,
)
from .jacobian import (
    assemble_contact_jacobian,
    contact_jacobian_entries,
    contact_jacobian_row,
)
from .narrow_phase import (
    ContactConstraint,
    SurfaceSample,
    compute_contact_constraints,
    contact_constraint_from_sample,
)
from .penalty import penalty_contact_response
from .sdf_geometry import DynamicSurfaceSDFContactGeometry

__all__ = [
    "ContactConstraint",
    "ContactBackendResult",
    "DynamicSurfaceSDFContactGeometry",
    "FieldContactConstraint",
    "PenaltyContactBackend",
    "SurfaceSample",
    "UniformTriangleAABBHash",
    "assemble_contact_jacobian",
    "assemble_field_contact_jacobian",
    "compute_contact_constraints",
    "compute_field_contact_constraints",
    "contact_constraint_from_sample",
    "contact_jacobian_entries",
    "contact_jacobian_row",
    "field_contact_constraint_from_sample",
    "field_contact_jacobian_entries",
    "field_contact_jacobian_row",
    "field_penalty_contact_response",
    "penalty_contact_response",
    "triangle_aabbs",
]
