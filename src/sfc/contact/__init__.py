"""Contact detection and response utilities."""

from .broad_phase import UniformTriangleAABBHash, triangle_aabbs
from .backends import ContactBackendResult, PenaltyContactBackend
from .field_contact import (
    FieldContactConstraint,
    FieldContactMatrixFreeStiffness,
    FieldSurfaceContactResponse,
    SurfaceQuadratureCache,
    assemble_field_contact_jacobian,
    compute_field_contact_constraints,
    field_contact_constraint_from_sample,
    field_contact_jacobian_entries,
    field_contact_jacobian_row,
    field_penalty_contact_response,
    node_to_surface_field_penalty_response,
    surface_to_surface_field_penalty_response,
    surface_to_surface_field_penalty_response_vectorized,
    triangle_surface_quadrature_cache,
    triangle_surface_quadrature_samples,
)
from .jacobian import (
    assemble_contact_jacobian,
    contact_jacobian_entries,
    contact_jacobian_row,
)
from .lagrangian_sdf_oracle import (
    LagrangianOracleConstraint,
    LagrangianOracleContactResponse,
    LagrangianPatchPairQueryResult,
    LagrangianPullbackQueryResult,
    LagrangianSDFContactOracle,
    LagrangianSDFQueryResult,
    lagrangian_oracle_constraint_from_sample,
    lagrangian_oracle_jacobian_row,
    lagrangian_oracle_penalty_response,
    lagrangian_oracle_penalty_response_batch,
    lagrangian_patch_pair_query,
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
    "FieldContactMatrixFreeStiffness",
    "FieldSurfaceContactResponse",
    "LagrangianOracleConstraint",
    "LagrangianOracleContactResponse",
    "LagrangianPatchPairQueryResult",
    "LagrangianPullbackQueryResult",
    "LagrangianSDFContactOracle",
    "LagrangianSDFQueryResult",
    "SurfaceQuadratureCache",
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
    "lagrangian_oracle_constraint_from_sample",
    "lagrangian_oracle_jacobian_row",
    "lagrangian_oracle_penalty_response",
    "lagrangian_oracle_penalty_response_batch",
    "lagrangian_patch_pair_query",
    "node_to_surface_field_penalty_response",
    "penalty_contact_response",
    "surface_to_surface_field_penalty_response",
    "surface_to_surface_field_penalty_response_vectorized",
    "triangle_aabbs",
    "triangle_surface_quadrature_cache",
    "triangle_surface_quadrature_samples",
]
