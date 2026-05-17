"""Signed-distance field utilities."""

from .dynamic_surface_sdf import (
    SurfaceSDFResult,
    dynamic_surface_sdf,
    query_dynamic_surface_sdf,
    surface_projection_distance_kernel,
)
from .dynamic_narrow_band_sdf import (
    DynamicNarrowBandSDF,
    FieldQueryPayload,
    SDFUpdateStats,
)
from .local_projection import closest_point_on_triangle, signed_point_triangle_gap
from .narrow_band_grid import (
    GridInterpolation,
    GridPayloadInterpolation,
    NarrowBandGrid,
    compute_finite_difference_gradient,
)

__all__ = [
    "DynamicNarrowBandSDF",
    "FieldQueryPayload",
    "GridInterpolation",
    "GridPayloadInterpolation",
    "NarrowBandGrid",
    "SDFUpdateStats",
    "SurfaceSDFResult",
    "closest_point_on_triangle",
    "compute_finite_difference_gradient",
    "dynamic_surface_sdf",
    "query_dynamic_surface_sdf",
    "signed_point_triangle_gap",
    "surface_projection_distance_kernel",
]
