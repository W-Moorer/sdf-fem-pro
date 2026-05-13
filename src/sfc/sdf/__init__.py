"""Signed-distance and local projection utilities."""

from .dynamic_surface_sdf import (
    SurfaceSDFResult,
    dynamic_surface_sdf,
    query_dynamic_surface_sdf,
)
from .local_projection import closest_point_on_triangle, signed_point_triangle_gap

__all__ = [
    "SurfaceSDFResult",
    "closest_point_on_triangle",
    "dynamic_surface_sdf",
    "query_dynamic_surface_sdf",
    "signed_point_triangle_gap",
]
