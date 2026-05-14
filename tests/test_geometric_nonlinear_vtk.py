from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_geometric_nonlinear_vtk import (
    _internal_response,
    _internal_response_with_tangent,
    _structured_block_mesh,
    _tet_reference_data,
    run_simulation,
)


def test_total_lagrangian_undeformed_state_has_zero_stress() -> None:
    X, elements = _structured_block_mesh(1, size=(0.8, 0.8, 0.8), bottom_z=0.05)
    volumes, grads = _tet_reference_data(X, elements)

    force, strain, stress, vm, energy = _internal_response(X, elements, volumes, grads, X.copy(), E=1000.0, nu=0.3)

    assert np.linalg.norm(force) == pytest.approx(0.0, abs=1.0e-10)
    assert np.linalg.norm(strain) == pytest.approx(0.0, abs=1.0e-12)
    assert np.linalg.norm(stress) == pytest.approx(0.0, abs=1.0e-10)
    assert np.max(vm) == pytest.approx(0.0, abs=1.0e-10)
    assert energy == pytest.approx(0.0, abs=1.0e-12)


def test_stvk_internal_tangent_matches_finite_difference() -> None:
    X, elements = _structured_block_mesh(1, size=(0.8, 0.8, 0.8), bottom_z=0.2)
    volumes, grads = _tet_reference_data(X, elements)
    x = X.copy()
    x[:, 0] += 0.04 * X[:, 2]
    x[:, 2] += 0.02 * X[:, 0]
    direction = np.linspace(-0.3, 0.4, x.size).reshape(x.shape)
    direction /= np.linalg.norm(direction)

    force, _, _, _, _, material, geometric, tangent = _internal_response_with_tangent(
        X,
        elements,
        volumes,
        grads,
        x,
        E=1000.0,
        nu=0.3,
        assemble_tangent=True,
    )
    h = 1.0e-6
    force_plus, *_ = _internal_response(X, elements, volumes, grads, x + h * direction, E=1000.0, nu=0.3)
    force_minus, *_ = _internal_response(X, elements, volumes, grads, x - h * direction, E=1000.0, nu=0.3)
    finite_difference = ((force_plus - force_minus) / (2.0 * h)).reshape(-1)

    assert np.linalg.norm(material.data) > 0.0
    assert np.linalg.norm(geometric.data) > 0.0
    assert np.linalg.norm(force) > 0.0
    assert tangent @ direction.reshape(-1) == pytest.approx(finite_difference, rel=1.0e-5, abs=1.0e-6)


def test_geometric_nonlinear_vtk_frames_are_consecutive(tmp_path: Path) -> None:
    outputs = run_simulation(tmp_path, resolution=1, duration=0.02, dt=0.001, frame_stride=5)
    vtk_dir = outputs["vtk_dir"]

    frame_names = sorted(path.name for path in vtk_dir.glob("frame_*.vtk"))
    assert frame_names == ["frame_0000.vtk", "frame_0001.vtk", "frame_0002.vtk", "frame_0003.vtk", "frame_0004.vtk"]

    text = (vtk_dir / "frame_0001.vtk").read_text(encoding="ascii")
    assert "DATASET UNSTRUCTURED_GRID" in text
    assert "SCALARS object_id int 1" in text
    assert "VECTORS displacement float" in text
    assert "SCALARS von_mises float 1" in text
    assert "TENSORS green_lagrange_strain float" in text
    assert "TENSORS cauchy_stress float" in text

    with outputs["frames"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["frame"]) for row in rows] == list(range(len(rows)))


def test_geometric_nonlinear_history_contains_energy_and_contact_fields(tmp_path: Path) -> None:
    outputs = run_simulation(tmp_path, resolution=1, duration=0.04, dt=0.001, frame_stride=10)

    with outputs["history"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = {
        "z_cm",
        "min_gap",
        "max_penetration",
        "active_contact_count",
        "kinetic_energy",
        "strain_energy",
        "gravitational_energy",
        "contact_energy",
        "total_energy",
        "max_von_mises",
        "material_tangent_norm",
        "geometric_tangent_norm",
        "contact_tangent_norm",
        "newton_iterations",
        "newton_residual_norm",
    }
    assert required <= set(rows[0])
    assert all(np.isfinite(float(row["total_energy"])) for row in rows)
    assert max(float(row["max_von_mises"]) for row in rows) >= 0.0
    assert max(int(row["newton_iterations"]) for row in rows) > 0


def test_sphere_drop_initial_frame_has_visible_plane_separation(tmp_path: Path) -> None:
    outputs = run_simulation(tmp_path, case="sphere_drop", resolution=1, duration=0.01, dt=0.001, frame_stride=5, initial_gap=0.08)

    with outputs["history"].open(newline="", encoding="utf-8") as f:
        first_row = next(csv.DictReader(f))
    assert float(first_row["min_gap"]) > 0.0

    text = (outputs["vtk_dir"] / "frame_0000.vtk").read_text(encoding="ascii")
    assert "CELL_TYPES" in text
    assert "\n9\n" in text
