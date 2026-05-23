from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_gear_explicit_sdf_comparison import (
    DEFAULT_SOURCE,
    GearMesh,
    build_explicit_input_text,
    parse_gear_input,
    _faces_from_surface_entries,
)


pytestmark = pytest.mark.skipif(not DEFAULT_SOURCE.exists(), reason="commercial gear input is not present")


def test_parse_flexible_gear_input_extracts_contact_and_hub_data() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)

    assert model.gear1.nodes.shape == (19322, 3)
    assert model.gear2.nodes.shape == (19562, 3)
    assert model.gear1_contact_faces.shape[0] == 11560
    assert model.gear2_contact_faces.shape[0] == 11560
    assert len(model.gear1_hub_labels) == 204
    assert len(model.gear2_hub_labels) == 204
    assert model.density == pytest.approx(7850.0)
    assert model.young == pytest.approx(2.05e11)
    assert model.poisson == pytest.approx(0.28)
    assert model.gear1_angular_velocity_z == pytest.approx(52.36)
    assert model.gear2_torque_z == pytest.approx(50.0)
    assert model.dynamic_initial_dt == pytest.approx(1.0e-5)
    assert model.dynamic_duration == pytest.approx(5.0e-2)
    assert model.dynamic_min_dt == pytest.approx(1.0e-10)
    assert model.dynamic_max_dt == pytest.approx(5.0e-5)
    assert model.contact_pressure_overclosure == "HARD"


def test_c3d4_surface_entries_use_abaqus_face_node_order() -> None:
    mesh = GearMesh(
        nodes=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        node_labels=np.asarray([1, 2, 3, 4], dtype=np.int64),
        label_to_index={1: 0, 2: 1, 3: 2, 4: 3},
        elements=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        element_labels=np.asarray([10], dtype=np.int64),
        element_label_to_index={10: 0},
    )

    faces = _faces_from_surface_entries(mesh, ((10, "S1"), (10, "S2"), (10, "S3"), (10, "S4")))

    np.testing.assert_array_equal(
        faces,
        np.asarray(
            [
                [0, 1, 2],
                [0, 3, 1],
                [1, 3, 2],
                [2, 3, 0],
            ],
            dtype=np.int64,
        ),
    )


def test_compact_explicit_deck_uses_penalty_linear_contact() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    text = build_explicit_input_text(
        model,
        duration=2.0e-5,
        fixed_dt=5.0e-9,
        output_interval=1.0e-5,
        contact_stiffness=5.0e9,
        max_abaqus_contact_faces=16,
    )

    assert "*Dynamic, Explicit, DIRECT USER CONTROL" in text
    assert "5.000000000000e-09, 2.000000000000e-05" in text
    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "5.000000000000e+09" in text
    assert "*Contact Pair, interaction=LINEAR_FRICTIONLESS, mechanical constraint=PENALTY" in text
    assert "type=SURFACE TO SURFACE" not in text
    assert "*Bulk Viscosity\n0., 0." in text
    assert "GEAR1_RP, 6, 6, 5.236000000000e+01" in text
    assert "GEAR2_LOAD_RP, 6, 5.000000000000e+01" in text
