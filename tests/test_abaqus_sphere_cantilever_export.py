from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.abaqus_odb_to_vtk import (
    _contact_point_scalar_arrays,
    _node_averaged_tensor_invariants,
    _node_average_cell_symmetric6,
    _select_frame_indices,
    _tensor_from_symmetric6,
    _von_mises_from_symmetric6,
)
from validation.run_abaqus_sphere_cantilever_explicit import (
    DEFAULT_OUT_DIR,
    ModelConfig,
    _abaqus_reported_wallclock_seconds,
    _beam_mesh,
    _sphere_mesh,
    build_input_text,
)


def test_sphere_cantilever_input_requests_explicit_contact_outputs() -> None:
    text = build_input_text(ModelConfig())

    assert "*Dynamic, Explicit, DIRECT USER CONTROL" in text
    assert "1.000000000000e-05, 3.000000000000e+00" in text
    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "5.000000000000e+09" in text
    assert "*Bulk Viscosity\n0., 0." in text
    assert "*Contact Inclusions, ALL EXTERIOR" in text
    assert "U, V" in text
    assert "S, LE" in text
    assert "C3D8R" in text
    assert "C3D4" in text


def test_default_output_directory_is_not_under_results() -> None:
    assert "results" not in DEFAULT_OUT_DIR.parts
    assert DEFAULT_OUT_DIR.parts[-2:] == ("commercial_software_comparison", "abaqus_sphere_cantilever")


def test_generated_meshes_are_nonempty_and_deterministic() -> None:
    cfg = ModelConfig()
    beam_nodes, beam_elements, fixed_nodes = _beam_mesh(cfg)
    sphere_nodes, sphere_elements, sphere_all = _sphere_mesh(cfg)

    assert len(beam_nodes) == (cfg.beam_nx + 1) * (cfg.beam_ny + 1) * (cfg.beam_nz + 1)
    assert len(beam_elements) == cfg.beam_nx * cfg.beam_ny * cfg.beam_nz
    assert len(fixed_nodes) == (cfg.beam_ny + 1) * (cfg.beam_nz + 1)
    assert len(sphere_nodes) == 3 + (cfg.sphere_latitudes - 1) * cfg.sphere_longitudes
    assert len(sphere_elements) == 2 * cfg.sphere_longitudes * (cfg.sphere_latitudes - 1)
    assert sphere_all == list(range(1, len(sphere_nodes) + 1))


def test_default_model_uses_fine_three_second_stiff_beam_case() -> None:
    cfg = ModelConfig()

    assert (cfg.beam_nx, cfg.beam_ny, cfg.beam_nz) == (60, 8, 4)
    assert (cfg.sphere_latitudes, cfg.sphere_longitudes) == (24, 48)
    assert cfg.duration == 3.0
    assert cfg.output_interval == 0.001
    assert cfg.fixed_dt == 1.0e-5
    assert cfg.beam_young == 2.5e9
    assert cfg.sphere_young == 5.0e6
    assert cfg.sphere_young < cfg.beam_young


def test_abaqus_tensor_helpers_use_symmetric_3d_order() -> None:
    tensor = _tensor_from_symmetric6((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    assert tensor == ((1.0, 4.0, 5.0), (4.0, 2.0, 6.0), (5.0, 6.0, 3.0))
    assert _von_mises_from_symmetric6((1.0, 1.0, 1.0, 0.0, 0.0, 0.0)) == 0.0


def test_abaqus_nodeavg_invariants_average_tensor_components_first() -> None:
    cells = [[0, 1, 2, 3], [1, 4, 2, 5]]
    stress = [(10.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 10.0, 0.0, 0.0, 0.0, 0.0)]
    strain = [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)]

    averaged = _node_average_cell_symmetric6(stress, cells, node_count=6)
    nodeavg_vm, _nodeavg_strain, _nodeavg_eq, scalaravg_vm, _scalaravg_strain, _scalaravg_eq = (
        _node_averaged_tensor_invariants(
            stress=stress,
            strain=strain,
            cells=cells,
            node_count=6,
            young=100.0,
            poisson=0.25,
        )
    )

    assert averaged[1] == (5.0, 5.0, 0.0, 0.0, 0.0, 0.0)
    assert nodeavg_vm[1] == pytest.approx(_von_mises_from_symmetric6((5.0, 5.0, 0.0, 0.0, 0.0, 0.0)))
    assert scalaravg_vm[1] == pytest.approx(10.0)
    assert nodeavg_vm[1] < scalaravg_vm[1]


def test_odb_frame_selection_supports_stride_and_time_window() -> None:
    class Frame:
        def __init__(self, time: float) -> None:
            self.frameValue = time

    frames = [Frame(i * 1.0e-5) for i in range(11)]

    assert _select_frame_indices(frames, frame_stride=2, time_end=5.0e-5) == [0, 2, 4]
    assert _select_frame_indices(frames, frame_stride=2, time_start=2.0e-5, time_end=8.0e-5) == [2, 4, 6, 8]


def test_contact_output_aliases_map_to_vtk_point_scalars() -> None:
    class Instance:
        name = "GEAR1-1"

    class Value:
        def __init__(self, node_label: int, data) -> None:  # noqa: ANN001
            self.instance = Instance()
            self.nodeLabel = node_label
            self.data = data

    class Field:
        def __init__(self, values: list[Value]) -> None:
            self.values = values

    class Frame:
        fieldOutputs = {
            "CSTRESS": Field([Value(10, (3.0, 0.0, 0.0)), Value(20, (7.0, 0.0, 0.0))]),
            "CDISP": Field([Value(10, (-0.2, 0.0, 0.0)), Value(20, (-3.4028234663852886e38, 0.0, 0.0))]),
            "CSTATUS": Field([Value(10, 1.0), Value(20, 0.0)]),
        }

    arrays, metrics = _contact_point_scalar_arrays(Frame(), {("GEAR1-1", 10): 0, ("GEAR1-1", 20): 1})

    assert arrays["contact_pressure_nodeavg"] == [3.0, 7.0]
    assert arrays["contact_opening_node"] == [-0.2, 0.0]
    assert arrays["contact_penetration_nodeavg"] == [0.2, 0.0]
    assert arrays["contact_status_node"] == [1.0, 0.0]
    assert arrays["contact_active_node"] == [1.0, 0.0]
    assert metrics["contact_output_available"] == 1
    assert metrics["contact_pressure_source"] == "CSTRESS"
    assert metrics["contact_opening_source"] == "CDISP"
    assert metrics["active_contact_node_count"] == 1
    assert metrics["max_contact_penetration_nodeavg"] == 0.2


def test_parse_abaqus_reported_wallclock_seconds(tmp_path: Path) -> None:
    sta = tmp_path / "job.sta"
    sta.write_text(
        "\n".join(
            [
                "  EXPLICIT EXECUTABLE TIME SUMMARY",
                "       USER TIME (SEC)      =   779.70",
                "       SYSTEM TIME (SEC)    =   16.700",
                "       WALLCLOCK TIME (SEC) =         1239",
            ]
        ),
        encoding="ascii",
    )

    assert _abaqus_reported_wallclock_seconds(sta) == 1239.0
