from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_fuzzycontact_calculix_native_contact_trajectory import (  # noqa: E402
    _parse_element_table,
    run_validation,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_native_contact_skip_outputs_independent_inputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path / "native", quick=True, skip_calculix=True)

    for key in ("comparison", "sfc_history", "commands", "timing", "summary"):
        assert outputs[key].exists(), key
        assert outputs[key].stat().st_size > 0, key

    rows = _rows(outputs["comparison"])
    assert [row["problem"] for row in rows] == [
        "problem_1",
        "problem_1",
        "problem_3",
        "problem_3",
        "problem_4",
        "problem_4",
    ]
    assert {row["status"] for row in rows} == {"not_run"}
    assert any(int(row["sfc_active_samples"]) > 0 for row in rows)
    assert {row["sfc_contact_backend"] for row in rows} == {"surface_to_surface_vectorized"}
    assert {row["sfc_quadrature_order"] for row in rows} == {"7"}
    assert all(int(row["sfc_quadrature_points"]) >= int(row["sfc_active_samples"]) for row in rows)
    assert any(int(row["sfc_quadrature_points"]) > int(row["sfc_active_samples"]) for row in rows)

    commands = _rows(outputs["commands"])
    assert len(commands) == 3
    for row in commands:
        inp = Path(row["input_file"])
        assert inp.exists()
        text = inp.read_text(encoding="utf-8")
        assert "*CONTACT PAIR" in text
        assert "*SURFACE INTERACTION" in text
        assert "*CONTACT PRINT" in text
        assert "*CLOAD" not in text

    timing = _rows(outputs["timing"])
    assert [row["problem"] for row in timing] == ["problem_1", "problem_3", "problem_4"]
    assert all(float(row["sfc_total_wall_seconds"]) > 0.0 for row in timing)
    assert {row["sfc_contact_backend"] for row in timing} == {"surface_to_surface_vectorized"}


def test_native_contact_is_not_equivalent_load_or_replay() -> None:
    source = (ROOT / "validation" / "run_fuzzycontact_calculix_native_contact_trajectory.py").read_text(
        encoding="utf-8"
    )

    assert "DynamicNarrowBandSDF" in source
    assert "node_to_surface_field_penalty_response" in source
    assert "surface_to_surface_field_penalty_response" in source
    assert "surface_to_surface_field_penalty_response_vectorized" in source
    assert "triangle_surface_quadrature_samples" in source
    assert "*CONTACT PAIR" in source
    assert "*CLOAD" not in source
    assert "run_fuzzycontact_vtu_reference" not in source
    assert "legacy_abaqus_prototype" not in source
    assert ".odb" not in source


def test_parse_element_table_stops_before_next_calculix_table(tmp_path: Path) -> None:
    dat = tmp_path / "native.dat"
    dat.write_text(
        "\n".join(
            [
                " stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EMASTER and time  0.1000000E+01",
                "",
                "         1   1  1.000000E+00  2.000000E+00  3.000000E+00  4.000000E+00  5.000000E+00  6.000000E+00",
                "",
                " strains (elem, integ.pnt.,exx,eyy,ezz,exy,exz,eyz) for set EMASTER and time  0.1000000E+01",
                "",
                "         1   1  7.000000E+00  8.000000E+00  9.000000E+00  1.000000E+01  1.100000E+01  1.200000E+01",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stress = _parse_element_table(dat, "stresses", 1)[1.0]
    strain = _parse_element_table(dat, "strains", 1)[1.0]

    np.testing.assert_allclose(stress[0], [1.0, 2.0, 3.0, 4.0, 6.0, 5.0])
    np.testing.assert_allclose(strain[0], [7.0, 8.0, 9.0, 20.0, 24.0, 22.0])
