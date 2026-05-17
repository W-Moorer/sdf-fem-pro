from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_fuzzycontact_first_frame_sfc_models import run_validation


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_field_tetra_vtu(path: Path, *, z0: float, field_name: str, components: int) -> None:
    if components == 3:
        values = "0 0 0  0 0 -0.01  0 0 -0.01  0 0 -0.02"
    elif components == 9:
        values = (
            "0.001 0 0 0 0.001 0 0 0 0.001 "
            "0.001 0 0 0 0.001 0 0 0 0.001 "
            "0.001 0 0 0 0.001 0 0 0 0.001 "
            "0.001 0 0 0 0.001 0 0 0 0.001"
        )
    else:
        raise ValueError("fixture supports vector or tensor fields")
    path.write_text(
        f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 {z0}
          1 0 {z0}
          0 1 {z0}
          0 0 {z0 + 0.25}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
      </Cells>
      <PointData>
        <DataArray type="Float64" Name="{field_name}" NumberOfComponents="{components}" format="ascii">
          {values}
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
""",
        encoding="utf-8",
    )


def _write_problem_pair(source: Path, problem: str, bodies: tuple[str, str]) -> None:
    folder = source / problem
    folder.mkdir(parents=True)
    for body, z0 in zip(bodies, (0.0, 0.42), strict=True):
        if problem == "problem_3":
            stem = f"{body}_"
            _write_field_tetra_vtu(folder / f"{stem}displacement.vtu", z0=z0, field_name="displacement (mm)", components=3)
            _write_field_tetra_vtu(folder / f"{stem}strain.vtu", z0=z0, field_name="strain", components=9)
            _write_field_tetra_vtu(folder / f"{stem}stress.vtu", z0=z0, field_name="stress (MPa)", components=9)
        else:
            _write_field_tetra_vtu(folder / f"displacement_{body}.vtu", z0=z0, field_name="displacement (mm)", components=3)
            _write_field_tetra_vtu(folder / f"strain_{body}.vtu", z0=z0, field_name="strain", components=9)
            _write_field_tetra_vtu(folder / f"stress_{body}.vtu", z0=z0, field_name="stress (MPa)", components=9)


def test_fuzzycontact_first_frame_sfc_models_runs_ordered_fixture(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_problem_pair(source, "problem_1", ("bottom", "top"))
    _write_problem_pair(source, "problem_3", ("base", "indenter"))

    outputs = run_validation(
        tmp_path / "out",
        source_dir=source,
        quick=True,
        problems=("problem_1", "problem_3"),
        max_contact_samples=2,
        max_master_faces=4,
    )

    for key in ("models", "contact", "samples", "plots", "summary"):
        assert outputs[key].exists(), key
        assert outputs[key].stat().st_size > 0, key

    model_rows = _rows(outputs["models"])
    contact_rows = _rows(outputs["contact"])

    assert [row["problem"] for row in model_rows] == ["problem_1", "problem_3"]
    assert model_rows[0]["master_reference_rule"] == "points_minus_displacement"
    assert model_rows[1]["master_reference_rule"] == "points"
    assert all(row["query_path"].startswith("DynamicNarrowBandSDF") for row in contact_rows)


def test_fuzzycontact_first_frame_sfc_models_uses_true_field_contact_path() -> None:
    source = (ROOT / "validation" / "run_fuzzycontact_first_frame_sfc_models.py").read_text(encoding="utf-8")

    assert "DynamicNarrowBandSDF" in source
    assert "field_contact" in source
    assert "dynamic_surface_sdf" not in source
    assert "points_minus_displacement" in source
