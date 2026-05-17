from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_fuzzycontact_derived_sfc_simulation import run_validation


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
        raise ValueError("fixture supports only vector/tensor fields")
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


def test_fuzzycontact_derived_sfc_simulation_runs_from_vtu_fixture(tmp_path: Path) -> None:
    source = tmp_path / "source" / "problem_1"
    source.mkdir(parents=True)
    for body, z0 in (("bottom", 0.0), ("top", 0.35)):
        _write_field_tetra_vtu(source / f"displacement_{body}.vtu", z0=z0, field_name="displacement (m)", components=3)
        _write_field_tetra_vtu(source / f"strain_{body}.vtu", z0=z0, field_name="strain", components=9)
        _write_field_tetra_vtu(source / f"stress_{body}.vtu", z0=z0, field_name="stress (Pa)", components=9)

    outputs = run_validation(
        tmp_path / "out",
        source_dir=source.parent,
        quick=True,
        max_cases=1,
        max_contact_samples=2,
    )

    for key in ("cases", "history", "summary"):
        assert outputs[key].exists(), key
        assert outputs[key].stat().st_size > 0, key

    rows = _rows(outputs["cases"])
    assert rows[0]["status"] == "completed"
    assert rows[0]["claim_scope"].startswith("fuzzycontact-derived SFC simulation")
    assert float(rows[0]["master_E_fit"]) > 0.0
    assert float(rows[0]["field_query_build_seconds_total"]) >= 0.0


def test_fuzzycontact_derived_sfc_simulation_uses_true_field_path() -> None:
    source = (ROOT / "validation" / "run_fuzzycontact_derived_sfc_simulation.py").read_text(encoding="utf-8")

    assert "DynamicNarrowBandSDF" in source
    assert "field_contact" in source
    assert "dynamic_surface_sdf" not in source
