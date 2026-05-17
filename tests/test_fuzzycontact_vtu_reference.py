from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation import run_fuzzycontact_vtu_reference as fuzzy


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_tetra_vtu(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0
          1 0 0
          0 1 0
          0 0 1
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
      </Cells>
      <PointData>
        <DataArray type="Float64" Name="displacement" NumberOfComponents="3" format="ascii">
          0 0 0
          0.1 0 0
          0 0.2 0
          0 0 0.3
        </DataArray>
        <DataArray type="Float64" Name="von_mises" format="ascii">
          1 2 3 4
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
""",
        encoding="utf-8",
    )


def _write_field_tetra_vtu(path: Path, *, z0: float, field_name: str, components: int) -> None:
    if components == 1:
        values = "1 2 3 4"
    elif components == 3:
        values = "0 0 0  0.1 0 0  0 0.2 0  0 0 0.3"
    else:
        values = "1 0 0 0 1 0 0 0 1  2 0 0 0 2 0 0 0 2  3 0 0 0 3 0 0 0 3  4 0 0 0 4 0 0 0 4"
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


def test_fuzzycontact_reference_parses_local_vtu_and_writes_3d_preview(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_tetra_vtu(source_dir / "solution.vtu")

    monkeypatch.setattr(
        fuzzy,
        "_fetch_metadata",
        lambda: {
            "id": "tdgkkjyr2f",
            "name": "Dataset of FuzzyContact",
            "doi": {"id": "10.17632/tdgkkjyr2f.3"},
            "version": "3",
            "publish_date": "2024-01-01",
            "data_licence": {"short_name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
            "files": [
                {
                    "filename": "Dataset_for_Solutions.zip",
                    "size": 257263414,
                    "content_details": {
                        "sha256_hash": "fixture",
                        "download_url": "https://example.invalid/Dataset_for_Solutions.zip",
                    },
                }
            ],
        },
    )

    out_dir = tmp_path / "out"
    outputs = fuzzy.run_reference(out_dir, source_dir=source_dir, max_files=1)

    assert set(outputs) == {
        "metadata",
        "inventory",
        "fields",
        "sfc_replay_samples",
        "sfc_replay_summary",
        "plots",
        "summary",
    }
    for key in ("metadata", "inventory", "fields", "plots", "summary"):
        assert outputs[key].is_file() and outputs[key].stat().st_size > 0
    assert (out_dir / "figures" / "fuzzycontact_vtu_preview.png").is_file()
    assert (out_dir / "figures" / "fuzzycontact_vtu_preview.pdf").is_file()

    inventory = _rows(outputs["inventory"])
    fields = _rows(outputs["fields"])
    plots = _rows(outputs["plots"])
    summary = outputs["summary"].read_text(encoding="utf-8")

    assert inventory[0]["status"] == "parsed"
    assert inventory[0]["points"] == "4"
    assert inventory[0]["cells"] == "1"
    assert {"displacement", "von_mises"} <= {row["field"] for row in fields}
    assert plots[0]["plot"] == "fuzzycontact_vtu_preview"
    assert "external VTU field reference" in summary


def test_fuzzycontact_reference_maps_vtu_fields_to_sfc_replay_samples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "source" / "problem_1"
    source_dir.mkdir(parents=True)
    for body, z0 in (("bottom", 0.0), ("top", 0.42)):
        _write_field_tetra_vtu(source_dir / f"displacement_{body}.vtu", z0=z0, field_name="displacement (m)", components=3)
        _write_field_tetra_vtu(source_dir / f"strain_{body}.vtu", z0=z0, field_name="strain", components=9)
        _write_field_tetra_vtu(source_dir / f"stress_{body}.vtu", z0=z0, field_name="stress (Pa)", components=9)

    monkeypatch.setattr(
        fuzzy,
        "_fetch_metadata",
        lambda: {
            "id": "tdgkkjyr2f",
            "name": "Dataset of FuzzyContact",
            "doi": {"id": "10.17632/tdgkkjyr2f.3"},
            "data_licence": {"short_name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
            "files": [{"filename": "Dataset_for_Solutions.zip", "size": 1, "content_details": {}}],
        },
    )

    out_dir = tmp_path / "out"
    outputs = fuzzy.run_reference(out_dir, source_dir=source_dir.parent, max_files=6, max_replay_groups=1, max_replay_samples=4)

    sample_rows = _rows(outputs["sfc_replay_samples"])
    summary_rows = _rows(outputs["sfc_replay_summary"])

    assert sample_rows
    assert summary_rows[0]["status"] == "mapped"
    assert sample_rows[0]["group_id"] == "problem_1:default:bottom->top"
    assert float(sample_rows[0]["slave_displacement_magnitude"]) >= 0.0
    assert float(sample_rows[0]["master_stress_norm"]) >= 0.0
    assert (out_dir / "figures" / "fuzzycontact_sfc_replay_samples.png").is_file()


def test_fuzzycontact_reference_metadata_only_does_not_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fuzzy,
        "_fetch_metadata",
        lambda: {
            "id": "tdgkkjyr2f",
            "name": "Dataset of FuzzyContact",
            "doi": {"id": "10.17632/tdgkkjyr2f.3"},
            "data_licence": {"short_name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
            "files": [{"filename": "Dataset_for_Solutions.zip", "size": 257263414, "content_details": {}}],
        },
    )

    out_dir = tmp_path / "out"
    outputs = fuzzy.run_reference(out_dir)

    inventory = _rows(outputs["inventory"])
    assert inventory[0]["status"] == "metadata_only_pending_vtu_source"
    assert not (out_dir / "downloads").exists()
