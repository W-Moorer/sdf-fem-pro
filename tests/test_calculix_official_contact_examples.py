from __future__ import annotations

import csv
import gzip
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_calculix_official_contact_examples.py"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _minimal_case(*, element_type: str = "C3D8", contact_type: str = "SURFACE TO SURFACE", dynamic: bool = False, law: str = "LINEAR") -> str:
    step = "*DYNAMIC\n0.001, 0.001\n" if dynamic else "*STATIC\n1.,1.\n"
    return f"""*NODE, NSET=Nall
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT, TYPE={element_type}, ELSET=Eall
1,1,2,3,4,5,6,7,8
*MATERIAL, NAME=EL
*ELASTIC
1.0,0.3
*SURFACE, NAME=Smast
Eall,S1
*SURFACE, NAME=Sslav
Eall,S2
*CONTACT PAIR, INTERACTION=SI1, TYPE={contact_type}
Sslav,Smast
*SURFACE INTERACTION, NAME=SI1
*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE={law}
1.E7
*STEP, NLGEOM
{step}*END STEP
"""


def test_official_calculix_example_catalog_claim_gates(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    files = {
        "contactenergy.inp": _minimal_case(),
        "contact1.inp": _minimal_case(contact_type="NODE TO SURFACE", law="EXPONENTIAL"),
        "contact3.inp": _minimal_case(contact_type="NODE TO SURFACE"),
        "contact6.inp": _minimal_case(contact_type="NODE TO SURFACE"),
    }
    for name in ["scheibe2f2f.inp.gz", "ball.inp.gz"]:
        content = _minimal_case(dynamic=name.startswith("ball"))
        if name.startswith("ball"):
            content = content.replace("*ELEMENT, TYPE=C3D8", "*ELEMENT, TYPE=C3D8") + "*ELEMENT, TYPE=S8, ELSET=Eflo\n2,1,2,3,4,5,6,7,8\n"
        with gzip.open(examples / name, "wt", encoding="utf-8") as f:
            f.write(content)
    for name, text in files.items():
        (examples / name).write_text(text, encoding="utf-8")

    out_dir = tmp_path / "official"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--quick",
            "--skip-run",
            "--examples-dir",
            str(examples),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    expected = [
        "official_calculix_example_catalog.csv",
        "official_calculix_example_runs.csv",
        "official_calculix_example_claim_gates.csv",
        "official_calculix_example_commands.csv",
        "official_calculix_example_plots.csv",
        "official_calculix_contact_examples_summary.md",
    ]
    for name in expected:
        path = out_dir / name
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    catalog = {row["case_id"]: row for row in _rows(out_dir / "official_calculix_example_catalog.csv")}
    assert set(catalog) == {
        "contactenergy_c3d8_static_energy",
        "scheibe2f2f_c3d8_nlgeom_static",
        "ball_c3d8_dynamic_drop",
        "contact1_c3d8_exponential_law",
        "contact3_c3d8_linear_law",
        "contact6_c3d8_stiff_linear_law",
    }
    assert catalog["contactenergy_c3d8_static_energy"]["adaptation_level"] == "direct"
    assert catalog["ball_c3d8_dynamic_drop"]["has_dynamic"] == "true"

    gates = _rows(out_dir / "official_calculix_example_claim_gates.csv")
    by_case_claim = {(row["case_id"], row["claim"]): row for row in gates}
    assert by_case_claim[("contactenergy_c3d8_static_energy", "paper_external_reference_direct")]["allowed"] == "true"
    assert by_case_claim[("contact1_c3d8_exponential_law", "small_law_alignment_reference")]["allowed"] == "true"
    assert by_case_claim[("scheibe2f2f_c3d8_nlgeom_static", "recommended_next_adaptation")]["allowed"] == "true"
    assert by_case_claim[("ball_c3d8_dynamic_drop", "recommended_next_adaptation")]["allowed"] == "true"
    assert all(
        row["allowed"] == "false"
        for row in gates
        if row["claim"] == "full_trajectory_or_full_field_equivalence"
    )

    for figure in [
        out_dir / "figures" / "official_calculix_example_classification.png",
        out_dir / "figures" / "official_calculix_example_runtime.png",
    ]:
        assert figure.exists(), figure
        assert figure.stat().st_size > 0, figure
