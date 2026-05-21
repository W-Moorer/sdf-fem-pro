from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_rp_mpc_global_assembly_alignment import build_case, run_sfc_global_assembly


def test_rp_mpc_global_assembly_reduces_full_mass_to_rp_dofs(tmp_path) -> None:
    case = build_case()

    path, runtime = run_sfc_global_assembly(case, tmp_path / "sfc.csv")

    rows = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == int(round(case.duration / case.dt)) + 2
    assert int(runtime["full_dofs"]) == 12
    assert int(runtime["reduced_dofs"]) == 6
    assert float(runtime["reduced_mass_rz_rz"]) > 0.0
    assert float(runtime["angular_acceleration_z"]) > 0.0
