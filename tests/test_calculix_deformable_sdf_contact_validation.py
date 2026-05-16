from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_calculix_deformable_sdf_contact_validation import (  # noqa: E402
    build_two_block_model,
    run_validation,
    write_calculix_input,
)
from validation.run_calculix_drop_impact_comparison import (  # noqa: E402
    _parse_calculix_dat_displacements,
    calculix_available,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_calculix_displacement_parser_ignores_contact_print_rows(tmp_path: Path) -> None:
    dat = tmp_path / "mock.dat"
    dat.write_text(
        """
 displacements (vx,vy,vz) for set NALL and time  0.1000000E+00

         1  1.000000E-01  2.000000E-01  3.000000E-01
         2  4.000000E-01  5.000000E-01  6.000000E-01

 relative contact displacement (slave element+face,normal,tang1,tang2) for all contact elements and time 0.1000000E+00

         1          1 -9.900000E+02  0.000000E+00  0.000000E+00
         2          1 -8.800000E+02  0.000000E+00  0.000000E+00

 displacements (vx,vy,vz) for set NALL and time  0.2000000E+00

         1  7.000000E-01  8.000000E-01  9.000000E-01
         2  1.000000E+00  1.100000E+00  1.200000E+00
""",
        encoding="utf-8",
    )

    blocks = _parse_calculix_dat_displacements(dat, np.asarray([1, 2], dtype=np.int64))

    assert blocks[0.1][0, 0] == pytest.approx(0.1)
    assert blocks[0.1][1, 0] == pytest.approx(0.4)
    assert np.max(np.abs(blocks[0.1])) < 1.0
    assert blocks[0.2][1, 2] == pytest.approx(1.2)


def test_two_block_calculix_input_uses_lateral_stabilization(tmp_path: Path) -> None:
    model = build_two_block_model(resolution=1, approach=0.06)
    inp = tmp_path / "two_block.inp"
    write_calculix_input(model, inp)

    text = inp.read_text(encoding="utf-8")

    assert "Nall, 1, 2, 0.0" in text
    assert "NLOWERBOTTOM, 3, 3, 0.0" in text
    assert "*CONTACT PAIR, INTERACTION=CONTACT, TYPE=SURFACE TO SURFACE" in text
    assert "Lateral DOFs are fixed for this scoped normal-contact replay." in text


@pytest.mark.skipif(not calculix_available(), reason="CalculiX/ccx is unavailable through WSL")
def test_calculix_deformable_sdf_quick_validation_supports_gap_gate(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path, quick=True, resolution=1, timeout=180)

    for path in outputs.values():
        assert path.exists()

    comparison = _rows(outputs["comparison"])
    claims = {row["claim_id"]: row for row in _rows(outputs["claims"])}

    assert all(row["calculix_completed"] == "true" for row in comparison)
    assert all(row["contact_sign_agreement"] == "true" for row in comparison)
    assert claims["sfc_dynamic_sdf_replay_matches_calculix_contact_sign"]["claim_status"] == "supported"
    assert claims["sfc_dynamic_sdf_replay_gap_scale_reported"]["claim_status"] == "supported"
    gap_diffs = [float(row["gap_min_abs_difference"]) for row in comparison if row["gap_min_abs_difference"]]
    assert gap_diffs
    assert max(gap_diffs) < 1.0e-6
