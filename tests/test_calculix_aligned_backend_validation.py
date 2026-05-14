from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_calculix_aligned_backend_validation import run_validation  # noqa: E402


def test_calculix_aligned_backend_validation_quick_outputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path, quick=True, run_calculix=False)

    for path in outputs.values():
        assert path.exists()

    with outputs["claims"].open(newline="", encoding="utf-8") as f:
        claims = {row["claim"]: row for row in csv.DictReader(f)}

    assert claims["consistent_mass_backend_check"]["supported"] == "true"
    assert claims["stvk_affine_patch_backend_check"]["supported"] == "true"
    assert claims["stvk_tangent_finite_difference_backend_check"]["supported"] == "true"
    assert claims["hht_newmark_freefall_backend_check"]["supported"] == "true"
    assert claims["no_contact_newton_cantilever_backend_check"]["supported"] == "true"

    with outputs["freefall"].open(newline="", encoding="utf-8") as f:
        freefall_rows = list(csv.DictReader(f))
    assert freefall_rows[0]["status"] == "passed"

    with outputs["cantilever"].open(newline="", encoding="utf-8") as f:
        cantilever_rows = list(csv.DictReader(f))
    assert cantilever_rows[0]["sfc_status"] == "passed"
