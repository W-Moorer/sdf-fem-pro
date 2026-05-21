from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_gear_explicit_sdf_comparison import DEFAULT_SOURCE
from validation.run_gear_inp_feature_audit import build_feature_rows


pytestmark = pytest.mark.skipif(not DEFAULT_SOURCE.exists(), reason="commercial gear input is not present")


def test_gear_inp_feature_audit_marks_hard_contact_and_torque_gap() -> None:
    rows = build_feature_rows(DEFAULT_SOURCE)
    by_feature = {str(row["feature"]): row for row in rows}

    assert by_feature["c3d4_tet4_volume_elements"]["sfc_status"] == "supported"
    assert "missing" in str(by_feature["surface_to_surface_hard_contact"]["sfc_status"])
    assert by_feature["moment_cload_on_rp"]["sfc_status"] == "missing"
    assert "38884 nodes" in str(by_feature["two_flexible_gear_parts"]["abaqus_usage"])
    assert "176486 C3D4" in str(by_feature["two_flexible_gear_parts"]["abaqus_usage"])
