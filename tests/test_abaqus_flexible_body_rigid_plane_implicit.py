from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_flexible_body_rigid_plane_implicit import PENALTY_NORMAL_STIFFNESS, build_implicit_input_text


def test_implicit_cube_input_uses_standard_dynamic_contact_settings() -> None:
    text = build_implicit_input_text("cube_drop")

    assert "*Dynamic, Explicit" not in text
    assert "*Dynamic" in text
    assert "1.000000000000e-03, 3.000000000000e+00, 1.000000000000e-08, 1.000000000000e-03" in text
    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert f"{PENALTY_NORMAL_STIFFNESS:.12e}" in text
    assert "*Contact Inclusions, ALL EXTERIOR" in text
    assert text.index("*Contact") < text.index("*Step, name=IMPLICIT_FREE_FALL_CONTACT")
    assert "*Friction\n0." in text
    assert "*Element Output, elset=CUBE-1.CUBE_EALL, directions=YES" in text


def test_implicit_sphere_input_uses_standard_dynamic_contact_settings() -> None:
    text = build_implicit_input_text("sphere_drop")

    assert "*Dynamic, Explicit" not in text
    assert "*Dynamic" in text
    assert "*Element Output, elset=SPHERE-1.SPHERE_EALL, directions=YES" in text
    assert "FRICTIONLESS_PENALTY_CONTACT" in text
