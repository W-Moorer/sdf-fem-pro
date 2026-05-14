from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_geometric_nonlinear_acceptance import (  # noqa: E402
    _cantilever_model,
    _parse_calculix_dat_displacements_strict,
    run_acceptance,
    stvk_patch_test,
    write_calculix_cantilever_input,
)


def test_stvk_patch_test_matches_analytic_reference() -> None:
    rows = stvk_patch_test([1])

    assert rows[0]["passed"] == "true"
    assert float(rows[0]["energy_rel_error"]) < 1.0e-12
    assert float(rows[0]["max_green_strain_error"]) < 1.0e-12


def test_calculix_cantilever_input_requests_nlgeom_static(tmp_path: Path) -> None:
    model = _cantilever_model(1)
    inp = tmp_path / "cantilever.inp"
    write_calculix_cantilever_input(model, inp)

    text = inp.read_text(encoding="utf-8").lower()
    assert "*step,nlgeom" in text
    assert "*static" in text
    assert "*node print,nset=nall" in text
    assert "*node print,nset=nfixed,totals=only,global=yes" in text
    assert "*el print,elset=eall" in text
    assert "*cload" in text


def test_strict_calculix_displacement_parser_ignores_stress_rows(tmp_path: Path) -> None:
    dat = tmp_path / "probe.dat"
    dat.write_text(
        """
 displacements (vx,vy,vz) for set NALL and time  0.1000000E+01

         1  1.000000E+00  2.000000E+00  3.000000E+00
         2  4.000000E+00  5.000000E+00  6.000000E+00

 stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EALL and time  0.1000000E+01

         1   1 -9.000000E+00 -9.000000E+00 -9.000000E+00 -9.000000E+00 -9.000000E+00 -9.000000E+00
""",
        encoding="utf-8",
    )

    parsed = _parse_calculix_dat_displacements_strict(dat, np.asarray([1, 2], dtype=int))

    assert np.allclose(parsed[1.0][0], [1.0, 2.0, 3.0])
    assert np.allclose(parsed[1.0][1], [4.0, 5.0, 6.0])


def test_geometric_nonlinear_acceptance_quick_outputs(tmp_path: Path) -> None:
    outputs = run_acceptance(tmp_path, quick=True, run_calculix=False)

    for path in outputs.values():
        assert path.exists()

    with outputs["claims"].open(newline="", encoding="utf-8") as f:
        claims = {row["claim"]: row for row in csv.DictReader(f)}
    assert claims["analytic_finite_deformation_stvk_patch"]["supported"] == "true"
    assert claims["mesh_convergence_trend_available"]["supported"] == "true"
    assert claims["timestep_convergence_trend_available"]["supported"] == "true"
    assert claims["no_contact_large_displacement_calculix_comparison"]["supported"] == "false"

    with outputs["contact"].open(newline="", encoding="utf-8") as f:
        contact_rows = list(csv.DictReader(f))
    assert contact_rows[0]["status"] == "passed"
