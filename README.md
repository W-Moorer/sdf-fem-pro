# sdf-fem-pro

Standalone finite-element dynamic signed-distance-field contact solver.

The core package is `sfc` under `src/sfc`. Abaqus-related code and data are
legacy validation material only and are not required by the core solver.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Implemented Modules

- TET4 volume mesh validation and boundary face extraction.
- TET4 small-strain linear elastic stiffness and mass matrices.
- Sparse global stiffness, mass, and gravity force assembly.
- Dirichlet DOF utilities and a linear Newmark-beta step.
- Point-triangle closest projection and oriented local dynamic surface distance.
- Uniform spatial-hash broad phase over padded triangle AABBs.
- Contact constraints, gap Jacobian, penalty force, and Gauss-Newton stiffness.
- Deterministic example and CI-sized benchmark scripts.

## Minimal Usage

```python
import numpy as np

from sfc.fem import DeformableBody, assemble_mass_matrix, assemble_stiffness_matrix
from sfc.mesh import VolumeMesh

mesh = VolumeMesh(
    X=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    elements=np.array([[0, 1, 2, 3]]),
)
body = DeformableBody(mesh=mesh, material={"E": 1.0e5, "nu": 0.3}, density=1.0)
K = assemble_stiffness_matrix(body)
M = assemble_mass_matrix(body)
```

## Examples

After editable install:

```bash
python examples/two_tet_patch.py
```

The example prints final vertical displacements for a deterministic two-tet
patch under gravity with one fixed node.

## Tests

```bash
pytest -q
```

## Benchmarks

```bash
python benchmarks/run_contact_benchmark.py --quick
python benchmarks/compare_bvh_vs_dynamic_sdf.py --quick
```

Benchmark outputs are written to `results/benchmarks/` by default.

## Validation

Physical and paper-evidence validation runners are kept outside the core
package:

```bash
python validation/run_phase7_physical_validation.py --quick --out-dir results/phase7
python validation/run_external_fem_comparison.py --quick --out-dir results/external_fem
python validation/run_phase8_engineering_cases.py --quick --out-dir results/phase8
python validation/run_external_contact_solver_comparison.py --quick --out-dir results/external_contact_solver
python validation/run_external_dynamic_contact_comparison.py --quick --out-dir results/external_dynamic_contact
```

The external FEM comparison uses `scikit-fem` as a validation-only open-source
reference for equivalent linear TET4 cantilever models. Phase 8 adds
engineering-style validation cases for a 3D beam, rigid flat indentation, and
deformable-deformable normal contact. The external contact comparison uses
SfePy as a validation-only open-source contact solver and replays its final
deformed contact state with SFC gap queries. The external dynamic contact
comparison uses SfePy transient elastodynamics with Newmark and mass-matrix
terms, plus SfePy penalty-contact snapshots along the same 3-second path.

## Current Limitations

- TET4 only.
- Small-strain linear FEM only.
- Penalty contact only.
- No friction.
- No self-contact.
- Broad phase is a uniform spatial hash, not a production BVH.
- The distance query is an oriented local surface distance; robust global
  closed-surface sign classification for arbitrary non-manifold geometry is not
  implemented.
