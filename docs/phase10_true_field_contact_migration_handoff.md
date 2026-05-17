# Phase-10 True Field Contact Migration Handoff

## Scope

This migration moves the first native TET4/HEX8 static and dynamic contact cases onto the true field path:

```text
current master surface
-> DynamicNarrowBandSDF.build(...)
-> field_contact interpolation query
-> penalty force / contact history
```

It also updates the C3D8 trajectory replay and Phase-9 HEX8/C3D8 contact adapter so those reusable validation paths no longer directly call the old `dynamic_surface_sdf(...)` query API.

## Changed Files

- `validation/run_phase10_true_field_contact_cases.py`
- `validation/run_calculix_contactenergy_replay.py`
- `validation/run_c3d8_contact_trajectory_validation.py`
- `validation/run_phase9_full_contact_validation.py`
- `tests/test_phase10_true_field_contact_cases.py`
- `tests/test_calculix_contactenergy_replay.py`
- `tests/test_c3d8_contact_trajectory_validation.py`
- `tests/test_phase9_full_contact_validation.py`
- `docs/phase10_existing_experiment_coverage_analysis.md`
- `docs/phase10_true_field_contact_migration_handoff.md`

## Native TET4/HEX8 Case Runner

Command:

```text
python validation/run_phase10_true_field_contact_cases.py --out-dir results/phase10_true_field_contact
```

Generated files:

- `results/phase10_true_field_contact/phase10_true_field_contact_cases.csv`
- `results/phase10_true_field_contact/phase10_true_field_contact_samples.csv`
- `results/phase10_true_field_contact/phase10_true_field_contact_history.csv`
- `results/phase10_true_field_contact/phase10_true_field_contact_summary.md`

Result table from `phase10_true_field_contact_cases.csv`:

| Case | Regime | Element | Nodes | Elements | Samples | Max gap error | Force rel. error | Max penetration | Reaction z | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `static_linear_tet4_field_contact` | static | tet4 | 18 | 20 | 9 | 0.000000e+00 | 0.000000e+00 | 2.000000e-02 | 2.700000e+01 | passed |
| `dynamic_linear_tet4_field_contact` | dynamic | tet4 | 18 | 20 | 9 | 6.635317e-17 | 1.605596e-14 | 1.381260e-02 | 1.000279e+01 | passed |
| `static_linear_hex8_field_contact` | static | hex8 | 18 | 4 | 9 | 0.000000e+00 | 0.000000e+00 | 2.000000e-02 | 2.700000e+01 | passed |
| `dynamic_linear_hex8_field_contact` | dynamic | hex8 | 18 | 4 | 9 | 4.857226e-17 | 5.009124e-14 | 1.537933e-02 | 9.585891e+00 | passed |

All four cases use `field_path=DynamicNarrowBandSDF+field_contact`.

## Legacy Adapter Migration

`validation/run_c3d8_contact_trajectory_validation.py`:

- Removed direct `dynamic_surface_sdf(...)` import and calls.
- Builds a cached `DynamicNarrowBandSDF` for the current master surface.
- Queries quadrature samples through `field_contact_constraint_from_sample(...)`.
- Keeps the existing CSV schema for compatibility.

`validation/run_calculix_contactenergy_replay.py`:

- Removed direct `dynamic_surface_sdf(...)` import and calls.
- Builds a cached `DynamicNarrowBandSDF` for the current C3D8 master surface.
- Uses field-contact constraints for contact-force replay, static contact-row assembly, and contact-pressure visualization.
- Keeps the existing CSV field names for compatibility with locked paper artifacts.

`validation/run_phase9_full_contact_validation.py`:

- Removed direct `dynamic_surface_sdf(...)` import and calls.
- Converts the reusable HEX8/C3D8 contact adapter to a field-backed adapter.
- Uses true field gap/normal/payload and projects vector master sensitivities back to scalar normal weights only to satisfy the legacy `ContactSample` assembly interface.

## Tests Run During Migration

Targeted:

```text
pytest -q tests/test_phase10_true_field_contact_cases.py
```

Result:

```text
2 passed in 1.15s
```

C3D8 trajectory migrated path:

```text
pytest -q tests/test_c3d8_contact_trajectory_validation.py::test_c3d8_contact_trajectory_quick_outputs_replay_files
```

Result:

```text
1 passed in 30.42s
```

Phase-9 migrated adapter smoke:

```text
pytest -q tests/test_phase9_full_contact_validation.py::test_phase9_quick_smoke_outputs_claim_gated_matrix
```

Result:

```text
1 passed in 60.12s
```

Contactenergy migrated path guard:

```text
pytest -q tests/test_calculix_contactenergy_replay.py::test_calculix_contactenergy_replay_uses_true_field_contact_path
```

Result:

```text
1 passed in 4.06s
```

Path guard tests:

```text
pytest -q tests/test_phase10_true_field_contact_cases.py tests/test_calculix_contactenergy_replay.py::test_calculix_contactenergy_replay_uses_true_field_contact_path tests/test_c3d8_contact_trajectory_validation.py::test_c3d8_trajectory_replay_uses_true_field_contact_path tests/test_phase9_full_contact_validation.py::test_phase9_hex8_contact_adapter_uses_true_field_contact_path
```

Result:

```text
5 passed in 7.66s
```

Full suite:

```text
pytest -q
```

Result:

```text
264 passed in 420.60s (0:07:00)
```

## Migrated C3D8/Phase-9 Outputs

Commands:

```text
python validation/run_c3d8_contact_trajectory_validation.py --quick --skip-calculix --out-dir results/c3d8_contact_trajectory_true_field_quick
python validation/run_calculix_contactenergy_replay.py --quick --skip-calculix --out-dir results/contactenergy_true_field_quick
python validation/run_phase9_full_contact_validation.py --quick --skip-calculix --out-dir results/phase9_true_field_quick
```

C3D8 true-field trajectory replay quick summary:

| Case | Rows | Active rows | Max penetration | Peak SFC normal force | Min-gap abs. error | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `block_block_c3d8` | 40 | 22 | 5.500000e-02 | 6.875000e+02 | 0.000000e+00 | synthetic_quick_check |
| `block_plane_c3d8` | 40 | 22 | 5.500000e-02 | 6.875000e+02 | 0.000000e+00 | synthetic_quick_check |

Phase-9 true-field quick matrix:

- `c3d8_linear_static_contact`: `supported`
- `c3d8_nonlinear_static_contact`: `supported`
- `c3d8_linear_dynamic_block_plane_contact`: `native_only_no_calculix`
- `c3d8_linear_dynamic_block_block_contact`: `native_only_no_calculix`
- `c3d8_nonlinear_dynamic_block_plane_contact`: `native_external_comparison_failed`
- `c3d8_nonlinear_dynamic_block_block_contact`: `native_external_comparison_failed`
- `c3d8_curved_nonplanar_contact_replay`: `blocked_no_calculix`

The nonlinear rows remain claim-gated supplementary evidence; the migration only changes their contact-query backend to true field contact.

C3D8 contactenergy true-field replay:

- contact force relative error: `3.798277448119336e-08`
- contact energy relative error: `3.7990359542993e-08`
- active quadrature count: `2`
- status: `ok`

## Claim Status

Supported:

- Native static linear TET4 true-field contact case.
- Native dynamic linear TET4 true-field contact history.
- Native static linear HEX8 true-field contact case.
- Native dynamic linear HEX8 true-field contact history.
- C3D8 contactenergy replay now uses the true field query path.
- C3D8 trajectory replay now uses the true field query path.
- Phase-9 HEX8/C3D8 contact adapter no longer directly calls the old projection-query API.

Not supported by this migration:

- Friction.
- Self-contact.
- Nonlinear FEM as the main method.
- GPU acceleration.
- Barrier contact.
- POD, neural SDF, or data-driven SDF.
- Abaqus-dependent core solve.
- Production BVH superiority.
- General arbitrary non-manifold global SDF robustness.

## Remaining Gap

The next paper-facing gap is total-step timing for contact-dominated workloads using the migrated true-field path. The field-query \(Q^\ast\) evidence already exists, but solver-level wall-clock evidence should include FEM assembly/solve/contact in one timing table.
