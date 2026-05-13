# Dynamic Integrator and Gentle Contact Handoff

## Purpose

The previous strong long drop-impact diagnostic is not acceptable as
validation evidence because CalculiX did not complete the run and the original
height diagnostic used a nodal average instead of a mass-weighted center of
mass. A second mismatch was the contact geometry: CalculiX contacts the shell
`SPOS` surface at `floor_z + 0.005`, while the earlier diagnostic reported gaps
to the shell mid-surface at `floor_z`.

This handoff separates the validation into two layers:

1. no-contact analytic dynamics, to verify the integrator path without contact;
2. a gentler external block/plane contact reference, to avoid strong-impact
   CalculiX convergence failures.

## Commands

No-contact analytic dynamics:

```bash
python validation/run_dynamic_integrator_benchmark.py --quick --out-dir results/dynamic_integrator_benchmark
```

Gentle CalculiX block/plane contact:

```bash
python validation/run_calculix_drop_impact_comparison.py --quick --case block_drop --duration 1.0 --dt 0.002 --output-frequency 1 --initial-velocity-z 0.0 --gravity 9.81 --contact-stiffness 5000 --out-dir results/calculix_gentle_contact_reference
```

Tests:

```bash
pytest -q
```

## Outputs

No-contact dynamics:

- `results/dynamic_integrator_benchmark/dynamic_integrator_timeseries.csv`
- `results/dynamic_integrator_benchmark/dynamic_integrator_metrics.csv`
- `results/dynamic_integrator_benchmark/dynamic_integrator_summary.md`
- displacement and energy PNG/PDF plots for free fall and harmonic oscillator

Gentle contact:

- `results/calculix_gentle_contact_reference/calculix_drop_time_history.csv`
- `results/calculix_gentle_contact_reference/calculix_drop_metrics.csv`
- `results/calculix_gentle_contact_reference/calculix_drop_summary.md`
- `results/calculix_gentle_contact_reference/calculix_drop_z_cm.png`
- `results/calculix_gentle_contact_reference/calculix_drop_min_gap.png`
- `results/calculix_gentle_contact_reference/calculix_drop_force_proxy.png`
- `results/calculix_gentle_contact_reference/calculix_drop_total_energy.png`

## Current Metrics

No-contact dynamics quick mode:

| Case | Metric | Value | Status |
| --- | --- | ---: | --- |
| constant-acceleration free fall | max displacement error | `4.38e-12` | pass |
| constant-acceleration free fall | max velocity error | `6.41e-11` | pass |
| constant-acceleration free fall | relative total energy drift | `1.99e-10` | pass |
| undamped harmonic oscillator | max displacement error | `5.99e-06` | pass |
| undamped harmonic oscillator | max velocity error | `6.49e-05` | pass |
| undamped harmonic oscillator | relative total energy drift | `2.94e-12` | pass |

Gentle CalculiX contact:

| Metric | Value | Status |
| --- | ---: | --- |
| CalculiX completed | `true` | evidence |
| first contact time, CalculiX | `9.60e-02 s` | evidence |
| first contact time, SFC | `9.60e-02 s` | evidence |
| first contact time absolute error | `0.00e+00 s` | supported |
| z center-of-mass relative error | `1.054328e-02` | evidence |
| min-gap L-infinity error | `1.446052e-02` | evidence |
| max rebound z after contact, CalculiX | `4.448849e-01` | supported |
| max rebound z after contact, SFC | `4.439993e-01` | supported |
| physical rebound height claim | `supported` | supported |
| external dynamic contact claim | `supported` | supported |

## Interpretation

The no-contact benchmark supports the Newmark average-acceleration integrator
for the tested analytic cases. This does not validate contact.

The gentle 1-second block/plane case is a cleaner external contact reference
than the strong long drop-impact case. It is configured as a drop from rest
under gravity so the conservative rebound-height gate is meaningful: after
first contact the center-of-mass height must not exceed its starting height.

The SFC validation runner now uses a face-centroid, area-scaled penalty contact
approximation for this comparison, consistent with the CalculiX
`springforc_f2f.f` structure where clearance is evaluated at slave-face
integration points and scaled by spring area.

The strong sphere/block long diagnostic remains unsupported and should not
be used for paper evidence.

The strong diagnostic has been shortened from 3 seconds to a 1-second run. Its
latest 1-second output is still rejected by the physicality gate: the
`sphere_like_drop` CalculiX run timed out before completion, and the
`block_drop` run rebounded above the starting mass-center height in both
CalculiX and SFC. The strong case is therefore retained only as a failure
diagnostic for aggressive impact/contact settings.

## Remaining Risks

- The gentle contact case is low-speed and simple; it is not a replacement for
  high-speed impact validation.
- The SFC contact model is still penalty normal contact only.
- HHT-alpha contact dynamics is still algorithmically dissipative and should
  not be described as energy-conserving.
- CalculiX and SFC contact counts differ because their contact discretizations
  and reporting conventions are different.
