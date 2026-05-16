# CalculiX Contact Reference Candidate Analysis

## Purpose

The Phase-8 rigid-flat-indenter paper figure has been retired from the bound
paper experiment set. This note records which local CalculiX contact examples
are available and whether they are suitable as external contact evidence for
the SFC FEM-induced dynamic-SDF method.

## Local Sources Checked

- CalculiX executable: `/usr/bin/ccx` from Debian package `calculix-ccx`
  version `2.17-3`.
- CalculiX package tests: `/usr/share/doc/calculix-ccx-test/examples/test`.
- Local source/test checkout: `/tmp/sfc_calculix_source/test`.

No web search was used for this audit.

## Key Finding

The local CalculiX test set contains many contact examples, but no official
example was found that simultaneously matches all of the current SFC paper
scope:

- C3D4/TET4 volume elements;
- frictionless normal contact;
- linear pressure-overclosure penalty law;
- clean CDIS/CSTR/CELS/RF output suitable for direct replay;
- a simple block-plane or block-block geometry;
- transient trajectory output suitable for a full 1 s or 3 s comparison.

Several examples are still useful as external contact references, but they are
not drop-in equivalents to the current SFC TET4 dynamic-SDF contact path.

## Candidate Examples

| Candidate | Local file | What it contains | Usefulness | Main mismatch |
| --- | --- | --- | --- | --- |
| Contact energy two-body compression | `/tmp/sfc_calculix_source/test/contactenergy.inp` | Two C3D8 solids, surface-to-surface contact, linear pressure-overclosure, node `U/RF`, element internal energy, and `CONTACT PRINT, TOTALS=YES` for `CELS`. | Best small external contact-law/energy evidence. It can validate reaction-force and contact-energy semantics and can be replayed by SFC current-surface gap queries. | Uses C3D8, not C3D4/TET4; static, not transient dynamics. |
| Beam between two plates | `/tmp/sfc_calculix_source/test/beamcontact.inp` | C3D20R/C3D8 model, two contact pairs, NLGEOM static, linear pressure-overclosure, `CONTACT FILE`. | Useful for multi-contact-zone behavior and contact output parsing once the simpler case is stable. | Uses higher-order and hex elements; geometry is more complex than the current TET4 block benchmarks. |
| Dynamic hard-contact cube cases | `/tmp/sfc_calculix_source/test/contact15.inp`, `contact15lin.inp`, `contact16.inp`, `contact17.inp`, `contact19.inp` | Dynamic contact examples, mostly C3D8, often hard contact and sometimes friction. | Potential future stress test for transient output parsing and trajectory diagnostics. | Hard contact/friction options do not match SFC's frictionless normal penalty contact; not ideal for proving current SDF query correctness. |
| Static C3D20 contact activation cases | `/tmp/sfc_calculix_source/test/contact13.inp`, `contact14.inp` | Two C3D20 bodies, contact pair activation/removal across steps, hard contact. | Useful for studying CalculiX contact lifecycle/model-change behavior. | C3D20 and hard contact; not a direct paper evidence case for current TET4 penalty contact. |
| Tetrahedral-looking interface tests | `/tmp/sfc_calculix_source/test/contact4tet.inp`, `contact4tetrefine.inp` | Despite the names, these are interface-loading/refinement tests with S8/C3D10 and no `*CONTACT PAIR`. | Not suitable as contact evidence. | They are not contact-pair tests and do not exercise CDIS/CSTR/CELS. |

## Recommended Path

For the current paper, the already locked
`paper/numerical_experiments/block_drop_dynamic_sdf_calculix_1s/` remains the
primary external contact trajectory evidence because it directly compares the
native SFC `persistent_dynamic_sdf_calculix_f2f` path against CalculiX on the
scoped block-plane benchmark.

The `contactenergy.inp` path has now been implemented as
`validation/run_calculix_contactenergy_replay.py` and locked under
`paper/numerical_experiments/calculix_contactenergy_c3d8_replay/`. The runner:

1. run the original CalculiX `contactenergy.inp`;
2. parse nodal `U/RF`, total internal energy, raw `CELS`, and total contact
   spring energy;
3. replay the deformed master/slave contact state with SFC dynamic-SDF local
   projection;
4. compare signed gaps, contact force resultants, contact energy, and active
   contact rows;
5. labels it clearly as a C3D8 CalculiX external contact-law/energy reference,
   not as a TET4 trajectory-equivalence claim.

If a strict TET4 paper contact reference is required, the better route is to
continue using the project-generated C3D4 block-plane CalculiX input already
used by `validation/run_geometric_nonlinear_contact_validation.py`, because the
local official CalculiX tests do not provide a matching C3D4 contact example.

## Paper Claim Boundary

Do not use the retired Phase-8 rigid indenter figure as manuscript evidence.
The contact claims should remain tied to:

- dynamic-SDF block-plane trajectory versus CalculiX for the scoped generated
  C3D4 case;
- SFC/scikit-fem agreement for non-contact linear FEM stress and displacement;
- analytic TET4 stress-strain patch correctness;
- optional future `contactenergy.inp` replay if it is implemented and gated.
