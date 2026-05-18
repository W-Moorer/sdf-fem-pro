"""Generate the paper method schematic for the dynamic narrow-band SDF pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]


def _box(ax, xy, size, title: str, body: str, *, face: str, edge: str = "#2b2b2b") -> None:
    x, y = xy
    w, h = size
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.012",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + 0.05 * w, y + h - 0.24 * h, title, ha="left", va="top", fontsize=9.0, fontweight="bold")
    ax.text(x + 0.05 * w, y + h - 0.48 * h, body, ha="left", va="top", fontsize=7.2, linespacing=1.12)


def _arrow(ax, start, end, *, text: str | None = None, rad: float = 0.0) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.05,
        color="#333333",
        shrinkA=2.5,
        shrinkB=2.5,
    )
    ax.add_patch(arrow)
    if text:
        mx = 0.5 * (start[0] + end[0])
        my = 0.5 * (start[1] + end[1])
        ax.text(
            mx,
            my + 0.04,
            text,
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.82},
        )


def generate(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    build_face = "#eaf2f8"
    query_face = "#eef6ef"
    payload_face = "#fff4df"
    force_face = "#f4edf7"

    ax.text(
        0.02,
        0.96,
        "Dynamic narrow-band SDF update and field-contact query",
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.905,
        "Projection is internal to field construction; contact queries interpolate the current SDF field.",
        ha="left",
        va="top",
        fontsize=8.6,
        color="#333333",
    )

    boxes = {
        "state": ((0.03, 0.67), (0.18, 0.15), "Current FEM state", r"$\mathbf{x}(\mathbf{q})$, velocities" "\n" r"boundary faces $\Gamma_h$"),
        "band": ((0.28, 0.67), (0.18, 0.15), "Narrow-band grid", r"current-space origin" "\n" r"spacing $h$, width $\rho$"),
        "populate": ((0.53, 0.67), (0.20, 0.15), "Grid population", "spatial hash + projection\nonly at grid nodes"),
        "payload": ((0.79, 0.67), (0.18, 0.15), "SDF payload grids", r"$\Phi_\ell, \nabla\Phi$ support" "\n" r"$f^*_\ell,\xi^*_\ell,\mathbf{n}_\ell,m_\ell$"),
        "samples": ((0.03, 0.32), (0.18, 0.15), "Slave samples", "nodes or surface\nquadrature points"),
        "interp": ((0.28, 0.32), (0.18, 0.15), "Interpolation query", r"$g=\hat\phi(\mathbf{x})$" "\n" r"$\nabla\hat\phi=\sum\Phi_\ell\nabla w_\ell$"),
        "jac": ((0.53, 0.32), (0.20, 0.15), "Contact sensitivity", r"slave: $N_A\nabla\hat\phi^T$" "\n" r"master: payload weights"),
        "force": ((0.79, 0.32), (0.18, 0.15), "Penalty response", r"$E_c=\frac{1}{2}k\langle-g\rangle_+^2$" "\n" "force and tangent"),
    }

    for key, (xy, size, title, body) in boxes.items():
        face = build_face if key in {"state", "band", "populate"} else payload_face if key == "payload" else query_face if key in {"samples", "interp"} else force_face
        _box(ax, xy, size, title, body, face=face)

    _arrow(ax, (0.21, 0.745), (0.28, 0.745), text="extract")
    _arrow(ax, (0.46, 0.745), (0.53, 0.745), text="visit nodes")
    _arrow(ax, (0.73, 0.745), (0.79, 0.745), text="store")
    _arrow(ax, (0.21, 0.395), (0.28, 0.395), text="points")
    _arrow(ax, (0.46, 0.395), (0.53, 0.395), text="payload")
    _arrow(ax, (0.73, 0.395), (0.79, 0.395), text="assemble")
    _arrow(ax, (0.88, 0.67), (0.37, 0.47), text="interpolation-only query", rad=0.08)

    ax.plot([0.02, 0.98], [0.565, 0.565], color="#555555", linewidth=0.8, linestyle="--")
    ax.text(0.025, 0.585, "field update", fontsize=8.2, fontweight="bold", color="#333333")
    ax.text(0.025, 0.535, "contact query and assembly", fontsize=8.2, fontweight="bold", color="#333333")

    ax.text(
        0.5,
        0.11,
        r"Amortization gate: $T_\mathrm{update}+Q\,T_\mathrm{field}+Q_rT_\mathrm{refine}<Q\,T_\mathrm{projection}$",
        ha="center",
        va="center",
        fontsize=9.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f7f7f7", "edgecolor": "#888888", "linewidth": 0.7},
    )

    png = out_dir / "dynamic_sdf_method_schematic.png"
    pdf = out_dir / "dynamic_sdf_method_schematic.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "numerical_experiments" / "method_schematic" / "figures")
    return parser.parse_args()


def main() -> int:
    outputs = generate(parse_args().out_dir)
    print("SDF method schematic generated.")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
