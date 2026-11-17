# -*- coding: utf-8 -*-
"""
Example: the dimensional crossover of ZT as the confinement size shrinks.

Reducing a dimension does two good things for the figure of merit ZT:
it sharpens the density of states (raising the Seebeck coefficient, the
Hicks-Dresselhaus effect) and it scatters phonons at the new boundary
(lowering the lattice thermal conductivity). This example shows both,
combined, as ZT, for three confinement geometries plotted against the
confined length ``d``:

- **3-D (bulk):** a 10x10x10 nm box that is *not* confined, so ZT is
  independent of ``d`` -- a flat reference line.
- **2-D (well):** confine one axis, box = 10 x 10 x d. As ``d`` shrinks,
  ZT rises.
- **1-D (wire):** confine two axes, box = 10 x d x d. ZT rises fastest.

For each point the chemical potential is optimized within the
lightly-doped window (ZT has a genuine optimum because the lattice
thermal conductivity bounds the denominator). The left panel shows the
characteristic density-of-states shapes behind the effect.

Run with::

    python examples/dimensional_crossover.py

Needs matplotlib (the ``[gui]`` extra). ZT values are order-of-magnitude
model results (CRTA, free-electron bands, a simple boundary-scattering
lattice model with illustrative parameters), not predictions for a
specific material.
"""

import numpy as np

from quantum_dos import QuantumBox, calculate_dos
from thermoelectric import figure_of_merit

TEMPERATURE_K = 300.0
DENSITY_M3 = 5.86e28
SIGMA_EV = 0.05
TAU_S = 1e-14
BULK_NM = 10.0
D_NM = np.linspace(1.0, 10.0, 40)
MU_WINDOW_EV = np.linspace(0.05, 3.0, 120)   # lightly-doped search window


def best_zt(dims_nm) -> float:
    """Return the ZT of a box, optimized over the lightly-doped window."""
    box = QuantumBox(lx_nm=dims_nm[0], ly_nm=dims_nm[1], lz_nm=dims_nm[2],
                     effective_mass=1.0)
    dos = calculate_dos(box, sigma_eV=SIGMA_EV,
                        electron_density_m3=DENSITY_M3, n_points=1600)
    best = 0.0
    for mu in MU_WINDOW_EV:
        if mu < dos.energy_eV[0] or mu > dos.energy_eV[-1]:
            continue
        zt = figure_of_merit(dos, box=box, temperature_K=TEMPERATURE_K,
                             tau_s=TAU_S, broadening_eV=SIGMA_EV, mu_eV=mu).zt
        best = max(best, zt)
    return best


def main() -> None:
    zt_3d = best_zt((BULK_NM, BULK_NM, BULK_NM))          # unconfined reference
    zt_2d = np.array([best_zt((BULK_NM, BULK_NM, d)) for d in D_NM])
    zt_1d = np.array([best_zt((BULK_NM, d, d)) for d in D_NM])

    print(f"T = {TEMPERATURE_K} K, tau = {TAU_S:.0e} s, "
          f"kappa_bulk = 1.5 W/m/K (defaults)")
    print(f"3-D bulk ZT (flat)      : {zt_3d:.3f}")
    print(f"{'d(nm)':>6} {'2-D well ZT':>12} {'1-D wire ZT':>12}")
    for target in (10.0, 5.0, 3.0, 2.0, 1.0):
        i = int(np.argmin(np.abs(D_NM - target)))
        print(f"{D_NM[i]:6.1f} {zt_2d[i]:12.3f} {zt_1d[i]:12.3f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: install the [gui] extra to plot.)")
        return

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: characteristic DOS shapes (schematic 3D/2D/1D free-electron).
    E = np.linspace(0.01, 3.0, 400)
    axl.plot(E, np.sqrt(E), label="3-D  ~ sqrt(E)")
    axl.plot(E, 0.9 * np.ones_like(E), label="2-D  ~ step")
    axl.plot(E, 0.6 / np.sqrt(E), label="1-D  ~ 1/sqrt(E)")
    axl.set_ylim(0, 3)
    axl.set_xlabel("Energy above sub-band edge (arb.)")
    axl.set_ylabel("DOS  g(E)  (arb.)")
    axl.set_title("Density-of-states shape by dimension")
    axl.legend()

    # Right: ZT vs confinement size for 3D (flat) / 2D / 1D.
    axr.axhline(zt_3d, ls="--", color="gray", label=f"3-D bulk (flat) = {zt_3d:.2f}")
    axr.plot(D_NM, zt_2d, marker="o", ms=3, label="2-D well (confine 1 axis)")
    axr.plot(D_NM, zt_1d, marker="s", ms=3, label="1-D wire (confine 2 axes)")
    axr.set_xlabel("confined length  d  (nm)")
    axr.set_ylabel("optimal ZT")
    axr.set_title("ZT vs confinement size")
    axr.legend()
    axr.invert_xaxis()   # size decreases to the right

    fig.suptitle("Dimensional crossover: ZT rises as confinement increases")
    fig.tight_layout()
    fig.savefig("dimensional_crossover.png", dpi=120)
    print("\nSaved figure to dimensional_crossover.png")
    plt.show()


if __name__ == "__main__":
    main()
