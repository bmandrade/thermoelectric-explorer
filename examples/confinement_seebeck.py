# -*- coding: utf-8 -*-
"""
Example: how quantum confinement enhances the Seebeck coefficient.

Builds the density of states for a bulk-like box and a strongly confined
(quantum-well-like) box with the Quantum DOS Explorer, then sweeps the
chemical potential and compares the largest Seebeck coefficient each can
reach. This is a numerical illustration of the Hicks-Dresselhaus
argument that a sharper, more confined density of states raises the
thermoelectric response.

Run with::

    python examples/confinement_seebeck.py

Why compare the Seebeck coefficient and not the power factor
------------------------------------------------------------
The Seebeck coefficient is **absolute and scale-free**: the relaxation
time cancels in it, and because it is a logarithmic derivative of the
transport distribution, the arbitrary normalization of each box's DOS
cancels too. That makes S directly comparable between two different
systems. The conductivity and power factor are in **arbitrary units**
that differ from box to box (a larger box simply has more states, hence
a larger raw DOS), so comparing their absolute values across different
boxes is not meaningful -- which is exactly why this example compares S.
"""

import numpy as np

from quantum_dos import QuantumBox, calculate_dos
from thermoelectric import thermoelectric_sweep

TEMPERATURE_K = 300.0
DENSITY_M3 = 5.86e28  # bulk-silver-like electron density, illustrative
SIGMA_EV = 0.05

# Sweep the chemical potential over an interior window, kept a few kT
# away from the grid edges so the Fermi window is fully represented.
MU_MIN_EV, MU_MAX_EV = 1.0, 10.0


def max_abs_seebeck(box: QuantumBox) -> tuple[float, float]:
    """Return (mu_of_max_eV, max_abs_seebeck_uV_per_K) for one box."""
    dos = calculate_dos(
        box, sigma_eV=SIGMA_EV, electron_density_m3=DENSITY_M3, n_points=2000
    )
    mu_grid = np.linspace(MU_MIN_EV, MU_MAX_EV, 400)
    sweep = thermoelectric_sweep(dos, mu_grid, TEMPERATURE_K)
    i = int(np.argmax(np.abs(sweep.seebeck_uV_per_K)))
    return float(sweep.mu_eV[i]), float(sweep.seebeck_uV_per_K[i])


def main() -> None:
    bulk = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=10.0, effective_mass=1.0)
    confined = QuantumBox(lx_nm=10.0, ly_nm=10.0, lz_nm=1.5, effective_mass=1.0)

    mu_bulk, s_bulk = max_abs_seebeck(bulk)
    mu_conf, s_conf = max_abs_seebeck(confined)

    print("Largest Seebeck coefficient reached (absolute, microvolts/kelvin):")
    print(f"  bulk-like 10x10x10 nm : |S| = {abs(s_bulk):6.1f} uV/K "
          f"at mu = {mu_bulk:.3f} eV")
    print(f"  confined 10x10x1.5 nm : |S| = {abs(s_conf):6.1f} uV/K "
          f"at mu = {mu_conf:.3f} eV")
    ratio = abs(s_conf) / abs(s_bulk) if s_bulk else float("inf")
    print(f"  confined / bulk ratio : {ratio:.2f}x")
    print()
    print("The confined box reaches a larger Seebeck coefficient: the "
          "sharper\ndensity of states gives a steeper transport "
          "distribution, which is\nthe Hicks-Dresselhaus enhancement.")


if __name__ == "__main__":
    main()
