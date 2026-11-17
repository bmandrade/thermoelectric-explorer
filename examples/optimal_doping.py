# -*- coding: utf-8 -*-
"""
Example: thermoelectric performance at OPTIMAL doping, versus size.

This is the companion to ``size_scan.py``. There we held the chemical
potential fixed and saw performance fall as the box shrank. Here we do
the honest thing instead: for each size we re-optimize the doping (the
chemical potential) and report the best power factor the box can reach.
This is what actually realizes the Hicks-Dresselhaus idea -- a sharper,
more confined density of states gives a higher *achievable* power factor,
but only if you tune the carrier concentration to exploit it.

Two modelling choices make the comparison fair and trustworthy, and both
reflect genuine limitations of the bare free-electron + constant
relaxation-time (CRTA) model:

1. **Optimise within a lightly-doped window near the conduction-band
   edge.** In a single parabolic band the Seebeck coefficient diverges as
   the chemical potential approaches the band edge (Mott: S ~ 1/E), while
   the conductivity grows without bound as the band is filled. So neither
   S nor the power factor has a clean interior optimum over the whole
   band -- the extrema sit at the boundaries, which is unphysical. Real
   thermoelectrics operate lightly doped, a few k_B*T from a band edge, so
   we search for the best power factor only in that window. A quantitative
   optimum over the full band needs an energy-dependent tau(E) and a
   finite/realistic band, which this model does not have.

2. **Report the power factor per unit volume.** The density of states
   here is extensive (a bigger box simply has more states), so the raw
   arbitrary-units power factor scales with volume. Dividing by the box
   volume removes that trivial size scaling, leaving an intensive quantity
   whose change reflects the density-of-states *shape*, i.e. the
   confinement physics we care about.

With those choices, the optimal (per-volume) power factor rises as the
box is confined -- the Hicks-Dresselhaus enhancement -- and the optimal
doping level shifts, showing you must re-dope as you confine.

Run with::

    python examples/optimal_doping.py

Needs matplotlib (the ``[gui]`` extra) to draw the figure; without it the
numerical summary is still printed. The Seebeck coefficient is absolute
(microvolts/kelvin); power factor is per-volume arbitrary units.
"""

import numpy as np

from quantum_dos import QuantumBox, calculate_dos
from quantum_dos.constants import BOLTZMANN, ELECTRON_MASS, EV_TO_J, HBAR
from thermoelectric import thermoelectric_sweep

TEMPERATURE_K = 300.0
LX_NM = LY_NM = 10.0
DENSITY_M3 = 5.86e28
SIGMA_EV = 0.05
LZ_NM = np.linspace(1.0, 12.0, 80)

#: Width of the lightly-doped search window above the band edge, in eV.
#: Wide enough to contain the near-edge power-factor optimum, narrow
#: enough to stay out of the unphysical high-filling regime.
DOPING_WINDOW_EV = 0.6


def band_edge_eV(box: QuantumBox) -> float:
    """Ground-state (conduction band edge) energy E(1,1,1) of the box, in eV."""
    mass_kg = box.effective_mass * ELECTRON_MASS
    l_m = np.array(box.dimensions_nm) * 1e-9
    return float(np.sum((np.pi**2 * HBAR**2) / (2.0 * mass_kg * l_m**2) / EV_TO_J))


def scan():
    """Return (Lz, best |S|, optimal PF per volume, optimal reduced doping).

    For each Lz the power factor is maximised over a lightly-doped window
    just above the band edge; the returned Seebeck coefficient and reduced
    doping (mu* - band edge) are taken at that optimum.
    """
    kT_eV = (BOLTZMANN / EV_TO_J) * TEMPERATURE_K
    seebeck_opt = np.empty_like(LZ_NM)
    pf_per_vol = np.empty_like(LZ_NM)
    reduced_mu = np.empty_like(LZ_NM)
    for i, lz in enumerate(LZ_NM):
        box = QuantumBox(lx_nm=LX_NM, ly_nm=LY_NM, lz_nm=lz, effective_mass=1.0)
        volume_nm3 = LX_NM * LY_NM * lz
        dos = calculate_dos(box, sigma_eV=SIGMA_EV,
                            electron_density_m3=DENSITY_M3, n_points=1500)
        e0 = band_edge_eV(box)
        mu_lo = max(dos.energy_eV[0], e0 - 2.0 * kT_eV)
        mu_grid = np.linspace(mu_lo, e0 + DOPING_WINDOW_EV, 400)
        sweep = thermoelectric_sweep(dos, mu_grid, TEMPERATURE_K)
        j = int(np.argmax(sweep.power_factor))
        seebeck_opt[i] = sweep.seebeck_uV_per_K[j]
        pf_per_vol[i] = sweep.power_factor[j] / volume_nm3
        reduced_mu[i] = sweep.mu_eV[j] - e0
    return LZ_NM, seebeck_opt, pf_per_vol, reduced_mu


def main() -> None:
    lz, s_opt, pf_v, red_mu = scan()

    print(f"Optimal doping within {DOPING_WINDOW_EV} eV of the band edge, "
          f"T = {TEMPERATURE_K} K, Lx = Ly = {LX_NM} nm")
    print(f"{'Lz(nm)':>7} {'S_opt(uV/K)':>12} {'PF/vol(arb)':>12} "
          f"{'mu*-Eedge(eV)':>14}")
    for target in (12.0, 8.0, 4.0, 2.0, 1.5, 1.0):
        i = int(np.argmin(np.abs(lz - target)))
        print(f"{lz[i]:7.1f} {s_opt[i]:12.1f} {pf_v[i]:12.3g} {red_mu[i]:14.3f}")
    print("\nThe per-volume optimal power factor rises as the box is confined "
          "(Hicks-Dresselhaus),\nwhile the optimal doping level shifts: you "
          "must re-dope to realize the gain.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: install the [gui] extra to plot.)")
        return

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(lz, pf_v, lw=1.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("optimal PF / volume\n(arb.)")
    ax[1].plot(lz, np.abs(s_opt), lw=1.8)
    ax[1].set_ylabel("|S| at optimum\n(uV/K)")
    ax[2].plot(lz, red_mu, lw=1.8)
    ax[2].set_ylabel("optimal doping\nmu* - E_edge (eV)")
    ax[2].set_xlabel(f"Lz  (nm)   [Lx=Ly={LX_NM:g} nm,  T={TEMPERATURE_K:g} K]")
    ax[0].set_title("Thermoelectric performance at optimal doping vs size")
    fig.tight_layout()
    fig.savefig("optimal_doping.png", dpi=120)
    print("\nSaved figure to optimal_doping.png")
    plt.show()


if __name__ == "__main__":
    main()
