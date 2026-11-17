# -*- coding: utf-8 -*-
"""
Example: thermoelectric parameters versus size at FIXED chemical potential.

Holds the chemical potential (and temperature) fixed and shrinks the
confined dimension Lz, plotting the Seebeck coefficient S, electrical
conductivity sigma and power factor PF against Lz.

The point of this example is a cautionary one. Naively "smaller = better"
for low-dimensional thermoelectrics, but that is only true if you also
re-optimize the carrier concentration. At a *fixed* chemical potential,
shrinking the box mostly removes states, so:

- sigma falls steeply (fewer states at the fixed energy),
- S stays small and merely oscillates,
- PF = S^2 * sigma therefore falls as well.

The oscillations are physical: as Lz changes, the quantized sub-band edges
(which move as 1/Lz^2) sweep through the fixed chemical potential, giving
quantum size effects (Sandomirskii-type oscillations in thin films). The
Hicks-Dresselhaus enhancement lives in the density-of-states *shape at an
optimally placed Fermi level*; to see it you must move the chemical
potential onto each new sub-band edge as you confine -- which this fixed-mu
scan deliberately does not do, to make the contrast clear.

Run with::

    python examples/size_scan.py

Needs matplotlib (the ``[gui]`` extra) to draw the figure; without it, the
numerical summary is still printed.

All conductivity/power-factor values are in arbitrary units (CRTA; the
density of states is un-normalized). The Seebeck coefficient is absolute,
in microvolts/kelvin.
"""

import numpy as np

from quantum_dos import QuantumBox, calculate_dos
from thermoelectric import TransportParams, thermoelectric_point

# Fixed operating point and the two unconfined axes.
MU_EV = 4.0
TEMPERATURE_K = 300.0
LX_NM = LY_NM = 10.0
DENSITY_M3 = 5.86e28
SIGMA_EV = 0.05
LZ_NM = np.linspace(0.8, 12.0, 120)


def scan():
    """Return (Lz, S, sigma, PF) arrays over the Lz range at fixed mu."""
    seebeck = np.empty_like(LZ_NM)
    conductivity = np.empty_like(LZ_NM)
    powerfactor = np.empty_like(LZ_NM)
    for i, lz in enumerate(LZ_NM):
        box = QuantumBox(lx_nm=LX_NM, ly_nm=LY_NM, lz_nm=lz, effective_mass=1.0)
        dos = calculate_dos(box, sigma_eV=SIGMA_EV,
                            electron_density_m3=DENSITY_M3, n_points=900)
        p = thermoelectric_point(
            dos, TransportParams(temperature_K=TEMPERATURE_K, mu_eV=MU_EV)
        )
        seebeck[i] = p.seebeck_uV_per_K
        conductivity[i] = p.conductivity
        powerfactor[i] = p.power_factor
    return LZ_NM, seebeck, conductivity, powerfactor


def main() -> None:
    lz, s, sigma, pf = scan()

    print(f"Fixed mu = {MU_EV} eV, T = {TEMPERATURE_K} K, "
          f"Lx = Ly = {LX_NM} nm")
    print(f"{'Lz(nm)':>7} {'S(uV/K)':>9} {'sigma(arb)':>11} {'PF(arb)':>11}")
    for target in (10.0, 6.0, 4.0, 2.0, 1.0):
        i = int(np.argmin(np.abs(lz - target)))
        print(f"{lz[i]:7.1f} {s[i]:9.2f} {sigma[i]:11.3g} {pf[i]:11.3g}")
    print("\nNote: at fixed mu, sigma and PF FALL as the box shrinks. The "
          "Hicks-Dresselhaus\nenhancement requires re-optimizing mu onto a "
          "sub-band edge as you confine.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: install the [gui] extra to plot.)")
        return

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(lz, s, lw=1.8)
    ax[0].axhline(0.0, ls=":", lw=0.8)
    ax[0].set_ylabel("S  (uV/K)")
    ax[1].plot(lz, sigma, lw=1.8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("sigma  (arb.)")
    ax[2].plot(lz, pf, lw=1.8)
    ax[2].set_yscale("log")
    ax[2].set_ylabel("PF  (arb.)")
    ax[2].set_xlabel(f"Lz  (nm)   [Lx=Ly={LX_NM:g} nm,  mu={MU_EV:g} eV,  "
                     f"T={TEMPERATURE_K:g} K]")
    ax[0].set_title("Thermoelectric parameters vs size at fixed chemical "
                    "potential")
    fig.tight_layout()
    fig.savefig("size_scan.png", dpi=120)
    print("\nSaved figure to size_scan.png")
    plt.show()


if __name__ == "__main__":
    main()
