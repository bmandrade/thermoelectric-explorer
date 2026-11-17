# -*- coding: utf-8 -*-
"""
Interactive GUI for Thermoelectric Explorer.

Sweeps the chemical potential and plots four quantities against it:

- the Seebeck coefficient S (absolute, microvolts/kelvin),
- the electrical conductivity sigma (absolute, S/m),
- the thermal conductivities -- electronic kappa_e (curve) and lattice
  kappa_lattice (horizontal line) -- in W/m/K,
- the dimensionless figure of merit ZT = S^2 sigma T / (kappa_e + kappa_lattice).

Because ZT needs absolute transport, the GUI uses
:mod:`thermoelectric.figure_of_merit` rather than the arbitrary-units
power factor: a relaxation-time slider ``tau`` sets the absolute scale of
sigma and kappa_e, and ``kappa_bulk`` / phonon mean free path control the
boundary-scattering lattice model. Shrinking a box dimension lowers
kappa_lattice (boundary scattering) and sharpens the DOS, both of which
raise ZT. The Fermi level of the current box is marked on every panel, and
the chemical potential of peak ZT (the optimal doping) is highlighted.

Launch it with::

    thermoelectric-gui

or::

    python -m thermoelectric.gui

matplotlib is imported lazily, so importing ``thermoelectric`` stays
matplotlib-free; the GUI requires the ``[gui]`` extra. This module contains
no physics: every number comes from :mod:`thermoelectric.figure_of_merit`
and the Quantum DOS Explorer. Per the course convention, visualization code
is not unit-tested; the physics it displays is tested in
``tests/test_transport.py`` and ``tests/test_figure_of_merit.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantum_dos.analysis import classify_regime
from quantum_dos.constants import SILVER_ELECTRON_DENSITY_M3
from quantum_dos.models import DOSResult, QuantumBox
from quantum_dos.physics import calculate_dos

from .figure_of_merit import figure_of_merit
from .transport import interior_mu_grid

__all__ = ["main", "build_app", "DosCache"]

# ---------------------------------------------------------------------------
# Presentation constants (this module's own dark palette; no physics here).
# ---------------------------------------------------------------------------
BG = "#15181e"       # window background
PANEL = "#1d212b"    # axes/panel background
GRID = "#2c3340"     # gridlines / borders
TEXT = "#e6e9ef"     # primary text
MUTED = "#8b94a7"    # secondary text
BLUE = "#5aa9e6"     # Seebeck
GREEN = "#6ccf8e"    # conductivity
ORANGE = "#e8a24c"   # electronic thermal conductivity
PURPLE = "#b48ead"   # lattice thermal conductivity
YELLOW = "#e5c07b"   # ZT
RED = "#e06c75"      # Fermi level

#: DOS energy grid and chemical-potential sweep resolution. A moderate
#: grid keeps slider dragging responsive while staying fine enough for the
#: transport moment integrals.
E_MAX_EV = 12.0
N_POINTS = 800
N_SWEEP = 160

#: Density is held at bulk silver's value (there is no carrier-density
#: slider), so the Fermi-energy filling always assumes this density.
DENSITY_M3 = SILVER_ELECTRON_DENSITY_M3

DEFAULTS = {
    "lx_nm": 10.0,
    "ly_nm": 10.0,
    "lz_nm": 3.0,
    "effective_mass": 1.0,
    "temperature_K": 300.0,
    "sigma_eV": 0.05,
    "tau_1e14_s": 1.0,       # relaxation time in units of 1e-14 s
    "kappa_bulk": 1.5,       # W/m/K
    "phonon_mfp_nm": 5.0,    # nm
}


@dataclass
class _CurveKey:
    """Inputs that determine the DOS curve (not the transport sweep).

    If these are unchanged, the cached density of states can be reused and
    only the transport sweep (temperature, tau, lattice parameters) needs
    recomputing.
    """

    lx_nm: float
    ly_nm: float
    lz_nm: float
    effective_mass: float
    sigma_eV: float


class DosCache:
    """Cache the last density-of-states computation for the GUI.

    Recomputing the DOS means re-enumerating and broadening the box's
    states, which is the expensive step. Moving the temperature, relaxation
    time or lattice sliders does not change the DOS (only the transport
    sweep built from it), so this cache returns the stored
    :class:`~quantum_dos.models.DOSResult` whenever the geometry, effective
    mass and broadening are unchanged, and only triggers a full
    :func:`~quantum_dos.physics.calculate_dos` otherwise. It holds no
    matplotlib state.
    """

    def __init__(self) -> None:
        """Create an empty cache holding no previous result."""
        self._key: Optional[_CurveKey] = None
        self._result: Optional[DOSResult] = None

    def get(self, box: QuantumBox, sigma_eV: float) -> DOSResult:
        """Return a DOSResult for this box, recomputing only when needed.

        The result is always numerically identical to calling
        :func:`~quantum_dos.physics.calculate_dos` directly with the fixed
        silver density and the module's grid settings.
        """
        key = _CurveKey(box.lx_nm, box.ly_nm, box.lz_nm,
                        box.effective_mass, sigma_eV)
        if self._key == key and self._result is not None:
            return self._result

        result = calculate_dos(
            box, sigma_eV=sigma_eV, electron_density_m3=DENSITY_M3,
            e_max_eV=E_MAX_EV, n_points=N_POINTS,
        )
        self._key = key
        self._result = result
        return result


def _sweep_zt(dos, box, temperature, tau_s, broadening_eV, kappa_bulk, mfp_nm):
    """Evaluate S, sigma, kappa_e, kappa_lattice and ZT over a mu sweep.

    Returns (mu, S_uV, sigma_Spm, kappa_e, kappa_lattice, ZT) arrays, with
    kappa_lattice a scalar (it depends only on the box size, not on mu).
    """
    mu_grid = interior_mu_grid(dos, temperature, n=N_SWEEP)
    results = [
        figure_of_merit(
            dos, box=box, temperature_K=temperature, tau_s=tau_s,
            broadening_eV=broadening_eV, mu_eV=float(mu),
            kappa_bulk_W_per_mK=kappa_bulk, phonon_mfp_nm=mfp_nm,
        )
        for mu in mu_grid
    ]
    seebeck = np.array([r.seebeck_uV_per_K for r in results])
    sigma = np.array([r.conductivity_S_per_m for r in results])
    kappa_e = np.array([r.kappa_e_W_per_mK for r in results])
    zt = np.array([r.zt for r in results])
    kappa_lat = results[0].kappa_lattice_W_per_mK
    return mu_grid, seebeck, sigma, kappa_e, kappa_lat, zt


def build_app(plt, Slider):
    """Build the GUI figure, sliders and update logic.

    Injecting ``plt`` and ``Slider`` keeps this module import-free of
    matplotlib. Returns a dict of handles (figure, sliders, the
    compute-and-draw callback, the cache) so the app can be driven
    headlessly if needed.
    """
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    try:
        fig.canvas.manager.set_window_title("Thermoelectric Explorer")
    except Exception:
        pass

    fig.text(0.155, 0.967, "Thermoelectric Explorer", ha="center",
             fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.155, 0.947, "absolute transport  |  S, sigma, kappa, ZT",
             ha="center", fontsize=8, color=MUTED)

    # Four stacked panels sharing the chemical-potential axis.
    ax_s = fig.add_axes([0.42, 0.760, 0.55, 0.170])
    ax_sig = fig.add_axes([0.42, 0.540, 0.55, 0.170], sharex=ax_s)
    ax_kap = fig.add_axes([0.42, 0.320, 0.55, 0.170], sharex=ax_s)
    ax_zt = fig.add_axes([0.42, 0.095, 0.55, 0.170], sharex=ax_s)

    ax_s.set_ylabel("S  (uV/K)", fontsize=10, color=BLUE)
    ax_sig.set_ylabel("sigma  (S/m)", fontsize=10, color=GREEN)
    ax_kap.set_ylabel("kappa  (W/m/K)", fontsize=10, color=ORANGE)
    ax_zt.set_ylabel("ZT", fontsize=10, color=YELLOW)
    ax_zt.set_xlabel("Chemical potential  mu  (eV)", fontsize=11)
    ax_s.set_xlim(0, E_MAX_EV)
    ax_sig.set_yscale("log")
    ax_kap.set_yscale("log")
    for a in (ax_s, ax_sig, ax_kap):
        plt.setp(a.get_xticklabels(), visible=False)

    mu0 = np.linspace(0.01, E_MAX_EV, N_SWEEP)
    zeros = np.zeros(N_SWEEP)
    ones = np.ones(N_SWEEP)  # positive placeholder for the log-scaled panels
    line_s, = ax_s.plot(mu0, zeros, color=BLUE, lw=2.0)
    line_sig, = ax_sig.plot(mu0, ones, color=GREEN, lw=2.0)
    line_ke, = ax_kap.plot(mu0, ones, color=ORANGE, lw=2.0, label="kappa_e")
    line_kl, = ax_kap.plot(mu0, ones, color=PURPLE, lw=1.6, ls="--",
                           label="kappa_lattice")
    line_zt, = ax_zt.plot(mu0, zeros, color=YELLOW, lw=2.0)
    dot_zt, = ax_zt.plot([], [], "o", color=RED, ms=6)  # peak-ZT marker
    ax_s.axhline(0.0, color=MUTED, lw=0.8, ls=":")       # S changes sign
    ax_kap.legend(loc="upper left", fontsize=7, framealpha=0.3,
                  labelcolor=TEXT)
    vlines = [a.axvline(x=1.0, color=RED, lw=1.2, ls="--", alpha=0.8)
              for a in (ax_s, ax_sig, ax_kap, ax_zt)]

    regime_badge = ax_s.text(
        0.015, 0.90, "", transform=ax_s.transAxes, va="top", ha="left",
        fontsize=10, fontweight="bold", color=BLUE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL,
                  edgecolor=BLUE, alpha=0.85, linewidth=1.2),
    )

    def _sl(rect, label, lo, hi, val, step, color=BLUE):
        a = fig.add_axes(rect)
        a.set_facecolor(BG)
        sl = Slider(a, label, lo, hi, valinit=val, valstep=step, color=color)
        sl.label.set_color(TEXT); sl.label.set_fontsize(9)
        sl.valtext.set_color(color); sl.valtext.set_fontsize(9)
        return sl

    def _section(y, label):
        fig.text(0.155, y, label, ha="center", fontsize=9, color=MUTED,
                 fontstyle="italic")

    _section(0.918, "Box geometry")
    sl_lx = _sl([0.06, 0.886, 0.18, 0.022], "Lx (nm)", 0.5, 20.0,
                DEFAULTS["lx_nm"], 0.5)
    sl_ly = _sl([0.06, 0.850, 0.18, 0.022], "Ly (nm)", 0.5, 20.0,
                DEFAULTS["ly_nm"], 0.5)
    sl_lz = _sl([0.06, 0.814, 0.18, 0.022], "Lz (nm)", 0.5, 20.0,
                DEFAULTS["lz_nm"], 0.5)

    _section(0.783, "Physics parameters")
    sl_mass = _sl([0.06, 0.751, 0.18, 0.022], "m* (xme)", 0.1, 2.0,
                  DEFAULTS["effective_mass"], 0.05)
    sl_temp = _sl([0.06, 0.715, 0.18, 0.022], "T   (K)", 10.0, 1000.0,
                  DEFAULTS["temperature_K"], 10.0)
    sl_sigma = _sl([0.06, 0.679, 0.18, 0.022], "sig (eV)", 0.01, 0.5,
                   DEFAULTS["sigma_eV"], 0.01)

    _section(0.648, "Transport & lattice")
    sl_tau = _sl([0.06, 0.616, 0.18, 0.022], "tau(e-14s)", 0.1, 10.0,
                 DEFAULTS["tau_1e14_s"], 0.1, color=GREEN)
    sl_kbulk = _sl([0.06, 0.580, 0.18, 0.022], "k_bulk", 0.2, 5.0,
                   DEFAULTS["kappa_bulk"], 0.1, color=PURPLE)
    sl_mfp = _sl([0.06, 0.544, 0.18, 0.022], "mfp (nm)", 1.0, 20.0,
                 DEFAULTS["phonon_mfp_nm"], 0.5, color=PURPLE)

    ax_info = fig.add_axes([0.02, 0.06, 0.27, 0.42])
    ax_info.set_facecolor(PANEL)
    ax_info.set_xticks([]); ax_info.set_yticks([])
    for sp in ax_info.spines.values():
        sp.set_edgecolor(GRID)
    info = ax_info.text(0.07, 0.95, "", transform=ax_info.transAxes, va="top",
                        ha="left", fontsize=9, color=TEXT, linespacing=1.7)

    cache = DosCache()

    def _compute_and_draw():
        box = QuantumBox(lx_nm=sl_lx.val, ly_nm=sl_ly.val, lz_nm=sl_lz.val,
                         effective_mass=sl_mass.val)
        temperature = sl_temp.val
        tau_s = sl_tau.val * 1e-14
        try:
            dos = cache.get(box, sigma_eV=sl_sigma.val)
            mu, s_uv, sig, ke, kl, zt = _sweep_zt(
                dos, box, temperature, tau_s, sl_sigma.val,
                sl_kbulk.val, sl_mfp.val,
            )
        except ValueError as exc:
            info.set_text(f"Cannot compute:\n\n{exc}")
            fig.canvas.draw_idle()
            return

        ef = dos.fermi_energy_eV
        line_s.set_data(mu, s_uv)
        line_sig.set_data(mu, np.where(sig > 0, sig, np.nan))
        line_ke.set_data(mu, np.where(ke > 0, ke, np.nan))
        line_kl.set_data(mu, np.full_like(mu, kl))
        line_zt.set_data(mu, zt)
        ax_s.set_xlim(mu[0], mu[-1])
        for v in vlines:
            v.set_xdata([ef, ef])

        for a in (ax_s, ax_sig, ax_kap, ax_zt):
            a.relim(); a.autoscale_view(scalex=False)

        # Peak ZT over the sweep (the optimal doping point).
        i_opt = int(np.argmax(zt))
        dot_zt.set_data([mu[i_opt]], [zt[i_opt]])

        i_ef = int(np.argmin(np.abs(mu - ef)))
        regime = classify_regime(box)
        regime_badge.set_text(regime)

        info.set_text(
            f"Regime :  {regime}\n"
            f"Ef     :  {ef:.3f} eV      T : {temperature:.0f} K\n"
            f"tau    :  {tau_s:.2e} s\n"
            f"k_latt :  {kl:.3f} W/m/K\n"
            f"\n"
            f"-- at Fermi level --\n"
            f"S      :  {s_uv[i_ef]:9.2f} uV/K\n"
            f"sigma  :  {sig[i_ef]:9.3g} S/m\n"
            f"kappa_e:  {ke[i_ef]:9.3g} W/m/K\n"
            f"ZT     :  {zt[i_ef]:9.4f}\n"
            f"\n"
            f"-- optimal doping --\n"
            f"ZT_max :  {zt[i_opt]:9.4f}\n"
            f"at mu  :  {mu[i_opt]:9.3f} eV\n"
            f"S there:  {s_uv[i_opt]:9.1f} uV/K"
        )
        fig.canvas.draw_idle()

    def _on_change(_=None):
        """Recompute and redraw when any slider changes.

        Computing synchronously in the callback is robust across matplotlib
        backends. The DOS cache makes temperature/tau/lattice changes cheap
        (no re-enumeration), so updates stay smooth for normal boxes.
        """
        _compute_and_draw()

    for slider in (sl_lx, sl_ly, sl_lz, sl_mass, sl_temp, sl_sigma,
                   sl_tau, sl_kbulk, sl_mfp):
        slider.on_changed(_on_change)

    _compute_and_draw()

    return {
        "fig": fig,
        "axes": {"s": ax_s, "sigma": ax_sig, "kappa": ax_kap, "zt": ax_zt},
        "sliders": {"lx": sl_lx, "ly": sl_ly, "lz": sl_lz, "mass": sl_mass,
                    "temp": sl_temp, "sigma": sl_sigma, "tau": sl_tau,
                    "kbulk": sl_kbulk, "mfp": sl_mfp},
        "compute_and_draw": _compute_and_draw,
        "cache": cache,
    }


def _select_interactive_backend() -> None:
    """Prefer a Qt backend for interactive use.

    On some very new Python/Tk combinations (e.g. Python 3.14) the default
    TkAgg backend has a slider bug: dragging a Slider snaps its value back
    to the initial position, so the controls appear unresponsive. Qt
    backends do not have this issue. If a Qt binding is installed we select
    QtAgg; otherwise matplotlib's default backend is left in place.

    Does nothing if a backend has already been chosen via the MPLBACKEND
    environment variable, so the user can always override this.
    """
    import os
    import matplotlib

    if os.environ.get("MPLBACKEND"):
        return

    for qt_binding, backend in (("PyQt6", "QtAgg"),
                                ("PySide6", "QtAgg"),
                                ("PyQt5", "QtAgg"),
                                ("PySide2", "QtAgg")):
        try:
            __import__(qt_binding)
        except ImportError:
            continue
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue


def main() -> None:
    """Launch the interactive thermoelectric explorer window."""
    _select_interactive_backend()
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    build_app(plt, Slider)
    plt.show()


if __name__ == "__main__":
    main()
