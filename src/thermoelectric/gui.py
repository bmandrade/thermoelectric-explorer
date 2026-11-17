# -*- coding: utf-8 -*-
"""
Interactive GUI for Thermoelectric Explorer.

Sweeps the chemical potential and plots the three thermoelectric
parameters -- Seebeck coefficient S (absolute, microvolts/kelvin),
electrical conductivity sigma and power factor PF (both arbitrary units)
-- as the box geometry, effective mass, temperature and broadening are
adjusted with sliders. The Fermi level of the current box is marked on
every panel.

Launch it with::

    thermoelectric-gui

or::

    python -m thermoelectric.gui

Like the scientific core, matplotlib is imported lazily, so importing
``thermoelectric`` stays matplotlib-free; the GUI requires the ``[gui]``
extra. This module contains no physics: every number comes from
:mod:`thermoelectric.transport` and the Quantum DOS Explorer. Per the
course convention, visualization code is not unit-tested; the logic it
displays (the transport functions) is tested in ``tests/test_transport.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantum_dos.analysis import classify_regime
from quantum_dos.constants import SILVER_ELECTRON_DENSITY_M3
from quantum_dos.models import DOSResult, QuantumBox
from quantum_dos.physics import calculate_dos

from .transport import interior_mu_grid, thermoelectric_sweep

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
ORANGE = "#e8a24c"   # power factor
RED = "#e06c75"      # Fermi level

#: DOS energy grid and chemical-potential sweep resolution. A moderate
#: grid keeps slider dragging responsive while staying fine enough for the
#: transport moment integrals.
E_MAX_EV = 12.0
N_POINTS = 800
N_SWEEP = 200

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
}


@dataclass
class _CurveKey:
    """Inputs that determine the DOS curve (not the temperature sweep).

    If these are unchanged, the cached density of states can be reused and
    only the chemical-potential sweep (which depends on temperature) needs
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
    states, which is the expensive step. Moving the temperature slider
    does not change the DOS (only the thermoelectric sweep built from it),
    so this cache returns the stored :class:`~quantum_dos.models.DOSResult`
    whenever the geometry, effective mass and broadening are unchanged, and
    only triggers a full :func:`~quantum_dos.physics.calculate_dos`
    otherwise. It holds no matplotlib state.
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

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor(BG)
    try:
        fig.canvas.manager.set_window_title("Thermoelectric Explorer")
    except Exception:
        pass

    fig.text(0.155, 0.963, "Thermoelectric Explorer", ha="center",
             fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.155, 0.940, "CRTA transport  |  S, sigma, power factor",
             ha="center", fontsize=8, color=MUTED)

    # Three stacked panels sharing the chemical-potential axis.
    ax_s = fig.add_axes([0.42, 0.70, 0.55, 0.23])
    ax_sig = fig.add_axes([0.42, 0.40, 0.55, 0.23], sharex=ax_s)
    ax_pf = fig.add_axes([0.42, 0.10, 0.55, 0.23], sharex=ax_s)

    ax_s.set_ylabel("S  (uV/K)", fontsize=10, color=BLUE)
    ax_sig.set_ylabel("sigma  (arb.)", fontsize=10, color=GREEN)
    ax_pf.set_ylabel("PF  (arb.)", fontsize=10, color=ORANGE)
    ax_pf.set_xlabel("Chemical potential  mu  (eV)", fontsize=11)
    ax_s.set_xlim(0, E_MAX_EV)
    for a in (ax_s, ax_sig):
        plt.setp(a.get_xticklabels(), visible=False)
    for a in (ax_sig, ax_pf):
        a.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        a.yaxis.get_offset_text().set_fontsize(7)

    mu0 = np.linspace(0.01, E_MAX_EV, N_SWEEP)
    line_s, = ax_s.plot(mu0, np.zeros(N_SWEEP), color=BLUE, lw=2.0)
    line_sig, = ax_sig.plot(mu0, np.zeros(N_SWEEP), color=GREEN, lw=2.0)
    line_pf, = ax_pf.plot(mu0, np.zeros(N_SWEEP), color=ORANGE, lw=2.0)
    ax_s.axhline(0.0, color=MUTED, lw=0.8, ls=":")  # S changes sign
    vlines = [a.axvline(x=1.0, color=RED, lw=1.3, ls="--", alpha=0.9)
              for a in (ax_s, ax_sig, ax_pf)]

    regime_badge = ax_s.text(
        0.015, 0.92, "", transform=ax_s.transAxes, va="top", ha="left",
        fontsize=10, fontweight="bold", color=BLUE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL,
                  edgecolor=BLUE, alpha=0.85, linewidth=1.2),
    )

    def _sl(rect, label, lo, hi, val, step, color=BLUE):
        a = fig.add_axes(rect)
        a.set_facecolor(BG)
        sl = Slider(a, label, lo, hi, valinit=val, valstep=step, color=color)
        sl.label.set_color(TEXT); sl.label.set_fontsize(10)
        sl.valtext.set_color(color); sl.valtext.set_fontsize(10)
        return sl

    def _section(y, label):
        fig.text(0.155, y, label, ha="center", fontsize=9, color=MUTED,
                 fontstyle="italic")

    _section(0.910, "Box geometry")
    sl_lx = _sl([0.06, 0.868, 0.18, 0.026], "Lx (nm)", 0.5, 20.0,
                DEFAULTS["lx_nm"], 0.5)
    sl_ly = _sl([0.06, 0.818, 0.18, 0.026], "Ly (nm)", 0.5, 20.0,
                DEFAULTS["ly_nm"], 0.5)
    sl_lz = _sl([0.06, 0.768, 0.18, 0.026], "Lz (nm)", 0.5, 20.0,
                DEFAULTS["lz_nm"], 0.5)

    _section(0.730, "Physics parameters")
    sl_mass = _sl([0.06, 0.690, 0.18, 0.026], "m* (xme)", 0.1, 2.0,
                  DEFAULTS["effective_mass"], 0.05)
    sl_temp = _sl([0.06, 0.645, 0.18, 0.026], "T   (K)", 10.0, 1000.0,
                  DEFAULTS["temperature_K"], 10.0)
    sl_sigma = _sl([0.06, 0.600, 0.18, 0.026], "sig (eV)", 0.01, 0.5,
                   DEFAULTS["sigma_eV"], 0.01)

    ax_info = fig.add_axes([0.02, 0.07, 0.27, 0.47])
    ax_info.set_facecolor(PANEL)
    ax_info.set_xticks([]); ax_info.set_yticks([])
    for sp in ax_info.spines.values():
        sp.set_edgecolor(GRID)
    ax_info.text(0.5, 0.97, "At the Fermi level", transform=ax_info.transAxes,
                 ha="center", va="top", fontsize=8, color=MUTED,
                 fontstyle="italic")
    info = ax_info.text(0.08, 0.88, "", transform=ax_info.transAxes, va="top",
                        ha="left", fontsize=9.5, color=TEXT, linespacing=1.9)

    cache = DosCache()

    def _compute_and_draw():
        box = QuantumBox(lx_nm=sl_lx.val, ly_nm=sl_ly.val, lz_nm=sl_lz.val,
                         effective_mass=sl_mass.val)
        temperature = sl_temp.val
        try:
            dos = cache.get(box, sigma_eV=sl_sigma.val)
            # Sweep an interior mu window so the grid-edge truncation of the
            # Fermi window does not spike the curves at the boundaries.
            mu_grid = interior_mu_grid(dos, temperature, n=N_SWEEP)
            sweep = thermoelectric_sweep(dos, mu_grid, temperature)
        except ValueError as exc:
            info.set_text(f"Cannot compute:\n\n{exc}")
            fig.canvas.draw_idle()
            return

        ef = dos.fermi_energy_eV
        line_s.set_data(sweep.mu_eV, sweep.seebeck_uV_per_K)
        line_sig.set_data(sweep.mu_eV, sweep.conductivity)
        line_pf.set_data(sweep.mu_eV, sweep.power_factor)
        ax_s.set_xlim(sweep.mu_eV[0], sweep.mu_eV[-1])
        for v in vlines:
            v.set_xdata([ef, ef])

        for a, y in ((ax_s, sweep.seebeck_uV_per_K),
                     (ax_sig, sweep.conductivity),
                     (ax_pf, sweep.power_factor)):
            a.relim(); a.autoscale_view(scalex=False)

        # Values at the Fermi level (nearest sweep point).
        i_ef = int(np.argmin(np.abs(sweep.mu_eV - ef)))
        regime = classify_regime(box)
        regime_badge.set_text(regime)

        info.set_text(
            f"Regime :  {regime}\n"
            f"Ef         :  {ef:.3f} eV\n"
            f"T          :  {temperature:.0f} K\n"
            f"\n"
            f"S    :  {sweep.seebeck_uV_per_K[i_ef]:8.2f} uV/K\n"
            f"       (absolute)\n"
            f"sig  :  {sweep.conductivity[i_ef]:8.3g}\n"
            f"PF   :  {sweep.power_factor[i_ef]:8.3g}\n"
            f"       (arb. units)\n"
            f"\n"
            f"Peak |S| :  {np.max(np.abs(sweep.seebeck_uV_per_K)):.1f} uV/K"
        )
        fig.canvas.draw_idle()

    def _on_change(_=None):
        """Recompute and redraw when any slider changes.

        Computing synchronously in the callback is robust across matplotlib
        backends. The DOS cache makes a temperature-only change cheap (no
        re-enumeration), so updates stay smooth for normal boxes.
        """
        _compute_and_draw()

    for slider in (sl_lx, sl_ly, sl_lz, sl_mass, sl_temp, sl_sigma):
        slider.on_changed(_on_change)

    _compute_and_draw()

    return {
        "fig": fig,
        "axes": {"s": ax_s, "sigma": ax_sig, "pf": ax_pf},
        "sliders": {"lx": sl_lx, "ly": sl_ly, "lz": sl_lz, "mass": sl_mass,
                    "temp": sl_temp, "sigma": sl_sigma},
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
