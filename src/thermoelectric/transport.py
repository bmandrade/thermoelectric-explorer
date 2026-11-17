# -*- coding: utf-8 -*-
"""
Thermoelectric transport parameters for a low-dimensional electron gas,
built on top of the Quantum DOS Explorer density of states.

This module computes the three electronic thermoelectric parameters --
the Seebeck coefficient ``S``, the electrical conductivity ``sigma``,
and the power factor ``PF = S^2 * sigma`` -- from a density of states
``g(E)`` using the constant relaxation-time approximation (CRTA). It has
no dependency on matplotlib, so it can be imported and tested
independently of the GUI, exactly like :mod:`quantum_dos.physics`::

    from quantum_dos import QuantumBox, calculate_dos
    from thermoelectric import thermoelectric_point, TransportParams

    box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
    dos = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
    result = thermoelectric_point(dos, TransportParams(temperature_K=300.0))
    print(result.seebeck_uV_per_K)       # absolute, in microvolts/kelvin

Physical model
--------------
All three parameters are moments of the **transport distribution
function** ``Sigma(E)`` against the Fermi window ``(-df/dE)`` (Mahan &
Sofo, PNAS 93 (1996) 7436; Madsen, Carrete & Verstraete, BoltzTraP2,
Comput. Phys. Commun. 231 (2018) 140). In the CRTA free-electron picture

    Sigma(E) = g(E) * v(E)^2 * tau   with   v(E)^2 proportional to E,

so ``Sigma(E) = g(E) * E * tau``. The factor of ``E`` is the
group-velocity weighting that distinguishes *transport* from the bare
DOS; omitting it is a common and serious error, so it lives in exactly
one place here (:func:`transport_distribution`).

Defining the moment integrals, for alpha = 0, 1, 2,

    L[alpha](mu, T) = integral of Sigma(E) * (E - mu)^alpha * (-df/dE) dE

(a universal ``q^2`` prefactor is dropped because it cancels in every
ratio below and does not affect the relative ``sigma``), the parameters
are

    sigma = L[0]
    S     = -(1 / T) * (L[1] / L[0])          [volts/kelvin]
    PF    = S^2 * sigma

**Absolute vs relative.** Because ``tau`` enters ``L[0]`` and ``L[1]``
identically, it cancels exactly in the ratio ``L[1]/L[0]``, so **S is a
genuine, absolute prediction in microvolts/kelvin**. It does *not*
cancel in ``sigma`` or ``PF``, which scale linearly with the unknown
``tau`` and -- because the DOS from this package is itself
un-normalized (arbitrary units; see :class:`~quantum_dos.models.DOSResult`)
-- are reported in **arbitrary units**. Supplying a ``tau`` fixes the
``tau``-scaling but not the DOS normalization, so ``sigma`` and ``PF``
remain relative curves useful for comparison, not SI values.

A convenient internal cross-check is the Mott/Sommerfeld low-temperature
limit (:func:`mott_seebeck`),

    S ~ -(pi^2 * kB^2 * T / 3q) * d ln Sigma(E)/dE at E = mu,

which the full integral must reproduce as T -> 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantum_dos.constants import BOLTZMANN, EV_TO_J
from quantum_dos.models import DOSResult

__all__ = [
    "TransportParams",
    "ThermoelectricResult",
    "ThermoelectricSweep",
    "transport_distribution",
    "transport_moments",
    "fermi_window",
    "seebeck_coefficient",
    "electrical_conductivity",
    "power_factor",
    "thermoelectric_point",
    "thermoelectric_sweep",
    "mott_seebeck",
]

#: Boltzmann constant in eV/K. Numerically this also equals kB/e in
#: volts/kelvin (because 1 eV per elementary charge == 1 volt), which is
#: what lets an energy difference in eV, divided by the charge, come out
#: directly in volts.
_KB_EV_PER_K: float = BOLTZMANN / EV_TO_J

#: Trapezoidal integrator, portable across NumPy versions: NumPy 2.0
#: renamed ``numpy.trapz`` to ``numpy.trapezoid``. Bind whichever this
#: install provides so the package works on both old and new NumPy.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------
# Parameter and result objects (unit-tagged field names, as in models.py)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TransportParams:
    """
    Operating point for a thermoelectric calculation.

    Parameters
    ----------
    temperature_K : float
        Temperature, in kelvin. Must be > 0 (unlike the Fermi-Dirac
        occupation, transport needs a finite Fermi window, so T = 0 is
        rejected rather than treated as a limit).
    mu_eV : float, optional
        Chemical potential (electrochemical potential), in eV. If
        ``None`` (the default), the Fermi energy stored on the
        :class:`~quantum_dos.models.DOSResult` being analysed is used.
        Must lie within the energy grid.
    tau_s : float, optional
        Constant relaxation time, in seconds. If ``None`` (the default),
        a value of 1.0 is used internally and ``sigma``/``PF`` are
        reported in arbitrary units. If given, it must be > 0 and fixes
        the ``tau``-scaling of ``sigma``/``PF`` (the DOS-normalization
        arbitrariness remains; see module docstring).
    """

    temperature_K: float
    mu_eV: Optional[float] = None
    tau_s: Optional[float] = None


@dataclass(frozen=True)
class ThermoelectricResult:
    """
    Thermoelectric parameters at a single operating point.

    Attributes
    ----------
    seebeck_uV_per_K : float
        Seebeck coefficient, in microvolts/kelvin. **Absolute** (the
        relaxation time cancels), so this is a physical prediction.
    conductivity : float
        Electrical conductivity in arbitrary units (scales with
        ``tau``). See ``conductivity_is_absolute``.
    power_factor : float
        Power factor ``S^2 * sigma``. Same arbitrary-units status as
        ``conductivity``.
    conductivity_is_absolute : bool
        ``True`` only if a relaxation time was supplied. Even then the
        DOS normalization is arbitrary, so treat ``conductivity`` and
        ``power_factor`` as relative unless you know the DOS is
        physically normalized.
    mu_eV : float
        Chemical potential used, in eV.
    temperature_K : float
        Temperature used, in kelvin.
    """

    seebeck_uV_per_K: float
    conductivity: float
    power_factor: float
    conductivity_is_absolute: bool
    mu_eV: float
    temperature_K: float


@dataclass(frozen=True)
class ThermoelectricSweep:
    """
    Thermoelectric parameters swept over chemical potential at fixed T.

    Attributes
    ----------
    mu_eV : np.ndarray
        Chemical-potential grid, in eV.
    seebeck_uV_per_K : np.ndarray
        Seebeck coefficient at each ``mu``, in microvolts/kelvin
        (absolute).
    conductivity : np.ndarray
        Electrical conductivity at each ``mu`` (arbitrary units).
    power_factor : np.ndarray
        Power factor at each ``mu`` (arbitrary units).
    temperature_K : float
        Temperature used for the whole sweep, in kelvin.
    """

    mu_eV: np.ndarray
    seebeck_uV_per_K: np.ndarray
    conductivity: np.ndarray
    power_factor: np.ndarray
    temperature_K: float


# --------------------------------------------------------------------------
# Input validation (exceptions for bad input, as in physics.py)
# --------------------------------------------------------------------------

def _validate_arrays(energy_eV: np.ndarray, dos: np.ndarray) -> None:
    """Reject empty or mismatched energy/DOS arrays with a ValueError."""
    energy_eV = np.asarray(energy_eV, dtype=float)
    dos = np.asarray(dos, dtype=float)
    if energy_eV.ndim != 1 or dos.ndim != 1:
        raise ValueError("energy_eV and dos must both be 1-D arrays.")
    if energy_eV.size < 2:
        raise ValueError(
            f"energy grid must have at least 2 points; got {energy_eV.size}."
        )
    if energy_eV.size != dos.size:
        raise ValueError(
            "energy_eV and dos must have the same length; got "
            f"{energy_eV.size} and {dos.size}."
        )


def _validate_temperature(temperature_K: float) -> None:
    """Reject a non-finite or non-positive temperature with a ValueError."""
    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError(
            "temperature_K must be a finite, positive number for transport "
            f"(T = 0 has no Fermi window); got {temperature_K!r}."
        )


def _validate_tau(tau_s: float) -> None:
    """Reject a non-finite or non-positive relaxation time with a ValueError."""
    if not np.isfinite(tau_s) or tau_s <= 0:
        raise ValueError(
            f"tau_s must be a finite, positive number (seconds); got {tau_s!r}."
        )


def _validate_mu_in_grid(mu_eV: float, energy_eV: np.ndarray) -> None:
    """Reject a chemical potential outside the energy grid with a ValueError.

    A ``mu`` outside the grid means the Fermi window is not represented
    by the available DOS, so every moment integral would be silently
    truncated and wrong. This is rejected rather than returning a
    plausible-looking but meaningless number.
    """
    lo, hi = float(energy_eV[0]), float(energy_eV[-1])
    if not np.isfinite(mu_eV) or mu_eV < lo or mu_eV > hi:
        raise ValueError(
            f"mu_eV = {mu_eV!r} lies outside the energy grid [{lo:g}, {hi:g}] "
            "eV, so the Fermi window is not captured by the DOS. Widen the "
            "grid (increase e_max_eV) or choose a mu inside it."
        )


# --------------------------------------------------------------------------
# Core physics (operate on plain arrays, no DOSResult required)
# --------------------------------------------------------------------------

def transport_distribution(
    energy_eV: np.ndarray, dos: np.ndarray, tau_s: float = 1.0
) -> np.ndarray:
    """
    Transport distribution function Sigma(E) = g(E) * E * tau.

    This is the single place the free-electron group-velocity weighting
    ``v(E)^2 proportional to E`` enters. Every transport parameter is a
    moment of the array returned here.

    Parameters
    ----------
    energy_eV : np.ndarray
        Energy grid, in eV, measured from the band bottom (where the
        kinetic energy, and hence ``v^2``, is zero).
    dos : np.ndarray
        Density of states ``g(E)`` on that grid (arbitrary units, as
        produced by :func:`quantum_dos.physics.calculate_dos`).
    tau_s : float, optional
        Constant relaxation time, in seconds. Default 1.0. Must be > 0.
        It enters as a pure multiplicative factor, so it cancels in the
        Seebeck coefficient and scales ``sigma``/``PF`` linearly.

    Returns
    -------
    np.ndarray
        ``Sigma(E)`` on ``energy_eV``, same shape. Zero wherever
        ``energy_eV`` is zero (no velocity, no transport).
    """
    _validate_arrays(energy_eV, dos)
    _validate_tau(tau_s)
    energy_eV = np.asarray(energy_eV, dtype=float)
    dos = np.asarray(dos, dtype=float)
    return dos * np.clip(energy_eV, 0.0, None) * tau_s


def fermi_window(
    energy_eV: np.ndarray, mu_eV: float, temperature_K: float
) -> np.ndarray:
    """
    Fermi window ``(-df/dE)``, the thermal broadening that selects which
    states carry transport.

    Computed analytically (not by finite-differencing ``f``) for
    accuracy at low temperature, using the numerically stable identity

        -df/dE = 1 / (4 * kB*T * cosh^2((E - mu) / (2 kB*T))).

    Parameters
    ----------
    energy_eV : np.ndarray
        Energy grid, in eV.
    mu_eV : float
        Chemical potential, in eV.
    temperature_K : float
        Temperature, in kelvin. Must be > 0.

    Returns
    -------
    np.ndarray
        ``(-df/dE)`` on ``energy_eV``, in 1/eV. Integrates to 1 over a
        grid wide enough to contain the window.
    """
    _validate_temperature(temperature_K)
    energy_eV = np.asarray(energy_eV, dtype=float)
    kT_eV = _KB_EV_PER_K * temperature_K
    half_u = np.clip((energy_eV - mu_eV) / (2.0 * kT_eV), -350.0, 350.0)
    return 1.0 / (4.0 * kT_eV * np.cosh(half_u) ** 2)


def transport_moments(
    energy_eV: np.ndarray,
    sigma_E: np.ndarray,
    mu_eV: float,
    temperature_K: float,
) -> tuple[float, float, float]:
    """
    The three transport moments ``L[0]``, ``L[1]``, ``L[2]`` of a
    transport distribution function.

        L[alpha] = integral Sigma(E) * (E - mu)^alpha * (-df/dE) dE

    (the universal ``q^2`` prefactor is omitted; it cancels in every
    ratio used downstream). The integral is evaluated by the trapezoidal
    rule on the supplied grid.

    Parameters
    ----------
    energy_eV : np.ndarray
        Energy grid, in eV.
    sigma_E : np.ndarray
        Transport distribution ``Sigma(E)`` on that grid (e.g. from
        :func:`transport_distribution`).
    mu_eV : float
        Chemical potential, in eV. Must lie within the grid.
    temperature_K : float
        Temperature, in kelvin. Must be > 0.

    Returns
    -------
    (float, float, float)
        ``(L0, L1, L2)``. ``L0`` has the units of ``Sigma``; ``L1`` and
        ``L2`` carry an extra ``eV`` and ``eV^2`` from the ``(E - mu)``
        factors.
    """
    _validate_arrays(energy_eV, sigma_E)
    _validate_temperature(temperature_K)
    _validate_mu_in_grid(mu_eV, np.asarray(energy_eV, dtype=float))

    energy_eV = np.asarray(energy_eV, dtype=float)
    sigma_E = np.asarray(sigma_E, dtype=float)
    window = fermi_window(energy_eV, mu_eV, temperature_K)
    base = sigma_E * window
    de = energy_eV - mu_eV

    l0 = float(_trapezoid(base, energy_eV))
    l1 = float(_trapezoid(base * de, energy_eV))
    l2 = float(_trapezoid(base * de**2, energy_eV))
    return l0, l1, l2


def mott_seebeck(
    energy_eV: np.ndarray,
    sigma_E: np.ndarray,
    mu_eV: float,
    temperature_K: float,
) -> float:
    """
    Low-temperature Mott/Sommerfeld estimate of the Seebeck coefficient.

        S ~ -(pi^2 / 3) * kB * (kB*T) * d ln Sigma(E)/dE at E = mu,

    returned in microvolts/kelvin. This is the closed-form limit the
    full integral (:func:`seebeck_coefficient`) must approach as
    ``T -> 0``; it is used both as an oracle in the test-suite and as a
    fast approximation.

    The logarithmic derivative is evaluated numerically at ``mu`` from
    the supplied ``Sigma(E)``.

    Parameters
    ----------
    energy_eV : np.ndarray
        Energy grid, in eV.
    sigma_E : np.ndarray
        Transport distribution ``Sigma(E)`` on that grid.
    mu_eV : float
        Chemical potential, in eV. Must lie within the grid.
    temperature_K : float
        Temperature, in kelvin. Must be > 0.

    Returns
    -------
    float
        Mott estimate of ``S``, in microvolts/kelvin.
    """
    _validate_arrays(energy_eV, sigma_E)
    _validate_temperature(temperature_K)
    _validate_mu_in_grid(mu_eV, np.asarray(energy_eV, dtype=float))

    energy_eV = np.asarray(energy_eV, dtype=float)
    sigma_E = np.asarray(sigma_E, dtype=float)

    # d ln Sigma / dE = (dSigma/dE) / Sigma, evaluated only where Sigma > 0.
    positive = sigma_E > 0
    grad = np.gradient(sigma_E, energy_eV)
    dln = np.where(positive, grad / np.where(positive, sigma_E, 1.0), np.nan)
    good = np.isfinite(dln)
    dln_at_mu = float(np.interp(mu_eV, energy_eV[good], dln[good]))

    kT_eV = _KB_EV_PER_K * temperature_K
    s_V_per_K = -(np.pi**2 / 3.0) * _KB_EV_PER_K * kT_eV * dln_at_mu
    return s_V_per_K * 1.0e6


# --------------------------------------------------------------------------
# Public API (take a DOSResult + TransportParams)
# --------------------------------------------------------------------------

def _resolve(dos_result: DOSResult, params: TransportParams):
    """Pull (energy, Sigma, mu, T, tau, tau_given) from a result + params.

    Centralizes the defaulting rules (mu defaults to the Fermi energy,
    tau defaults to 1.0 and is flagged as not-supplied) and all input
    validation, so every public function below validates identically.
    """
    temperature_K = params.temperature_K
    _validate_temperature(temperature_K)

    energy_eV = np.asarray(dos_result.energy_eV, dtype=float)
    dos = np.asarray(dos_result.dos, dtype=float)
    _validate_arrays(energy_eV, dos)

    mu_eV = dos_result.fermi_energy_eV if params.mu_eV is None else params.mu_eV
    _validate_mu_in_grid(mu_eV, energy_eV)

    tau_given = params.tau_s is not None
    tau_s = 1.0 if params.tau_s is None else params.tau_s
    _validate_tau(tau_s)

    sigma_E = transport_distribution(energy_eV, dos, tau_s=tau_s)
    return energy_eV, sigma_E, float(mu_eV), temperature_K, tau_given


def seebeck_coefficient(dos_result: DOSResult, params: TransportParams) -> float:
    """
    Seebeck coefficient ``S``, in microvolts/kelvin (**absolute**).

        S = -(1 / T) * (L[1] / L[0])

    The relaxation time cancels in the ratio ``L[1]/L[0]``, so the value
    returned is a genuine physical prediction independent of ``tau``. A
    DOS rising with energy at ``mu`` (electron-like) gives a negative
    ``S``; a DOS falling with energy gives a positive ``S``.

    Parameters
    ----------
    dos_result : DOSResult
        Density of states to analyse (its ``fermi_energy_eV`` is the
        default chemical potential).
    params : TransportParams
        Operating point (temperature, optional ``mu``, optional ``tau``).

    Returns
    -------
    float
        ``S`` in microvolts/kelvin.
    """
    energy_eV, sigma_E, mu_eV, temperature_K, _ = _resolve(dos_result, params)
    l0, l1, _ = transport_moments(energy_eV, sigma_E, mu_eV, temperature_K)
    if l0 == 0.0:
        return 0.0
    s_V_per_K = -(1.0 / temperature_K) * (l1 / l0)
    return s_V_per_K * 1.0e6


def electrical_conductivity(dos_result: DOSResult, params: TransportParams) -> float:
    """
    Electrical conductivity ``sigma = L[0]`` (arbitrary units).

    Scales linearly with the relaxation time ``tau`` and with the
    (un-normalized) DOS, so it is a **relative** quantity meaningful for
    comparison across operating points or systems, not an SI value. See
    :func:`thermoelectric_point` for the ``conductivity_is_absolute``
    flag.

    Parameters
    ----------
    dos_result : DOSResult
        Density of states to analyse.
    params : TransportParams
        Operating point.

    Returns
    -------
    float
        ``sigma`` in arbitrary units (proportional to ``tau``).
    """
    energy_eV, sigma_E, mu_eV, temperature_K, _ = _resolve(dos_result, params)
    l0, _, _ = transport_moments(energy_eV, sigma_E, mu_eV, temperature_K)
    return l0


def power_factor(dos_result: DOSResult, params: TransportParams) -> float:
    """
    Thermoelectric power factor ``PF = S^2 * sigma`` (arbitrary units).

    Combines the absolute Seebeck coefficient with the relative
    conductivity, so ``PF`` inherits the arbitrary-units status of
    ``sigma`` (it scales with ``tau``). It is computed as ``S^2 * sigma``
    from the very same values the individual functions return, so it is
    exactly consistent with them.

    Parameters
    ----------
    dos_result : DOSResult
        Density of states to analyse.
    params : TransportParams
        Operating point.

    Returns
    -------
    float
        ``PF`` in arbitrary units.
    """
    s = seebeck_coefficient(dos_result, params)
    sigma = electrical_conductivity(dos_result, params)
    return s**2 * sigma


def thermoelectric_point(
    dos_result: DOSResult, params: TransportParams
) -> ThermoelectricResult:
    """
    Compute all three thermoelectric parameters at one operating point.

    Parameters
    ----------
    dos_result : DOSResult
        Density of states to analyse.
    params : TransportParams
        Operating point (temperature, optional ``mu``, optional ``tau``).

    Returns
    -------
    ThermoelectricResult
        Seebeck coefficient (absolute, microvolts/kelvin), conductivity
        and power factor (arbitrary units), the
        ``conductivity_is_absolute`` flag, and the ``mu``/``T`` used.
    """
    energy_eV, sigma_E, mu_eV, temperature_K, tau_given = _resolve(dos_result, params)
    l0, l1, _ = transport_moments(energy_eV, sigma_E, mu_eV, temperature_K)

    seebeck = 0.0 if l0 == 0.0 else -(1.0 / temperature_K) * (l1 / l0) * 1.0e6
    conductivity = l0
    return ThermoelectricResult(
        seebeck_uV_per_K=seebeck,
        conductivity=conductivity,
        power_factor=seebeck**2 * conductivity,
        conductivity_is_absolute=tau_given,
        mu_eV=mu_eV,
        temperature_K=temperature_K,
    )


def thermoelectric_sweep(
    dos_result: DOSResult,
    mu_grid_eV: np.ndarray,
    temperature_K: float,
    tau_s: Optional[float] = None,
) -> ThermoelectricSweep:
    """
    Sweep the thermoelectric parameters over chemical potential at fixed
    temperature.

    This produces the curves ``S(mu)``, ``sigma(mu)``, ``PF(mu)`` that
    reveal the confinement signature: a sharper DOS (stronger
    confinement) yields a larger peak power factor (Hicks & Dresselhaus,
    Phys. Rev. B 47 (1993) 12727).

    Parameters
    ----------
    dos_result : DOSResult
        Density of states to analyse.
    mu_grid_eV : np.ndarray
        Chemical-potential values to sweep, in eV. Every value must lie
        within the DOS energy grid.
    temperature_K : float
        Temperature, in kelvin. Must be > 0.
    tau_s : float, optional
        Constant relaxation time, in seconds. If ``None``, ``sigma``/``PF``
        are in arbitrary units.

    Returns
    -------
    ThermoelectricSweep
        Arrays of ``S`` (microvolts/kelvin), ``sigma`` and ``PF``
        (arbitrary units) at each ``mu``.
    """
    _validate_temperature(temperature_K)
    mu_grid_eV = np.asarray(mu_grid_eV, dtype=float)
    if mu_grid_eV.ndim != 1 or mu_grid_eV.size == 0:
        raise ValueError("mu_grid_eV must be a non-empty 1-D array.")

    seebeck = np.empty_like(mu_grid_eV)
    conductivity = np.empty_like(mu_grid_eV)
    powerfactor = np.empty_like(mu_grid_eV)

    for i, mu in enumerate(mu_grid_eV):
        point = thermoelectric_point(
            dos_result,
            TransportParams(temperature_K=temperature_K, mu_eV=float(mu), tau_s=tau_s),
        )
        seebeck[i] = point.seebeck_uV_per_K
        conductivity[i] = point.conductivity
        powerfactor[i] = point.power_factor

    return ThermoelectricSweep(
        mu_eV=mu_grid_eV,
        seebeck_uV_per_K=seebeck,
        conductivity=conductivity,
        power_factor=powerfactor,
        temperature_K=temperature_K,
    )
