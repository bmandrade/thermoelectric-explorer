# -*- coding: utf-8 -*-
"""
Figure of merit ZT: absolute transport, lattice thermal conductivity, and
the dimensionless thermoelectric figure of merit.

The core :mod:`thermoelectric.transport` module reports the conductivity
and power factor in *arbitrary units*, because the constant
relaxation-time approximation leaves the relaxation time ``tau`` unknown
and the density of states is un-normalized. That is enough for the
Seebeck coefficient (which is a ratio and comes out absolute) but not for
the figure of merit

    ZT = S^2 * sigma * T / (kappa_e + kappa_lattice),

which needs the *absolute* electronic-to-thermal balance. This module
supplies the missing pieces:

1. **Absolute electronic transport** (:func:`absolute_transport`): given a
   relaxation time ``tau`` (seconds), an effective mass, the box volume and
   the broadening used to build the DOS, it normalizes the density of
   states to states per joule per cubic metre, includes the free-electron
   group velocity v^2 = 2E/m*, and returns sigma in S/m, the Seebeck
   coefficient in V/K, and the electronic thermal conductivity kappa_e in
   W/m/K -- the standard CRTA transport integrals (cf. BoltzTraP2).

2. **Lattice thermal conductivity** (:func:`lattice_thermal_conductivity`):
   a boundary-scattering model in which phonons are increasingly scattered
   by the sample surfaces as the confined dimension shrinks, so
   kappa_lattice falls with size. This is the main reason low-dimensional
   structures make good thermoelectrics (Hicks & Dresselhaus 1993).

3. **The figure of merit** (:func:`figure_of_merit`): combines the two into
   ZT.

The implementation is validated by the Wiedemann-Franz law: the Lorenz
number kappa_e / (sigma * T) must equal (pi^2/3)(k_B/e)^2 ~ 2.44e-8 in the
degenerate limit, which is an oracle test in the suite.

Caveats (inherited and new):
- Constant relaxation time; a real tau(E) would reshape sigma and kappa_e.
- Free-electron parabolic bands; the group velocity v^2 = 2E/m* and the
  isotropic 1/3 factor are the simplest possible choices.
- The lattice model is a simple boundary-scattering interpolation with
  illustrative material parameters (kappa_bulk, phonon mean free path), not
  a first-principles phonon calculation.
So ZT here is a physically transparent, order-of-magnitude model, not a
quantitative prediction for a specific material.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantum_dos.constants import BOLTZMANN, ELECTRON_MASS, EV_TO_J
from quantum_dos.models import DOSResult, QuantumBox

__all__ = [
    "ThermoelectricSI",
    "ZTResult",
    "absolute_transport",
    "lorenz_number",
    "lattice_thermal_conductivity",
    "figure_of_merit",
]

#: Sommerfeld (degenerate) Lorenz number, (pi^2/3)(k_B/e)^2, in W.Ohm/K^2.
LORENZ_SOMMERFELD: float = (np.pi**2 / 3.0) * (BOLTZMANN / EV_TO_J) ** 2


@dataclass(frozen=True)
class ThermoelectricSI:
    """Absolute electronic transport at one operating point.

    Attributes
    ----------
    seebeck_uV_per_K : float
        Seebeck coefficient, in microvolts/kelvin.
    conductivity_S_per_m : float
        Electrical conductivity, in siemens/metre (ohm^-1 m^-1).
    kappa_e_W_per_mK : float
        Electronic thermal conductivity, in watts/metre/kelvin.
    mu_eV : float
        Chemical potential used, in eV.
    temperature_K : float
        Temperature used, in kelvin.
    """

    seebeck_uV_per_K: float
    conductivity_S_per_m: float
    kappa_e_W_per_mK: float
    mu_eV: float
    temperature_K: float


@dataclass(frozen=True)
class ZTResult:
    """Figure of merit and its ingredients at one operating point.

    Attributes
    ----------
    zt : float
        Dimensionless figure of merit S^2 sigma T / (kappa_e + kappa_lattice).
    seebeck_uV_per_K : float
        Seebeck coefficient, in microvolts/kelvin.
    conductivity_S_per_m : float
        Electrical conductivity, in siemens/metre.
    kappa_e_W_per_mK : float
        Electronic thermal conductivity, in watts/metre/kelvin.
    kappa_lattice_W_per_mK : float
        Lattice thermal conductivity, in watts/metre/kelvin.
    mu_eV : float
        Chemical potential used, in eV.
    temperature_K : float
        Temperature used, in kelvin.
    """

    zt: float
    seebeck_uV_per_K: float
    conductivity_S_per_m: float
    kappa_e_W_per_mK: float
    kappa_lattice_W_per_mK: float
    mu_eV: float
    temperature_K: float


def _validate_positive(name: str, value: float) -> None:
    """Raise ValueError unless ``value`` is finite and strictly positive."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, positive number; got {value!r}.")


def _si_moments(
    energy_eV: np.ndarray,
    dos: np.ndarray,
    volume_m3: float,
    effective_mass: float,
    tau_s: float,
    broadening_eV: float,
    mu_eV: float,
    temperature_K: float,
):
    """Return the SI transport moments (K0, K1, K2) at one operating point.

    Normalizes the (unit-Gaussian-sum) DOS to states/(J*m^3), applies the
    free-electron group velocity v^2 = 2E/m* with the isotropic 1/3 factor
    and the relaxation time, and integrates against the Fermi window in SI
    units (everything in joules). ``sigma = e^2 K0``,
    ``S = -(1/eT) K1/K0``, ``kappa_e = (1/T)(K2 - K1^2/K0)``.
    """
    e = EV_TO_J
    energy_J = np.asarray(energy_eV, dtype=float) * e
    mu_J = mu_eV * e

    # Unit-Gaussian sum -> states per eV is dos/(broadening*sqrt(2pi));
    # -> states per joule divides by e; -> per volume divides by volume.
    g_SI = np.asarray(dos, dtype=float) / (broadening_eV * np.sqrt(2.0 * np.pi))
    g_SI = g_SI / e / volume_m3  # states / (J * m^3)

    v2 = 2.0 * np.clip(energy_J, 0.0, None) / (effective_mass * ELECTRON_MASS)
    sigma_E = (1.0 / 3.0) * v2 * tau_s * g_SI  # transport distribution (SI)

    kT = BOLTZMANN * temperature_K
    half_u = np.clip((energy_J - mu_J) / (2.0 * kT), -350.0, 350.0)
    window = 1.0 / (4.0 * kT * np.cosh(half_u) ** 2)  # 1/J

    base = sigma_E * window
    de = energy_J - mu_J
    k0 = float(np.trapezoid(base, energy_J))
    k1 = float(np.trapezoid(base * de, energy_J))
    k2 = float(np.trapezoid(base * de**2, energy_J))
    return k0, k1, k2


def absolute_transport(
    dos_result: DOSResult,
    *,
    box: QuantumBox,
    temperature_K: float,
    tau_s: float,
    broadening_eV: float,
    mu_eV: Optional[float] = None,
) -> ThermoelectricSI:
    """
    Absolute electronic transport (sigma in S/m, S in uV/K, kappa_e in W/m/K).

    Parameters
    ----------
    dos_result : DOSResult
        Density of states to analyse.
    box : QuantumBox
        The box the DOS was computed for (supplies the volume and effective
        mass needed to put the transport on an absolute scale).
    temperature_K : float
        Temperature, in kelvin. Must be > 0.
    tau_s : float
        Constant relaxation time, in seconds. Must be > 0. This is the
        material input that sets the absolute scale of sigma and kappa_e.
    broadening_eV : float
        The Gaussian broadening (``sigma_eV``) used to build ``dos_result``.
        Needed to normalize the unit-Gaussian DOS back to real state counts.
        Must be > 0.
    mu_eV : float, optional
        Chemical potential, in eV. Defaults to the Fermi energy of
        ``dos_result``. Must lie within the energy grid.

    Returns
    -------
    ThermoelectricSI
        Seebeck coefficient, conductivity and electronic thermal
        conductivity in absolute units.
    """
    _validate_positive("temperature_K", temperature_K)
    _validate_positive("tau_s", tau_s)
    _validate_positive("broadening_eV", broadening_eV)

    energy_eV = np.asarray(dos_result.energy_eV, dtype=float)
    mu = dos_result.fermi_energy_eV if mu_eV is None else mu_eV
    if not np.isfinite(mu) or mu < energy_eV[0] or mu > energy_eV[-1]:
        raise ValueError(
            f"mu_eV = {mu!r} is outside the energy grid "
            f"[{energy_eV[0]:g}, {energy_eV[-1]:g}] eV."
        )

    k0, k1, k2 = _si_moments(
        energy_eV, dos_result.dos, box.volume_m3, box.effective_mass,
        tau_s, broadening_eV, float(mu), temperature_K,
    )
    e = EV_TO_J
    if k0 <= 0.0:
        return ThermoelectricSI(0.0, 0.0, 0.0, float(mu), temperature_K)

    sigma = e**2 * k0
    seebeck = -(1.0 / (e * temperature_K)) * (k1 / k0)
    kappa_e = (1.0 / temperature_K) * (k2 - k1**2 / k0)
    return ThermoelectricSI(
        seebeck_uV_per_K=seebeck * 1.0e6,
        conductivity_S_per_m=sigma,
        kappa_e_W_per_mK=kappa_e,
        mu_eV=float(mu),
        temperature_K=temperature_K,
    )


def lorenz_number(transport: ThermoelectricSI) -> float:
    """
    Lorenz number L = kappa_e / (sigma * T), in W.Ohm/K^2.

    In the degenerate limit this must approach the Sommerfeld value
    :data:`LORENZ_SOMMERFELD` ~ 2.44e-8, which is the Wiedemann-Franz law and
    serves as a validation oracle.
    """
    denom = transport.conductivity_S_per_m * transport.temperature_K
    if denom <= 0.0:
        return 0.0
    return transport.kappa_e_W_per_mK / denom


def lattice_thermal_conductivity(
    size_nm: float,
    *,
    kappa_bulk_W_per_mK: float = 1.5,
    phonon_mfp_nm: float = 5.0,
) -> float:
    """
    Lattice thermal conductivity with phonon boundary scattering.

    Uses a Matthiessen interpolation between bulk and boundary-limited
    transport,

        1 / kappa = 1 / kappa_bulk + 1 / kappa_boundary,
        kappa_boundary = kappa_bulk * (size / mfp),

    which gives

        kappa_lattice(size) = kappa_bulk / (1 + mfp / size).

    As the confined dimension ``size`` shrinks below the bulk phonon mean
    free path, surface scattering suppresses kappa_lattice (-> 0 as
    size -> 0); for size >> mfp it recovers kappa_bulk. This size-driven
    reduction is the main lever low-dimensional structures have on ZT.

    Parameters
    ----------
    size_nm : float
        The relevant (smallest, most strongly confined) dimension, in nm.
        Must be > 0.
    kappa_bulk_W_per_mK : float, optional
        Bulk lattice thermal conductivity, in W/m/K. Material-dependent;
        default 1.5 (a representative good-thermoelectric value).
    phonon_mfp_nm : float, optional
        Bulk phonon mean free path, in nm. Material-dependent; default 5.0.

    Returns
    -------
    float
        Lattice thermal conductivity, in W/m/K.
    """
    _validate_positive("size_nm", size_nm)
    _validate_positive("kappa_bulk_W_per_mK", kappa_bulk_W_per_mK)
    _validate_positive("phonon_mfp_nm", phonon_mfp_nm)
    return kappa_bulk_W_per_mK / (1.0 + phonon_mfp_nm / size_nm)


def figure_of_merit(
    dos_result: DOSResult,
    *,
    box: QuantumBox,
    temperature_K: float,
    tau_s: float,
    broadening_eV: float,
    mu_eV: Optional[float] = None,
    kappa_bulk_W_per_mK: float = 1.5,
    phonon_mfp_nm: float = 5.0,
) -> ZTResult:
    """
    Dimensionless thermoelectric figure of merit ZT.

        ZT = S^2 * sigma * T / (kappa_e + kappa_lattice)

    Combines the absolute electronic transport (:func:`absolute_transport`)
    with the boundary-scattering lattice thermal conductivity
    (:func:`lattice_thermal_conductivity`), evaluated at the box's smallest
    dimension. Unlike the bare power factor, ZT has a well-defined optimum
    in the chemical potential because kappa_lattice bounds the denominator.

    Parameters
    ----------
    dos_result : DOSResult
        Density of states to analyse.
    box : QuantumBox
        The box the DOS was computed for.
    temperature_K : float
        Temperature, in kelvin. Must be > 0.
    tau_s : float
        Constant relaxation time, in seconds. Must be > 0.
    broadening_eV : float
        The Gaussian broadening used to build ``dos_result``. Must be > 0.
    mu_eV : float, optional
        Chemical potential, in eV. Defaults to the Fermi energy.
    kappa_bulk_W_per_mK, phonon_mfp_nm : float, optional
        Lattice-model parameters (see
        :func:`lattice_thermal_conductivity`).

    Returns
    -------
    ZTResult
        ZT and all of its ingredients.
    """
    transport = absolute_transport(
        dos_result, box=box, temperature_K=temperature_K, tau_s=tau_s,
        broadening_eV=broadening_eV, mu_eV=mu_eV,
    )
    size_nm = min(box.dimensions_nm)
    kappa_lat = lattice_thermal_conductivity(
        size_nm, kappa_bulk_W_per_mK=kappa_bulk_W_per_mK,
        phonon_mfp_nm=phonon_mfp_nm,
    )

    s_V_per_K = transport.seebeck_uV_per_K * 1.0e-6
    power_factor = s_V_per_K**2 * transport.conductivity_S_per_m  # W/m/K^2
    denom = transport.kappa_e_W_per_mK + kappa_lat
    zt = power_factor * temperature_K / denom if denom > 0 else 0.0

    return ZTResult(
        zt=zt,
        seebeck_uV_per_K=transport.seebeck_uV_per_K,
        conductivity_S_per_m=transport.conductivity_S_per_m,
        kappa_e_W_per_mK=transport.kappa_e_W_per_mK,
        kappa_lattice_W_per_mK=kappa_lat,
        mu_eV=transport.mu_eV,
        temperature_K=temperature_K,
    )
