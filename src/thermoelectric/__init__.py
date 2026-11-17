# -*- coding: utf-8 -*-
"""
Thermoelectric Explorer — Seebeck coefficient, electrical conductivity
and power factor of a low-dimensional electron gas, in the constant
relaxation-time approximation (CRTA), built on top of the Quantum DOS
Explorer density of states.

Public API (no matplotlib dependency)::

    from quantum_dos import QuantumBox, calculate_dos
    from thermoelectric import thermoelectric_point, TransportParams

    box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
    dos = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)
    result = thermoelectric_point(dos, TransportParams(temperature_K=300.0))
    print(result.seebeck_uV_per_K)   # absolute, microvolts/kelvin
"""

from .transport import (
    ThermoelectricResult,
    ThermoelectricSweep,
    TransportParams,
    electrical_conductivity,
    fermi_window,
    mott_seebeck,
    power_factor,
    seebeck_coefficient,
    thermoelectric_point,
    thermoelectric_sweep,
    transport_distribution,
    transport_moments,
)

__all__ = [
    "TransportParams",
    "ThermoelectricResult",
    "ThermoelectricSweep",
    "transport_distribution",
    "fermi_window",
    "transport_moments",
    "mott_seebeck",
    "seebeck_coefficient",
    "electrical_conductivity",
    "power_factor",
    "thermoelectric_point",
    "thermoelectric_sweep",
]

__version__ = "0.1.0"
