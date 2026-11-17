# -*- coding: utf-8 -*-
"""Tests for quantum_dos.thermoelectric (CRTA transport core, no matplotlib).

The thermoelectric model has unusually strong oracles: several exact
facts pin the output independent of implementation details (the
Mott/Sommerfeld limit, the exact cancellation of the relaxation time in
the Seebeck coefficient, the carrier-type sign rule). Those are the
backbone of this suite. Each test states WHY it exists, then
GIVEN/WHEN/THEN, and is written to fail for a single reason.
"""

import numpy as np
import pytest

from quantum_dos.models import DOSResult
from thermoelectric.transport import (
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


def _dos_result(energy_eV: np.ndarray, dos: np.ndarray, mu_eV: float) -> DOSResult:
    """Wrap a bare (energy, DOS) pair as a DOSResult for the public API.

    Lets a test build an arbitrary analytic DOS directly, without going
    through the (slower, more constrained) full calculate_dos pipeline,
    while still exercising the real public functions.
    """
    return DOSResult(
        energy_eV=energy_eV,
        dos=dos,
        fermi_energy_eV=mu_eV,
        n_states_enumerated=int(energy_eV.size),
    )


def _free_electron_dos(e_max_eV: float = 12.0, n: int = 4000):
    """A smooth 3-D free-electron DOS g(E) ~ sqrt(E) on a fine grid.

    Used wherever a test needs a known, differentiable DOS with no
    discreteness artifacts, so that the comparison is against the
    continuum physics rather than broadening noise.
    """
    energy_eV = np.linspace(0.01, e_max_eV, n)
    dos = np.sqrt(energy_eV)
    return energy_eV, dos


# --------------------------------------------------------------------------
# transport_distribution
# --------------------------------------------------------------------------

class TestTransportDistribution:
    def test_includes_the_group_velocity_factor_of_energy(self):
        """Sigma(E) must be g(E)*E*tau, not g(E)*tau.

        WHY: the velocity weighting v^2 proportional to E is the one thing
        that distinguishes transport from the bare DOS, and dropping it is
        the classic thermoelectric bug. This test pins that the energy
        factor is actually present by checking Sigma against an
        independently formed g*E*tau.

        GIVEN a DOS and a relaxation time,
        WHEN the transport distribution is formed,
        THEN it equals g(E)*E*tau pointwise.
        """
        energy_eV, dos = _free_electron_dos()
        tau = 3.0
        sigma = transport_distribution(energy_eV, dos, tau_s=tau)
        assert np.allclose(sigma, dos * energy_eV * tau)

    def test_scales_linearly_with_relaxation_time(self):
        """Doubling tau must exactly double Sigma(E).

        WHY: tau enters Sigma as a pure multiplicative constant; this is
        the property that later makes sigma proportional to tau and makes
        tau cancel in S.

        GIVEN one DOS,
        WHEN Sigma is formed at tau and at 2*tau,
        THEN the second is exactly twice the first.
        """
        energy_eV, dos = _free_electron_dos()
        single = transport_distribution(energy_eV, dos, tau_s=1.0)
        double = transport_distribution(energy_eV, dos, tau_s=2.0)
        assert np.allclose(double, 2.0 * single)


# --------------------------------------------------------------------------
# fermi_window
# --------------------------------------------------------------------------

class TestFermiWindow:
    def test_integrates_to_one_over_a_wide_grid(self):
        """The Fermi window (-df/dE) must integrate to 1.

        WHY: (-df/dE) is the weight that selects transporting states; its
        area is exactly 1 (it is the derivative of a function going 1->0).
        A broken analytic derivative would silently bias every moment, so
        this normalization is a load-bearing invariant.

        GIVEN a wide, fine energy grid and a temperature,
        WHEN the Fermi window is integrated over the grid,
        THEN the result is 1 to a tight tolerance.
        """
        energy_eV = np.linspace(0.01, 12.0, 20000)
        window = fermi_window(energy_eV, mu_eV=5.0, temperature_K=300.0)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        area = trapezoid(window, energy_eV)
        assert area == pytest.approx(1.0, rel=1e-4)

    def test_is_peaked_at_the_chemical_potential(self):
        """The window must peak at E = mu.

        WHY: transport is dominated by states within ~kB*T of mu; the
        window's maximum marks that center. If it peaked elsewhere, every
        moment would be sampled from the wrong energies.

        GIVEN a grid and a chemical potential on it,
        WHEN the window is evaluated,
        THEN its maximum falls at the grid point closest to mu.
        """
        energy_eV = np.linspace(0.01, 12.0, 4000)
        mu = 4.0
        window = fermi_window(energy_eV, mu_eV=mu, temperature_K=300.0)
        peak_energy = energy_eV[np.argmax(window)]
        assert abs(peak_energy - mu) < (energy_eV[1] - energy_eV[0]) * 2


# --------------------------------------------------------------------------
# Seebeck coefficient: the core oracles
# --------------------------------------------------------------------------

class TestSeebeckOracles:
    def test_matches_mott_formula_at_low_temperature(self):
        """The full Seebeck integral must reduce to the Mott formula as T->0.

        WHY: this is the headline oracle. The Sommerfeld expansion gives a
        closed form, S = -(pi^2 kB^2 T / 3q) dlnSigma/dE at mu, which the
        full moment integral must reproduce at low temperature. Agreement
        validates the entire chain -- window, moments, prefactors, sign --
        against an independent analytic result.

        GIVEN a smooth free-electron DOS and a low temperature (50 K),
        WHEN the full seebeck_coefficient and the mott_seebeck estimate
        are both evaluated at the same mu,
        THEN they agree to a few percent (the residual is the higher-order
        Sommerfeld correction, which shrinks with T).
        """
        energy_eV, dos = _free_electron_dos()
        mu, T = 5.0, 50.0
        sigma = transport_distribution(energy_eV, dos, tau_s=1.0)

        full = seebeck_coefficient(_dos_result(energy_eV, dos, mu),
                                   TransportParams(temperature_K=T, mu_eV=mu))
        mott = mott_seebeck(energy_eV, sigma, mu_eV=mu, temperature_K=T)
        assert full == pytest.approx(mott, rel=0.05)

    def test_is_independent_of_the_relaxation_time(self):
        """S must not depend on tau.

        WHY: tau appears identically in L0 and L1, so it cancels exactly in
        their ratio. tau-independence is the single most important physical
        property of the model and the reason S is quantitatively absolute;
        any leak of tau into S is a correctness bug.

        GIVEN one DOS and one operating point,
        WHEN S is computed with tau = 1 and with tau = 1000,
        THEN the two values are equal to floating-point tolerance.
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=5.0)
        s_small = seebeck_coefficient(result, TransportParams(temperature_K=300.0, tau_s=1.0))
        s_large = seebeck_coefficient(result, TransportParams(temperature_K=300.0, tau_s=1000.0))
        assert s_small == pytest.approx(s_large, rel=1e-9)

    def test_is_negative_for_an_electron_like_dos(self):
        """A DOS rising with energy at mu must give a negative Seebeck.

        WHY: the sign of S encodes carrier type. For a DOS whose transport
        distribution increases with energy at mu (electron-like, n-type),
        the convention here requires S < 0. This test pins the sign rule,
        which is the easiest thing to get backwards.

        GIVEN a free-electron DOS (Sigma increasing with E at mu),
        WHEN S is computed,
        THEN it is negative.
        """
        energy_eV, dos = _free_electron_dos()
        s = seebeck_coefficient(_dos_result(energy_eV, dos, 5.0),
                                TransportParams(temperature_K=300.0, mu_eV=5.0))
        assert s < 0.0

    def test_sign_flips_when_the_dos_slope_flips(self):
        """A transport distribution falling with energy at mu gives S > 0.

        WHY: completing the sign rule -- a decreasing transport
        distribution (hole-like, p-type) must flip S to positive. Testing
        both signs guards against an implementation that happens to get one
        case right by accident. Note the relevant slope is that of
        Sigma = g*E, not of g alone: the sign is set by the transport
        distribution, so the DOS here is a peak sitting below mu, placing mu
        on the falling flank of Sigma.

        GIVEN a DOS peaked well below mu, so Sigma decreases with energy at
        mu,
        WHEN S is computed,
        THEN it is positive, the opposite sign to the electron-like case.
        """
        energy_eV = np.linspace(0.01, 12.0, 4000)
        dos = np.exp(-((energy_eV - 3.0) ** 2) / (2.0 * 0.5**2))  # peak at 3 eV
        s = seebeck_coefficient(_dos_result(energy_eV, dos, 4.5),
                                TransportParams(temperature_K=300.0, mu_eV=4.5))
        assert s > 0.0

    def test_grows_with_temperature_in_the_degenerate_limit(self):
        """For a degenerate gas |S| must be small and grow with T.

        WHY: deep in a filled band the Mott formula gives S proportional to
        T, small in magnitude (tens of microvolts/kelvin or less). A gross
        unit error would show up either as the wrong magnitude scale or as
        the wrong temperature trend, so both are checked.

        GIVEN a free-electron DOS with mu well above the band bottom,
        WHEN |S| is evaluated at 100 K and at 300 K,
        THEN |S| is modest (< 100 microvolts/kelvin) and larger at 300 K.
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=6.0)
        s100 = abs(seebeck_coefficient(result, TransportParams(temperature_K=100.0, mu_eV=6.0)))
        s300 = abs(seebeck_coefficient(result, TransportParams(temperature_K=300.0, mu_eV=6.0)))
        assert s100 < 100.0
        assert s300 > s100


# --------------------------------------------------------------------------
# Conductivity and power factor
# --------------------------------------------------------------------------

class TestConductivityAndPowerFactor:
    def test_conductivity_scales_linearly_with_relaxation_time(self):
        """sigma must double when tau doubles.

        WHY: sigma = L0 is proportional to tau, which is exactly why it is
        a relative quantity. Pinning the linear tau-scaling documents that
        relativity and is the counterpart to the tau-independence of S.

        GIVEN one operating point,
        WHEN sigma is computed at tau and 2*tau,
        THEN sigma(2*tau) is exactly twice sigma(tau).
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=5.0)
        s1 = electrical_conductivity(result, TransportParams(temperature_K=300.0, tau_s=1.0))
        s2 = electrical_conductivity(result, TransportParams(temperature_K=300.0, tau_s=2.0))
        assert s2 == pytest.approx(2.0 * s1, rel=1e-12)

    def test_power_factor_equals_seebeck_squared_times_conductivity(self):
        """PF must be exactly S^2 * sigma.

        WHY: the power factor is a derived quantity; it must be consistent
        with the individual S and sigma the other functions return, not
        computed by some divergent path. This guards against the two ever
        drifting apart.

        GIVEN one operating point,
        WHEN power_factor is compared with S^2 * sigma from the individual
        functions,
        THEN they are equal.
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=5.0)
        params = TransportParams(temperature_K=300.0, mu_eV=5.0)
        pf = power_factor(result, params)
        s = seebeck_coefficient(result, params)
        sigma = electrical_conductivity(result, params)
        assert pf == pytest.approx(s**2 * sigma, rel=1e-12)


# --------------------------------------------------------------------------
# The Hicks-Dresselhaus confinement signature
# --------------------------------------------------------------------------

class TestConfinementSignature:
    def test_sharper_dos_yields_a_larger_peak_power_factor(self):
        """A sharper DOS onset must give a larger peak power factor.

        WHY: this is the scientific claim of the whole project -- quantum
        confinement sharpens the DOS and thereby raises the thermoelectric
        power factor (Hicks & Dresselhaus 1993). The suite should assert
        that physical direction, not merely that the arithmetic runs. A
        steeper DOS onset is the direct, controlled stand-in for stronger
        confinement.

        GIVEN two DOS curves with a step-like onset at the same energy, one
        with a gentle (wide) onset and one with a sharp (narrow) onset, each
        swept over chemical potential at fixed T and tau,
        WHEN the maximum power factor over the sweep is taken for each,
        THEN the sharper onset yields the larger peak power factor.
        """
        energy_eV = np.linspace(0.01, 12.0, 4000)
        onset = 4.0

        def logistic_onset(width_eV: float) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-(energy_eV - onset) / width_eV))

        gentle = _dos_result(energy_eV, logistic_onset(0.60), mu_eV=onset)
        sharp = _dos_result(energy_eV, logistic_onset(0.10), mu_eV=onset)

        mu_grid = np.linspace(2.5, 5.5, 60)
        gentle_peak = thermoelectric_sweep(gentle, mu_grid, 300.0, tau_s=1.0).power_factor.max()
        sharp_peak = thermoelectric_sweep(sharp, mu_grid, 300.0, tau_s=1.0).power_factor.max()
        assert sharp_peak > gentle_peak


# --------------------------------------------------------------------------
# Result objects and defaulting
# --------------------------------------------------------------------------

class TestResultObjects:
    def test_point_defaults_chemical_potential_to_the_fermi_energy(self):
        """With no mu given, the DOSResult's Fermi energy must be used.

        WHY: the convenience default (mu = Fermi energy) must actually take
        effect, so a caller who omits mu gets the Fermi-level result rather
        than a silent zero or an error.

        GIVEN a DOSResult whose Fermi energy is 5 eV and params with mu
        left as None,
        WHEN a point is computed and compared with the same point at an
        explicit mu = 5 eV,
        THEN the two Seebeck values are identical and the reported mu is 5.
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=5.0)
        implicit = thermoelectric_point(result, TransportParams(temperature_K=300.0))
        explicit = thermoelectric_point(result, TransportParams(temperature_K=300.0, mu_eV=5.0))
        assert implicit.mu_eV == 5.0
        assert implicit.seebeck_uV_per_K == pytest.approx(explicit.seebeck_uV_per_K)

    def test_conductivity_is_flagged_absolute_only_when_tau_is_given(self):
        """The absolute flag must track whether a relaxation time was supplied.

        WHY: the whole honesty of the API rests on never presenting a
        relative conductivity as physical. The flag must be False when tau
        is defaulted and True only when the caller supplies one.

        GIVEN one DOSResult,
        WHEN a point is computed without tau and with tau,
        THEN conductivity_is_absolute is False then True respectively.
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=5.0)
        without = thermoelectric_point(result, TransportParams(temperature_K=300.0))
        with_tau = thermoelectric_point(result, TransportParams(temperature_K=300.0, tau_s=1e-14))
        assert without.conductivity_is_absolute is False
        assert with_tau.conductivity_is_absolute is True

    def test_sweep_returns_arrays_aligned_with_the_mu_grid(self):
        """A sweep must return one value per chemical-potential point.

        WHY: downstream plotting and peak-finding assume the output arrays
        line up with the input mu grid; a length mismatch would misalign
        every curve.

        GIVEN a mu grid of 30 points,
        WHEN a sweep is run,
        THEN each output array has length 30 and the mu grid is echoed back.
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=5.0)
        mu_grid = np.linspace(2.0, 8.0, 30)
        sweep = thermoelectric_sweep(result, mu_grid, 300.0)
        assert isinstance(sweep, ThermoelectricSweep)
        assert sweep.seebeck_uV_per_K.shape == (30,)
        assert sweep.conductivity.shape == (30,)
        assert sweep.power_factor.shape == (30,)
        assert np.allclose(sweep.mu_eV, mu_grid)


# --------------------------------------------------------------------------
# Input validation (exceptions, not asserts)
# --------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_temperature_is_rejected(self):
        """A negative temperature must raise ValueError.

        WHY: transport needs a finite Fermi window, so T <= 0 is
        meaningless here (unlike the Fermi-Dirac occupation, which treats
        T = 0 as a limit). Rejecting it prevents a divide-by-zero producing
        NaNs.

        GIVEN a valid DOSResult,
        WHEN a point is requested at T = -10 K,
        THEN a ValueError is raised.
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=5.0)
        with pytest.raises(ValueError):
            thermoelectric_point(result, TransportParams(temperature_K=-10.0))

    def test_zero_temperature_is_rejected(self):
        """T = 0 must raise ValueError for transport.

        WHY: at exactly T = 0 the Fermi window is a delta function with no
        numerical support on a grid; the model cannot integrate it, so it
        must fail loudly rather than return a spurious zero.

        GIVEN a valid DOSResult,
        WHEN a point is requested at T = 0,
        THEN a ValueError is raised.
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=5.0)
        with pytest.raises(ValueError):
            thermoelectric_point(result, TransportParams(temperature_K=0.0))

    def test_non_positive_relaxation_time_is_rejected(self):
        """A non-positive tau must raise ValueError.

        WHY: a relaxation time is a positive physical quantity; zero or
        negative tau is nonsensical and would produce a zero or
        sign-flipped conductivity.

        GIVEN a valid DOSResult,
        WHEN a point is requested with tau = 0,
        THEN a ValueError is raised.
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=5.0)
        with pytest.raises(ValueError):
            thermoelectric_point(result, TransportParams(temperature_K=300.0, tau_s=0.0))

    def test_chemical_potential_outside_the_grid_is_rejected(self):
        """A mu outside the energy grid must raise ValueError.

        WHY: if mu lies outside the grid the Fermi window is not captured by
        the DOS and every moment is silently truncated. Failing loudly
        forces the caller to widen the grid instead of trusting a wrong
        number.

        GIVEN a DOS grid that ends at 12 eV,
        WHEN a point is requested at mu = 20 eV,
        THEN a ValueError is raised.
        """
        energy_eV, dos = _free_electron_dos()
        result = _dos_result(energy_eV, dos, mu_eV=5.0)
        with pytest.raises(ValueError):
            thermoelectric_point(result, TransportParams(temperature_K=300.0, mu_eV=20.0))

    def test_mismatched_energy_and_dos_lengths_are_rejected(self):
        """Energy and DOS arrays of different lengths must raise ValueError.

        WHY: a length mismatch is a programming error that would otherwise
        broadcast into a wrong integral or a crash deep in the moments;
        catching it at the boundary gives a clear message.

        GIVEN an energy grid and a DOS array of different sizes,
        WHEN the transport distribution is formed,
        THEN a ValueError is raised.
        """
        energy_eV = np.linspace(0.01, 12.0, 100)
        dos = np.sqrt(np.linspace(0.01, 12.0, 90))
        with pytest.raises(ValueError):
            transport_distribution(energy_eV, dos)
