# -*- coding: utf-8 -*-
"""Tests for quantum figure-of-merit (absolute transport, kappa_lattice, ZT).

The central oracle here is the Wiedemann-Franz law: the electronic thermal
conductivity and the electrical conductivity must share the Sommerfeld
Lorenz number in the degenerate limit. That single exact relation
validates that sigma and kappa_e are on a mutually consistent absolute
scale. Each test states WHY it exists, then GIVEN/WHEN/THEN.
"""

import numpy as np
import pytest

from quantum_dos import QuantumBox, calculate_dos
from thermoelectric import seebeck_coefficient, TransportParams
from thermoelectric.figure_of_merit import (
    LORENZ_SOMMERFELD,
    ZTResult,
    absolute_transport,
    figure_of_merit,
    lattice_thermal_conductivity,
    lorenz_number,
)


def _dos(box, sigma_eV=0.05):
    """A silver-filled DOS for a box (shared fixture-like helper)."""
    return calculate_dos(box, sigma_eV=sigma_eV,
                         electron_density_m3=5.86e28, n_points=2000)


# --------------------------------------------------------------------------
# Absolute transport: the Wiedemann-Franz oracle
# --------------------------------------------------------------------------

class TestAbsoluteTransport:
    def test_lorenz_number_matches_wiedemann_franz(self):
        """kappa_e / (sigma T) must equal the Sommerfeld Lorenz number.

        WHY: this is THE validation that the absolute scales of the
        electrical and thermal conductivities are mutually consistent. The
        Lorenz number is a scattering- and material-independent constant
        (pi^2/3)(k_B/e)^2 in the degenerate limit; if sigma and kappa_e are
        computed on a consistent footing their ratio reproduces it,
        independent of the (unknown) relaxation time, which cancels.

        GIVEN a metallic (silver-filled) box evaluated at its Fermi level,
        WHEN sigma and kappa_e are computed and the Lorenz number formed,
        THEN it equals the Sommerfeld value to a few percent (the residual
        is the higher-order Sommerfeld correction at finite temperature).
        """
        box = QuantumBox(10.0, 10.0, 10.0, 1.0)
        tr = absolute_transport(_dos(box), box=box, temperature_K=300.0,
                                tau_s=1e-14, broadening_eV=0.05)
        assert lorenz_number(tr) == pytest.approx(LORENZ_SOMMERFELD, rel=0.05)

    def test_seebeck_matches_the_relative_module(self):
        """The SI Seebeck must equal the arbitrary-units module's Seebeck.

        WHY: the Seebeck coefficient is a ratio in which the relaxation time
        and DOS normalization cancel, so the absolute (SI) path and the
        relative (CRTA) path must give the identical value. Agreement proves
        the new SI machinery did not corrupt the one quantity both share.

        GIVEN the same box and operating point,
        WHEN S is computed by absolute_transport and by seebeck_coefficient,
        THEN the two agree to tight tolerance.
        """
        box = QuantumBox(10.0, 10.0, 3.0, 1.0)
        dos = _dos(box)
        si = absolute_transport(dos, box=box, temperature_K=300.0,
                                tau_s=1e-14, broadening_eV=0.05)
        rel = seebeck_coefficient(dos, TransportParams(temperature_K=300.0))
        assert si.seebeck_uV_per_K == pytest.approx(rel, rel=1e-6)

    def test_conductivity_scales_linearly_with_relaxation_time(self):
        """sigma must double when tau doubles.

        WHY: in CRTA sigma is proportional to tau; this pins the absolute
        scale's linear dependence on the one material input.

        GIVEN one box and operating point,
        WHEN sigma is computed at tau and 2*tau,
        THEN sigma(2 tau) is twice sigma(tau).
        """
        box = QuantumBox(10.0, 10.0, 10.0, 1.0)
        dos = _dos(box)
        s1 = absolute_transport(dos, box=box, temperature_K=300.0,
                                tau_s=1e-14, broadening_eV=0.05)
        s2 = absolute_transport(dos, box=box, temperature_K=300.0,
                                tau_s=2e-14, broadening_eV=0.05)
        assert s2.conductivity_S_per_m == pytest.approx(
            2.0 * s1.conductivity_S_per_m, rel=1e-9)

    def test_non_positive_relaxation_time_is_rejected(self):
        """tau <= 0 must raise ValueError.

        WHY: the absolute scale is meaningless without a positive relaxation
        time; the boundary must reject it rather than return zero or a
        sign-flipped conductivity.

        GIVEN a valid box, WHEN absolute_transport is called with tau = 0,
        THEN a ValueError is raised.
        """
        box = QuantumBox(10.0, 10.0, 10.0, 1.0)
        with pytest.raises(ValueError):
            absolute_transport(_dos(box), box=box, temperature_K=300.0,
                               tau_s=0.0, broadening_eV=0.05)


# --------------------------------------------------------------------------
# Lattice thermal conductivity
# --------------------------------------------------------------------------

class TestLatticeThermalConductivity:
    def test_recovers_bulk_value_for_large_size(self):
        """kappa_lattice -> kappa_bulk as the size greatly exceeds the mfp.

        WHY: a large structure has no boundary-scattering penalty, so the
        model must reduce to the bulk value; otherwise the whole size
        dependence would be miscalibrated.

        GIVEN a size far above the phonon mean free path,
        WHEN kappa_lattice is evaluated,
        THEN it is within a percent of kappa_bulk.
        """
        kap = lattice_thermal_conductivity(1000.0, kappa_bulk_W_per_mK=1.5,
                                           phonon_mfp_nm=5.0)
        assert kap == pytest.approx(1.5, rel=0.01)

    def test_is_suppressed_far_below_the_mean_free_path(self):
        """kappa_lattice must fall well below bulk when size << mfp.

        WHY: the physical point of confinement is that boundary scattering
        throttles phonons; a size much smaller than the mean free path must
        give a strongly reduced lattice conductivity.

        GIVEN a size far below the phonon mean free path,
        WHEN kappa_lattice is evaluated,
        THEN it is a small fraction of the bulk value.
        """
        kap = lattice_thermal_conductivity(0.5, kappa_bulk_W_per_mK=1.5,
                                           phonon_mfp_nm=5.0)
        assert kap < 0.5 * 1.5

    def test_increases_monotonically_with_size(self):
        """kappa_lattice must grow with size.

        WHY: less confinement means less boundary scattering means higher
        lattice conductivity; a non-monotonic model would be unphysical.

        GIVEN three increasing sizes,
        WHEN kappa_lattice is evaluated at each,
        THEN the values increase.
        """
        vals = [lattice_thermal_conductivity(s) for s in (1.0, 3.0, 10.0)]
        assert vals[0] < vals[1] < vals[2]


# --------------------------------------------------------------------------
# Figure of merit ZT
# --------------------------------------------------------------------------

class TestFigureOfMerit:
    def test_zt_is_positive_and_dimensionless(self):
        """ZT must be a positive finite number.

        WHY: ZT = S^2 sigma T / (kappa_e + kappa_lattice) is a ratio of
        positive quantities; a negative or non-finite result would signal a
        sign or units error.

        GIVEN a confined box at a lightly-doped operating point,
        WHEN ZT is computed,
        THEN it is finite and positive.
        """
        box = QuantumBox(10.0, 10.0, 2.0, 1.0)
        zt = figure_of_merit(_dos(box), box=box, temperature_K=300.0,
                             tau_s=1e-14, broadening_eV=0.05, mu_eV=0.3)
        assert isinstance(zt, ZTResult)
        assert np.isfinite(zt.zt) and zt.zt > 0.0

    def test_lower_lattice_conductivity_raises_zt(self):
        """Reducing kappa_lattice must increase ZT, all else equal.

        WHY: the entire low-dimensional thermoelectric strategy rests on
        this -- suppressing the lattice thermal conductivity raises the
        figure of merit. The model must reflect that monotonic dependence.

        GIVEN the same box and operating point but a smaller phonon mean
        free path (hence larger kappa_lattice) versus a larger one,
        WHEN ZT is computed for each,
        THEN the case with the lower kappa_lattice has the higher ZT.
        """
        box = QuantumBox(10.0, 10.0, 2.0, 1.0)
        dos = _dos(box)
        kw = dict(box=box, temperature_K=300.0, tau_s=1e-14,
                  broadening_eV=0.05, mu_eV=0.3)
        high_kappa = figure_of_merit(dos, phonon_mfp_nm=1.0, **kw)   # less suppression
        low_kappa = figure_of_merit(dos, phonon_mfp_nm=20.0, **kw)   # more suppression
        assert low_kappa.kappa_lattice_W_per_mK < high_kappa.kappa_lattice_W_per_mK
        assert low_kappa.zt > high_kappa.zt
