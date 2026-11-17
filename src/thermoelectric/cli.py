# -*- coding: utf-8 -*-
"""
Command-line interface for Thermoelectric Explorer.

Computes a density of states for a rectangular quantum box (via the
Quantum DOS Explorer), then reports the thermoelectric transport
parameters -- Seebeck coefficient, electrical conductivity, and power
factor -- at a chosen chemical potential and temperature, and optionally
writes a chemical-potential sweep to CSV.

Examples
--------
Single operating point at the Fermi level, summary printed to stdout::

    thermoelectric --lx 5 --ly 5 --lz 5 --mass 1.0 --sigma 0.05 \\
        --density 5.86e28 --temperature 300

Sweep the chemical potential and save the S/sigma/PF curves::

    thermoelectric --lx 5 --ly 5 --lz 5 --mass 1.0 --sigma 0.05 \\
        --density 5.86e28 --sweep --output sweep.csv

See ``thermoelectric --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from quantum_dos.constants import SILVER_ELECTRON_DENSITY_M3
from quantum_dos.models import QuantumBox
from quantum_dos.physics import calculate_dos

from .transport import TransportParams, thermoelectric_point, thermoelectric_sweep

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the command-line argument parser.

    Defines every option the ``thermoelectric`` command accepts -- the
    box geometry and physics parameters that fix the density of states,
    the operating point (chemical potential, temperature, optional
    relaxation time), the numerical grid, and the output options -- with
    their units, defaults, and help text. Kept separate from
    :func:`main` so the parser can be inspected and tested on its own.

    Returns
    -------
    argparse.ArgumentParser
        The fully configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="thermoelectric",
        description=(
            "Compute the Seebeck coefficient, electrical conductivity and "
            "power factor of a rectangular quantum box (free-electron model) "
            "in the constant relaxation-time approximation, from its "
            "density of states."
        ),
        epilog=(
            "The Seebeck coefficient is absolute (microvolts/kelvin); the "
            "relaxation time cancels in it. Conductivity and power factor "
            "scale with the relaxation time and the (un-normalized) DOS, so "
            "they are reported in arbitrary units.\n\n"
            f"Example: thermoelectric --lx 5 --ly 5 --lz 5 --mass 1.0 "
            f"--sigma 0.05 --density {SILVER_ELECTRON_DENSITY_M3:.3g} "
            f"--temperature 300"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    geometry = parser.add_argument_group("box geometry")
    geometry.add_argument(
        "--lx", type=float, required=True, metavar="NM",
        help="Box side length along x, in nanometres. Must be > 0.",
    )
    geometry.add_argument(
        "--ly", type=float, required=True, metavar="NM",
        help="Box side length along y, in nanometres. Must be > 0.",
    )
    geometry.add_argument(
        "--lz", type=float, required=True, metavar="NM",
        help="Box side length along z, in nanometres. Must be > 0.",
    )

    physics = parser.add_argument_group("physics parameters")
    physics.add_argument(
        "--mass", type=float, required=True, metavar="M_E",
        help="Electron effective mass, in units of the free-electron mass "
             "m_e (e.g. 1.0 = free-electron mass). Must be > 0.",
    )
    physics.add_argument(
        "--sigma", type=float, required=True, metavar="EV",
        help="Gaussian broadening width of each discrete energy level, in "
             "eV. Must be > 0. Typical experimental resolution 0.05-0.2 eV.",
    )
    physics.add_argument(
        "--density", type=float, required=True, metavar="M^-3",
        help="Electron number density used to fill the box and set the "
             "Fermi energy, in electrons per cubic metre. Must be > 0 "
             f"(e.g. bulk silver is approximately "
             f"{SILVER_ELECTRON_DENSITY_M3:.3g} m^-3).",
    )

    operating = parser.add_argument_group("operating point")
    operating.add_argument(
        "--temperature", type=float, default=300.0, metavar="K",
        help="Temperature, in Kelvin. Must be > 0. Default: 300 K.",
    )
    operating.add_argument(
        "--mu", type=float, default=None, metavar="EV",
        help="Chemical potential, in eV. Default: the computed Fermi "
             "energy of the box. Must lie within the energy grid.",
    )
    operating.add_argument(
        "--tau", type=float, default=None, metavar="S",
        help="Constant relaxation time, in seconds. If omitted, "
             "conductivity and power factor are in arbitrary units; if "
             "given, it fixes their relaxation-time scaling (the DOS "
             "normalization remains arbitrary). Must be > 0.",
    )

    numerics = parser.add_argument_group("numerical grid")
    numerics.add_argument(
        "--e-max", type=float, default=12.0, metavar="EV", dest="e_max",
        help="Upper bound of the energy grid and state-enumeration cutoff, "
             "in eV. Must be > 0. Default: 12.0 eV.",
    )
    numerics.add_argument(
        "--n-points", type=int, default=2000, metavar="N", dest="n_points",
        help="Number of points in the energy grid. A fine grid is used by "
             "default because the transport moments integrate over the "
             "narrow Fermi window. Default: 2000.",
    )

    sweep = parser.add_argument_group("chemical-potential sweep")
    sweep.add_argument(
        "--sweep", action="store_true",
        help="Instead of a single point, sweep the chemical potential "
             "across the energy grid and report (with --output) the "
             "S/sigma/PF curves plus the mu of peak power factor.",
    )
    sweep.add_argument(
        "--sweep-points", type=int, default=200, metavar="N",
        dest="sweep_points",
        help="Number of chemical-potential values in the sweep. "
             "Default: 200.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output", type=str, default=None, metavar="PATH",
        help="If given with --sweep, save the sweep to this CSV file with "
             "columns: mu_eV, seebeck_uV_per_K, conductivity_arb, "
             "power_factor_arb.",
    )
    output.add_argument(
        "--quiet", action="store_true",
        help="Suppress the human-readable summary on stdout.",
    )

    return parser


def _run(args: argparse.Namespace) -> int:
    """Run one calculation from parsed arguments and report the result.

    Builds the box, computes its density of states, then either reports a
    single thermoelectric operating point or a chemical-potential sweep.
    Invalid input is reported as a clean error on stderr with a non-zero
    exit code rather than an uncaught exception.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 if the calculation could not
        be performed (e.g. invalid parameters or too few enumerated
        states for the requested filling).
    """
    try:
        box = QuantumBox(
            lx_nm=args.lx, ly_nm=args.ly, lz_nm=args.lz,
            effective_mass=args.mass,
        )
        dos = calculate_dos(
            box,
            sigma_eV=args.sigma,
            electron_density_m3=args.density,
            e_max_eV=args.e_max,
            n_points=args.n_points,
        )

        if args.sweep:
            mu_grid = np.linspace(dos.energy_eV[0], dos.energy_eV[-1],
                                  args.sweep_points)
            result = thermoelectric_sweep(
                dos, mu_grid, args.temperature, tau_s=args.tau
            )
        else:
            point = thermoelectric_point(
                dos,
                TransportParams(
                    temperature_K=args.temperature, mu_eV=args.mu, tau_s=args.tau
                ),
            )
    except ValueError as exc:
        print(f"thermoelectric: error: {exc}", file=sys.stderr)
        return 1

    tau_note = "fixed by --tau" if args.tau is not None else "arb. units"

    if args.sweep:
        peak = int(np.argmax(result.power_factor))
        if not args.quiet:
            print(f"Box            : {box.lx_nm:g} x {box.ly_nm:g} x "
                  f"{box.lz_nm:g} nm")
            print(f"Fermi energy   : {dos.fermi_energy_eV:.4f} eV")
            print(f"Temperature    : {args.temperature:g} K")
            print(f"Sweep          : {args.sweep_points} points, "
                  f"mu in [{result.mu_eV[0]:.3f}, {result.mu_eV[-1]:.3f}] eV")
            print(f"Peak power factor at mu = {result.mu_eV[peak]:.4f} eV")
            print(f"  S   = {result.seebeck_uV_per_K[peak]:.3f} uV/K")
            print(f"  PF  = {result.power_factor[peak]:.4g} ({tau_note})")
        if args.output:
            with open(args.output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["mu_eV", "seebeck_uV_per_K", "conductivity_arb",
                     "power_factor_arb"]
                )
                for mu, s, c, pf in zip(
                    result.mu_eV, result.seebeck_uV_per_K,
                    result.conductivity, result.power_factor
                ):
                    writer.writerow(
                        [f"{mu:.6f}", f"{s:.6f}", f"{c:.6e}", f"{pf:.6e}"]
                    )
            if not args.quiet:
                print(f"Saved sweep to : {args.output}")
    else:
        if not args.quiet:
            print(f"Box            : {box.lx_nm:g} x {box.ly_nm:g} x "
                  f"{box.lz_nm:g} nm")
            print(f"Fermi energy   : {dos.fermi_energy_eV:.4f} eV")
            print(f"Chemical pot.  : {point.mu_eV:.4f} eV")
            print(f"Temperature    : {point.temperature_K:g} K")
            print(f"Seebeck S      : {point.seebeck_uV_per_K:.4f} uV/K "
                  f"(absolute)")
            print(f"Conductivity   : {point.conductivity:.4g} ({tau_note})")
            print(f"Power factor   : {point.power_factor:.4g} ({tau_note})")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the ``thermoelectric`` command.

    Parses the command-line arguments (or ``argv`` if given, which makes
    the entry point testable in-process) and runs one calculation.

    Parameters
    ----------
    argv : list of str, optional
        Argument list to parse. Defaults to ``None``, in which case the
        real process arguments are used.

    Returns
    -------
    int
        The process exit code (0 on success, non-zero on error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
