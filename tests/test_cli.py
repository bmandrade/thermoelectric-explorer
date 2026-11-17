# -*- coding: utf-8 -*-
"""Tests for the thermoelectric command-line interface.

These exercise the CLI in-process via ``main(argv)`` (which returns an
exit code) so no subprocess or display is needed. Each test states WHY
it exists, then GIVEN/WHEN/THEN, and checks a single behaviour.
"""

import csv

import pytest

from thermoelectric.cli import build_parser, main


# A minimal valid argument set reused across tests (silver-like density,
# a 5 nm cube, modest broadening).
_BASE_ARGS = [
    "--lx", "5", "--ly", "5", "--lz", "5",
    "--mass", "1.0", "--sigma", "0.05", "--density", "5.86e28",
]


class TestParser:
    def test_requires_the_box_geometry(self):
        """The parser must reject a call with no geometry.

        WHY: box geometry has no sensible default -- it defines the system
        -- so the CLI must fail loudly rather than invent one. argparse
        signals a missing required argument by exiting with code 2.

        GIVEN no arguments,
        WHEN the parser parses them,
        THEN it raises SystemExit (argparse's required-argument error).
        """
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestSinglePoint:
    def test_single_point_run_succeeds_and_prints_seebeck(self, capsys):
        """A valid single-point run must exit 0 and report the Seebeck value.

        WHY: the default (non-sweep) path is the primary use of the CLI;
        this is its smoke test -- parameters flow through to a density of
        states, to a transport point, to a printed, unit-labelled summary.

        GIVEN a valid box and operating point,
        WHEN the CLI is run,
        THEN it exits 0 and prints a Seebeck line in uV/K.
        """
        code = main(_BASE_ARGS + ["--temperature", "300"])
        out = capsys.readouterr().out
        assert code == 0
        assert "Seebeck" in out
        assert "uV/K" in out

    def test_quiet_suppresses_the_summary(self, capsys):
        """--quiet must silence stdout while still succeeding.

        WHY: scripted callers want the exit code without the human summary;
        --quiet must remove the printed report but not change the outcome.

        GIVEN a valid run with --quiet,
        WHEN the CLI is run,
        THEN it exits 0 and prints nothing to stdout.
        """
        code = main(_BASE_ARGS + ["--quiet"])
        out = capsys.readouterr().out
        assert code == 0
        assert out.strip() == ""

    def test_invalid_temperature_exits_nonzero_with_a_message(self, capsys):
        """A non-positive temperature must exit 1 with an error on stderr.

        WHY: bad physical input must be a clean CLI error, not a traceback;
        transport needs T > 0, so T = 0 is rejected at the boundary.

        GIVEN a valid box but temperature 0,
        WHEN the CLI is run,
        THEN it exits 1 and writes an error to stderr.
        """
        code = main(_BASE_ARGS + ["--temperature", "0"])
        err = capsys.readouterr().err
        assert code == 1
        assert "error" in err.lower()


class TestSweep:
    def test_sweep_writes_a_csv_with_the_expected_columns(self, tmp_path, capsys):
        """A sweep with --output must write a well-formed CSV.

        WHY: the sweep CSV is the data product used for plotting the
        confinement signature; its header and row shape are a contract. The
        occupied columns must be present and every data row must have four
        numeric fields.

        GIVEN a sweep run with an output path,
        WHEN the CLI is run,
        THEN the file exists with the documented header and four columns
        per row.
        """
        out_path = tmp_path / "sweep.csv"
        code = main(
            _BASE_ARGS + ["--sweep", "--sweep-points", "20",
                          "--output", str(out_path)]
        )
        assert code == 0
        assert out_path.exists()

        with open(out_path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == [
            "mu_eV", "seebeck_uV_per_K", "conductivity_arb", "power_factor_arb"
        ]
        assert len(rows) == 1 + 20
        for data_row in rows[1:]:
            assert len(data_row) == 4
            for field in data_row:
                float(field)  # raises if any field is not numeric

    def test_sweep_reports_the_peak_power_factor(self, capsys):
        """A sweep must report where the power factor peaks.

        WHY: locating the optimal chemical potential (carrier filling) is
        the point of the sweep -- that is the quantity confinement is meant
        to improve -- so the summary must name the peak explicitly.

        GIVEN a sweep run,
        WHEN the CLI is run,
        THEN the summary mentions the peak power factor.
        """
        code = main(_BASE_ARGS + ["--sweep", "--sweep-points", "20"])
        out = capsys.readouterr().out
        assert code == 0
        assert "Peak power factor" in out
