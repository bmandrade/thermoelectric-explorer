# Thermoelectric Explorer

Compute the **Seebeck coefficient**, **electrical conductivity**, and
**power factor** of a low-dimensional electron gas, in the constant
relaxation-time approximation (CRTA), from a density of states produced
by the [Quantum DOS Explorer](https://github.com/bmandrade/quantum-dos-explorer).

All three parameters are moments of the **transport distribution
function** Σ(E) against the Fermi window (−∂f/∂E). The package has no
plotting dependency in its scientific core, so it can be imported,
scripted, and tested independently of any GUI.

## What it computes, and in what units

| Quantity | Symbol | Unit | Absolute? |
| --- | --- | --- | --- |
| Seebeck coefficient | S | µV/K | **Yes** — the relaxation time cancels, so this is a genuine physical prediction |
| Electrical conductivity | σ | arbitrary units (∝ τ) | No — relative |
| Power factor | PF = S²σ | arbitrary units (∝ τ) | No — relative |

The honesty of that table is a design goal: because the relaxation time
τ cancels in the ratio that defines S, the Seebeck coefficient is
absolute, while σ and PF scale with the unknown τ and with the
un-normalized DOS, so they are reported as relative curves useful for
comparison (e.g. bulk vs confined), not as SI values.

## Scientific model

For free electrons the group velocity satisfies v(E)² ∝ E, so the
transport distribution function is

    Σ(E) = g(E) · E · τ

where g(E) is the density of states. Defining the moment integrals
(α = 0, 1, 2)

    L[α](µ, T) = ∫ Σ(E) · (E − µ)^α · (−∂f/∂E) dE

the parameters are

    σ = L[0]
    S = −(1/T) · (L[1] / L[0])
    PF = S² · σ

A built-in Mott/Sommerfeld low-temperature estimate (`mott_seebeck`)
provides an independent oracle that the full integral must reproduce as
T → 0.

### References

- L. D. Hicks & M. S. Dresselhaus, *Effect of quantum-well structures on
  the thermoelectric figure of merit*, Phys. Rev. B **47**, 12727 (1993).
- G. D. Mahan & J. O. Sofo, *The best thermoelectric*, PNAS **93**, 7436
  (1996).
- G. K. H. Madsen, J. Carrete & M. J. Verstraete, *BoltzTraP2*, Comput.
  Phys. Commun. **231**, 140 (2018).
- A. Batabyal & B. N. Dev, Physica E **64**, 224 (2014) — the DOS model
  reused from the Quantum DOS Explorer.

## Installation

This package depends on the Quantum DOS Explorer, which is not on PyPI.
Install that first, then this package:

```bash
# 1. the density-of-states dependency
pip install "git+https://github.com/bmandrade/quantum-dos-explorer"

# 2. this package (editable, with test extras)
git clone https://github.com/bmandrade/thermoelectric-explorer
cd thermoelectric-explorer
pip install -e ".[dev]"
```

## Usage

### Python API

```python
from quantum_dos import QuantumBox, calculate_dos
from thermoelectric import thermoelectric_point, TransportParams

box = QuantumBox(lx_nm=5.0, ly_nm=5.0, lz_nm=5.0, effective_mass=1.0)
dos = calculate_dos(box, sigma_eV=0.05, electron_density_m3=5.86e28)

result = thermoelectric_point(dos, TransportParams(temperature_K=300.0))
print(result.seebeck_uV_per_K)          # absolute, µV/K
print(result.conductivity)              # arbitrary units (∝ τ)
print(result.power_factor)              # arbitrary units
print(result.conductivity_is_absolute)  # False unless a τ was supplied
```

Sweeping the chemical potential (carrier filling) reveals the
confinement signature:

```python
import numpy as np
from thermoelectric import thermoelectric_sweep, interior_mu_grid

# interior_mu_grid stays clear of the energy-grid edges, where the Fermi
# window would be truncated and spike the curves (a numerical artifact).
mu_grid = interior_mu_grid(dos, temperature_K=300.0, n=300)
sweep = thermoelectric_sweep(dos, mu_grid, temperature_K=300.0)
peak = int(np.argmax(sweep.power_factor))
print(sweep.mu_eV[peak], sweep.power_factor[peak])
```

See `examples/confinement_seebeck.py` for a bulk-vs-confined comparison
(it compares the Seebeck coefficient, which — unlike the arbitrary-units
power factor — is absolute and so comparable across different systems).

### Command line

```bash
# single operating point at the Fermi level
thermoelectric --lx 5 --ly 5 --lz 5 --mass 1.0 --sigma 0.05 \
    --density 5.86e28 --temperature 300

# sweep the chemical potential and save the curves
thermoelectric --lx 5 --ly 5 --lz 5 --mass 1.0 --sigma 0.05 \
    --density 5.86e28 --sweep --output sweep.csv
```

Run `thermoelectric --help` for the full list of options.

### Interactive GUI

An interactive window plots the S / σ / PF chemical-potential sweep and
updates live as you drag sliders for the box geometry, effective mass,
temperature, and broadening. It needs the `[gui]` extra (matplotlib):

```bash
pip install -e ".[gui]"
thermoelectric-gui
```

On very new Python/Tk combinations (e.g. Python 3.14) matplotlib's
default TkAgg backend has a slider bug; the GUI automatically prefers a
Qt backend if one is installed (`pip install PyQt5`). You can force a
backend with the `MPLBACKEND` environment variable.

## Public API

| Function | Returns | Notes |
| --- | --- | --- |
| `transport_distribution(energy_eV, dos, tau_s=1.0)` | Σ(E) array | g(E)·E·τ; the one place v²∝E lives |
| `fermi_window(energy_eV, mu_eV, temperature_K)` | (−∂f/∂E) array | analytic, 1/eV |
| `transport_moments(energy_eV, sigma_E, mu_eV, temperature_K)` | (L0, L1, L2) | |
| `mott_seebeck(energy_eV, sigma_E, mu_eV, temperature_K)` | S (µV/K) | low-T oracle |
| `seebeck_coefficient(dos_result, params)` | S (µV/K) | **absolute** |
| `electrical_conductivity(dos_result, params)` | σ | arb. units |
| `power_factor(dos_result, params)` | PF | arb. units |
| `thermoelectric_point(dos_result, params)` | `ThermoelectricResult` | all three at one (µ, T) |
| `thermoelectric_sweep(dos_result, mu_grid_eV, temperature_K, tau_s=None)` | `ThermoelectricSweep` | curves vs µ |
| `interior_mu_grid(dos_result, temperature_K, n=200)` | µ array | sweep grid inset from the grid edges |

Invalid input (T ≤ 0, τ ≤ 0, µ outside the energy grid, mismatched
arrays) raises `ValueError`; asserts are reserved for internal
invariants.

## Testing

```bash
pytest
```

The suite is written in a behaviour-driven style: each test documents
**why** it exists and a GIVEN/WHEN/THEN scenario, and is designed to
fail for a single reason. It leans on the model's strong oracles — the
Mott/Sommerfeld limit, the exact τ-cancellation in S, the carrier-type
sign rule, and the Hicks–Dresselhaus "sharper DOS → larger peak power
factor" direction test.

## Limitations

- **CRTA only.** The relaxation time is a single energy- and
  temperature-independent constant. Real scattering depends on both.
- **σ and PF are relative.** Without a supplied τ they are in arbitrary
  units; even with τ, the un-normalized DOS leaves the absolute scale
  convention-dependent.
- **Free-electron parabolic bands.** The v²∝E weighting assumes a single
  parabolic band; this is a pedagogical/qualitative model, not a
  replacement for a DFT + BoltzTraP2 calculation.
- **No full ZT.** The figure of merit ZT = S²σT/κ needs the lattice
  thermal conductivity, which a DOS-based electronic model cannot
  provide. This package computes the electronic power factor, never ZT.
- **Inherits the DOS model's limits** (confinement thresholds, Gaussian
  broadening, finite-box enumeration) from the Quantum DOS Explorer.

## Use of AI tools

AI tools (Anthropic's Claude) were used extensively throughout the
development of this project — including refactoring, writing source code,
tests, and documentation, and for optimization and debugging. The
scientific direction, the choice of model and approximations, and the
validation strategy are the author's own, and all AI-assisted output was
reviewed and checked (notably against the analytic oracles in the test
suite) before inclusion.

## License

MIT — see [LICENSE](LICENSE).
