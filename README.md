# FX Heston calibration and simulation

A standalone numerical codebase for vector and matrix Heston FX models with
stochastic interest rates. Both MATLAB and Python provide calibration, Fourier
pricing and Monte Carlo simulation for the same vector and matrix models.
The Python implementation also retains independent standard-library pricing
and mathematical checks.

## Quick start in Python

Use Python 3.11 or later. Python 3.13, NumPy 2.5.4 and SciPy 1.18.1 were used for
local validation. MATLAB is not required for the Python workflows.

From the repository root:

```sh
python -m pip install -r requirements.txt
python python/run_heston.py vector price --factors 2 --maturity 1
python python/run_heston.py matrix price --all-maturities
python python/run_heston.py vector calibrate --factors 1 --max-iterations 150
python python/run_heston.py vector simulate --factors 1 --parameters results/python/vector_N1/calibration.json --scheme QE --paths 200000 --steps 1000
python python/run_heston.py matrix calibrate --max-iterations 150
python python/run_heston.py matrix simulate --parameters results/python/matrix/calibration.json --paths 200000 --steps 1000
```

Use `--scheme EM` for full-truncation Euler vector simulation. Omit
`--parameters` to price or simulate the saved example instead. `--parameters`
and `--initial-parameters` accept a MATLAB or Python calibration JSON file, or
raw parameters encoded as JSON. The one-factor MATLAB row-vector format is
accepted. Each command accepts `--data` for an input JSON file with the schema
of the corresponding file in `data/`, and `--output-dir` to keep experiments
separate. `price` selects one year by default; `--all-maturities` prices the full
surface. `--maturity` must select a maturity present in the input.

`--max-evaluations` limits calibration work. Vector calibration additionally
accepts `--alpha` and `--lambda`; matrix calibration accepts `--parallel` for
threaded finite differences. Vector simulation accepts `--batches` and `--seed`.
Both simulations accept `--paths`, `--steps` and `--maturity`. Run a subcommand
with `--help` for all controls. A completed command does not imply optimizer
convergence: inspect the JSON `exitflag`, solver message and fit diagnostics.

For direct use, add `python/` to your import path. The public functions are
`price_vector`, `calibrate_vector`, `simulate_vector` in `vector_model.py`, and
`price_matrix`, `calibrate_matrix`, `simulate_matrix` in `matrix_model.py`.
They return JSON-compatible dictionaries. `vector_prices` and `matrix_prices`
provide NumPy array pricing for custom strikes and maturities. Options accept
MATLAB names such as `Parameters`, `InitialParameters` and `OutputDir`, as well
as their snake_case equivalents; `OutputDir=None` disables library output.

```python
import sys
sys.path.insert(0, "python")
from vector_model import calibrate_vector, simulate_vector

fit = calibrate_vector(2, 150, {"OutputDir": "results/python/vector_N2"})
mc = simulate_vector(2, {"Parameters": fit["parameters"], "Scheme": "QE",
                         "Paths": 20000, "Steps": 250, "OutputDir": None})
```

## Quick start in MATLAB

MATLAB R2024b was used for validation. Pricing and simulation use base MATLAB;
calibration requires Optimization Toolbox. Computation is serial by default.
Parallel Computing Toolbox is needed only if you enable `UseParallel` for
matrix calibration.

From the repository root:

```matlab
addpath('matlab');
setupHeston;
verifyModels;                 % Check prices, parity, and matrix transforms.

% Run the saved parameter examples without fitting.
vectorMC = runVectorSimulation(1);
matrixMC = runMatrixSimulation;
```

### Calibrate and simulate a vector model

```matlab
fit = calibrateVector(1, 150); % One factor; use 2 for the two-factor example.
mc = runVectorSimulation(1, struct('Parameters', fit.parameters, ...
    'Scheme', 'QE', 'Paths', 200000, 'Steps', 1000));
```

`calibrateVector` fits the preserved 187 unique January 2013 quotes with the
corrected half-moment pricer. It minimises relative-price mean squared error
plus an elastic-net penalty on physical parameters. `Alpha` and `Lambda` default
to `1e-4` and `0.5`. The bounds retain the original vector parameter ranges:
`kappa` in `[0.01, 2]`, `sigma` in `[0.01, 1]`, `rho` in `[-1, 1]`, nonnegative
initial/long-run variance bounded by the largest market variance, rate loadings
in `[-1, 1]`, `am=0`, and the first FX loading fixed to one. Required moments
must remain finite over the fitted maturities.

This is a new local calibration driver for the corrected pricer. It does not
reconstruct the historical optimiser trajectory or claim to reproduce the saved
parameter estimates. Inspect `exitflag`, `output`, fit errors, and the final
quadrature check. A run stopped at its iteration limit is not a converged fit.

Simulation supports full-truncation Euler (`'EM'`) and quadratic-exponential
variance stepping (`'QE'`). The default is 20,000 paths, 250 steps, and 40 batches.
Use at least two batches and a path count divisible by the batch count.
`T`, `Steps`, `Paths`, `Blocks`, `Scheme`, `Seed`, `Parameters`, and `OutputDir`
can be supplied in the options structure. Maturities must occur in the input data.

### Calibrate and simulate the matrix model

```matlab
fit = calibrateMatrix(150);
mc = runMatrixSimulation(struct('Parameters', fit.parameters, ...
    'Paths', 200000, 'Steps', 1000, 'T', 1));
```

This retains the two independent 2-by-2 Wishart blocks, fixed `beta=1/2`,
relative-price elastic-net objective, and saved refinement starting point.
The example contains 135 quotes. `InitialParameters`, `MaxFunctionEvaluations`,
`UseParallel`, and `OutputDir` are calibration options.

Matrix simulation uses exact Gaussian OU-factor variance transitions, coupled
log-price Euler increments, and trapezoidal rate integration. Asset and rate
discretisation error remains. Paths are split into 40 batches and must be a
multiple of 40, with at least 80 paths. Batch seeds are fixed for reproducibility.

Both model families match domestic discount and foreign forward moments using
the same pricing transform. Parameter support is restricted to `am=0`; the
matrix implementation additionally requires disjoint rate and FX blocks.
Matrix FX loadings must be symmetric; unsupported nonsymmetric inputs are
rejected before pricing or simulation.

## Language parity and numerical validation

| Capability | MATLAB | Python |
| --- | --- | --- |
| Vector pricing and discount/forward curve matching | `vectorPrices` | `vector_prices`, `price_vector` |
| Vector bounds, physical elastic-net objective and final grid check | `calibrateVector` | `calibrate_vector` |
| Vector EM and QE, batch errors, characteristic functions and boundary diagnostics | `runVectorSimulation` | `simulate_vector` |
| Matrix Riccati pricing and discount/forward curve matching | `matrixAuditPrices` | `matrix_prices`, `price_matrix` |
| Matrix physical penalty, encoded constraints and saved refinement start | `calibrateMatrix` | `calibrate_matrix` |
| Matrix exact OU variance transitions and correlated FX increments | `runMatrixSimulation` | `simulate_matrix` |
| Reprice and simulate newly fitted parameters | `Parameters` option | `Parameters` option or `--parameters` |

The financial models, objectives, bounds, parameter conventions and simulation
schemes are shared. MATLAB uses `fmincon` SQP and `ode15s`; Python uses SciPy
SLSQP and BDF integration with a sparse analytic Jacobian. Optimizer trajectories,
stopping details and fitted parameters can therefore differ. The output records
the solver status; short or evaluation-limited fits exercise the workflow and
are not calibration-quality results.

Both languages seed independent simulation batches and estimate sampling
standard errors from their batch means. NumPy and MATLAB generate normals and
order draws differently, so equal seeds do not produce identical paths.
Compare simulated estimates statistically. Sampling errors exclude the remaining
time-discretisation bias; increase steps and paths to study it. Matrix Python
library options additionally allow `Blocks` and `Seed`, with defaults matching
MATLAB's 40 batches and base seed 922761.

Run all Python numerical and command-line workflow tests with:

```sh
python python/verify_framework.py
python -m unittest discover -s tests -v
```

The tests cover pricing, transforms, curve matching, calibration constraints,
simulations and saved-fit interchange. GitHub Actions installs dependencies and
runs these commands on pushes and pull requests. MATLAB checks run with:

```matlab
verifyModels(true); % Includes one-iteration fits and small Monte Carlo runs.
```

To generate a fresh cross-language comparison, run in MATLAB from the repository
root and then run the Python command below:

```matlab
addpath('matlab'); setupHeston;
exportLanguageParity; % Writes results/language_parity/matlab.json.
```

```sh
python python/verify_language_parity.py
```

This compares all example prices, raw transforms, initial penalized objectives,
and 20,000-path simulations with 250 time steps for both vector schemes and the
matrix model. Deterministic tolerances are recorded in the comparison JSON;
Monte Carlo comparisons allow six combined batch standard errors. It is a
regression check at the supplied examples, not a proof of accuracy for every
parameter choice. Use `exportLanguageParity(path, false)` and
`--no-simulations` for deterministic comparisons only.

### Independent standard-library checks

Python 3.10 or later is sufficient; no third-party packages or MATLAB installation
are required for these commands:

```sh
python python/verify_framework.py
python -m unittest discover -s tests -p test_vector_reference.py -v
python python/run_vector.py --factors 2 --maturity 1
```

Reprice a newly generated MATLAB vector calibration independently:

```sh
python python/run_vector.py --factors 1 --parameters results/vector_N1/calibration.json
```

The independent checker tests covariance, noncommuting Riccati equations,
scalar reduction, numeraire moments, saved vector prices and put-call parity.
It is kept separate from the NumPy/SciPy model implementations.

## Layout and outputs

| Directory | Purpose |
| --- | --- |
| `matlab/common/` | Data loading, Black-Scholes prices and IV inversion |
| `matlab/vector/` | Vector transform, pricing, calibration and simulation engine |
| `matlab/matrix/` | Matrix Riccati transform, pricing, calibration and simulation |
| `examples/` | Simulation entry points accepting saved or newly fitted parameters |
| `python/` | Vector/matrix pricing, calibration, simulation, CLI and independent checks |
| `data/` | Three compact numerical input files and price-check fixtures |
| `tests/` | MATLAB and Python numerical checks |
| `results/` | Generated results and checkpoints; ignored by Git |

Calibration writes `calibration.json`; matrix fitting also saves improving
checkpoints. Simulations write JSON summaries containing analytical references,
sampling standard errors and settings. Vector simulations additionally save
batch diagnostics in MATLAB `.mat` or Python `.npz` files; Python JSON uses
`null` for inapplicable QE direct-spot diagnostics. Python CLI results default
to `results/python/vector_N1`, `results/python/vector_N2` and
`results/python/matrix`; MATLAB defaults remain `results/vector_N1`,
`results/vector_N2` and `results/matrix`. Repeated runs with the same output
directory replace those outputs; use `OutputDir` or `--output-dir` to retain
separate experiments. Generated results are ignored by Git.

The example data preserve the numerical arrays and starting parameters used by
the research implementation. Vector IV inputs are decimal; matrix `marketIV`
inputs are percentages, as recorded in its `units` field. Prices are USD per EUR
and yields are continuously compounded annual decimals. Original vendor exports
and rate instrument identifiers are not included. The vector and matrix samples
use different quote subsets and rate arrays, so their fit errors are not a
controlled comparison of calibration performance.

This repository contains executable research code and required example inputs.
Manuscripts, reviewer records, figures, generated PDFs and local experiment
archives are maintained separately.
