# FX Heston calibration and simulation

A standalone numerical codebase for vector and matrix Heston FX models with
stochastic interest rates. MATLAB provides calibration, Fourier pricing, and
Monte Carlo simulation. Python provides independent pricing and mathematical
checks using only the standard library.

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

## Python checks and independent pricing

Python 3.10 or later is sufficient; no third-party packages or MATLAB installation
are required for these commands:

```sh
python python/verify_framework.py
python -m unittest discover -s tests -v
python python/run_vector.py --factors 2 --maturity 1
```

Reprice a newly generated MATLAB vector calibration independently:

```sh
python python/run_vector.py --factors 1 --parameters results/vector_N1/calibration.json
```

The Python checker tests covariance, noncommuting Riccati equations, scalar
reduction, numeraire moments, saved vector prices, and put-call parity. GitHub
Actions runs these checks on every push. MATLAB calibration and simulation are
validated locally with:

```matlab
verifyModels(true); % Includes one-iteration fits and small Monte Carlo runs.
```

These short fits exercise the workflows; they are not calibration-quality runs.

## Layout and outputs

| Directory | Purpose |
| --- | --- |
| `matlab/common/` | Data loading, Black-Scholes prices and IV inversion |
| `matlab/vector/` | Vector transform, pricing, calibration and simulation engine |
| `matlab/matrix/` | Matrix Riccati transform, pricing, calibration and simulation |
| `examples/` | Simulation entry points accepting saved or newly fitted parameters |
| `python/` | Independent pricing and mathematical verification |
| `data/` | Three compact numerical input files and price-check fixtures |
| `tests/` | MATLAB and Python numerical checks |
| `results/` | Generated results and checkpoints; ignored by Git |

Calibration writes `calibration.json`; matrix fitting also saves improving
checkpoints. Simulations write JSON summaries containing analytical references,
sampling standard errors and settings. Vector simulations additionally save a
MAT file. Repeated runs with the same output directory replace those outputs;
use `OutputDir` to retain separate experiments.

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
