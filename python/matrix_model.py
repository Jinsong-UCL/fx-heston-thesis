"""Two-block Wishart FX pricing, physical elastic-net calibration and simulation.

This is the Python counterpart of ``matlab/matrix``.  It uses the declared
SDE covariance order ``sigma.T @ rho @ (an-am)``.  NumPy/SciPy and MATLAB use
different ODE/optimisation implementations and normal-number streams; their
prices and distributions, rather than individual optimisation steps or paths,
are comparable.  Rates and volatilities are decimal-valued unless a field
explicitly contains ``PP`` or is the supplied market's ``marketIV`` (percent).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import lru_cache
import json
from pathlib import Path
import threading
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.sparse import bsr_matrix
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
MATRIX_FIELDS = ("kappa", "sigma", "rho", "v0", "am", "an", "rm", "rn")
PENALTY_CONVENTION = (
    "Concatenate physical kappa,sigma,rho,v0,am,an,rm,rn,beta for each block. "
    "Full matrix entries, including symmetric duplicates and fixed entries; "
    "no optimiser scaling."
)
CALIBRATION_SETTINGS = {"nodes": 210, "cutoff": 900, "tolerance": 1e-10}
FINE_SETTINGS = {"nodes": 560, "cutoff": 1200, "tolerance": 1e-10}


def load_example():
    """Read the same compact, public January 2013 input used by MATLAB."""
    return json.loads((ROOT / "data" / "matrix.json").read_text(encoding="utf-8"))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_result(directory, filename, result):
    if directory is not None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / filename).write_text(
            json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    if not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _options(options, defaults):
    aliases = {key.replace("_", "").lower(): key for key in defaults}
    result, seen = dict(defaults), set()
    for key, value in (options or {}).items():
        canonical = aliases.get(key.replace("_", "").lower())
        if canonical is None:
            raise ValueError(f"Unknown option: {key}")
        if canonical in seen:
            raise ValueError(f"Duplicate option: {canonical}")
        result[canonical] = value
        seen.add(canonical)
    return result


def _parameters(parameters):
    result = []
    for original in parameters:
        block = {}
        for name in MATRIX_FIELDS:
            item = np.asarray(original[name], dtype=float)
            if item.shape != (2, 2) or not np.all(np.isfinite(item)):
                raise ValueError(f"Each {name} must be a finite 2-by-2 matrix.")
            block[name] = item.copy()
        beta = float(original["beta"])
        if not np.isfinite(beta):
            raise ValueError("beta must be finite.")
        block["beta"] = beta
        result.append(block)
    if not result:
        raise ValueError("At least one variance block is required.")
    return result


def validate_matrix_diffusion(parameters):
    """Validate the same 2D Wishart admissibility conditions as MATLAB."""
    blocks = _parameters(parameters)
    for block in blocks:
        v0 = block["v0"]
        if np.linalg.norm(v0 - v0.T) >= 1e-12:
            raise ValueError("v0 must be symmetric.")
        if np.linalg.eigvalsh(v0).min() < -1e-12:
            raise ValueError("v0 must be positive semidefinite.")
        if np.linalg.matrix_rank(block["sigma"]) != 2:
            raise ValueError("sigma must have full rank.")
        if block["beta"] < 0.25:
            raise ValueError("beta must be >= (d-1)/4 = 0.25.")
        if np.linalg.norm(block["rho"], 2) > 1 + 1e-12:
            raise ValueError("rho must be a contraction (spectral norm <= 1).")
        if np.any(block["am"] != 0):
            raise ValueError("The reference-currency SDE requires am = 0.")
        if np.linalg.norm(block["an"] - block["an"].T) >= 1e-12:
            raise ValueError("The matrix FX loading an must be symmetric.")
    return blocks


def _two_block_structure(blocks):
    if len(blocks) != 2:
        raise ValueError("Calibration and OU simulation require exactly two blocks.")
    if any(abs(block["beta"] - 0.5) >= 1e-12 for block in blocks):
        raise ValueError("The two-column OU representation requires beta = 0.5.")
    if np.any(blocks[0]["an"] != 0) or np.any(blocks[0]["rho"] != 0):
        raise ValueError("The first block must be the independent rate block: an=rho=0.")
    if np.any(blocks[1]["rn"] != 0) or np.any(blocks[1]["rm"] != 0):
        raise ValueError("The second block must be the FX block: rn=rm=0.")


def matrix_log_transform(block, frequencies, maturities, tolerance=2e-7):
    """Integrate one block's matrix Riccati system at all supplied frequencies.

    The result has shape ``(len(maturities), len(frequencies))``.  This is a
    *log contribution* before yield-curve anchoring, not a normalised CF.
    """
    p = _parameters([block])[0]
    xi = np.asarray(frequencies, dtype=complex).reshape(-1)
    maturity = np.asarray(maturities, dtype=float).reshape(-1)
    if (not xi.size or not maturity.size or not np.all(np.isfinite(xi))
            or not np.all(np.isfinite(maturity)) or np.any(maturity < 0)):
        raise ValueError("Finite frequencies and nonnegative maturities are required.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("ODE tolerance must be positive and finite.")
    nf = xi.size
    if maturity.max() == 0:
        return np.zeros((maturity.size, nf), dtype=complex)
    a = p["an"] - p["am"]
    q = p["sigma"].T @ p["sigma"]
    coupling = p["sigma"].T @ p["rho"] @ a
    c = p["kappa"][None, :, :] - 1j * xi[:, None, None] * coupling
    h = ((xi * xi + 1j * xi)[:, None, None] * (a @ a)
         - 2j * xi[:, None, None] * (p["rn"] - p["rm"]) + 2 * p["rn"])
    h = 0.5 * (h + h.transpose(0, 2, 1))
    ct = c.transpose(0, 2, 1)

    def rhs(_t, y):
        z = y.reshape(nf, 5)
        b = z[:, :4].reshape(nf, 2, 2)
        f = -0.5 * (b @ c + ct @ b - b @ q @ b + h)
        dz = np.empty_like(z)
        dz[:, :4] = f.reshape(nf, 4)
        dz[:, 4] = p["beta"] * (z[:, :4] @ q.ravel())
        return dz.ravel()

    def jacobian(_t, y):
        b = y.reshape(nf, 5)[:, :4].reshape(nf, 2, 2)
        jac = np.zeros((nf, 5, 5), dtype=complex)
        for k in range(4):
            direction = np.zeros((2, 2))
            direction.flat[k] = 1
            derivative = -0.5 * (
                direction @ c + ct @ direction
                - direction @ q @ b - b @ q @ direction
            )
            jac[:, :4, k] = derivative.reshape(nf, 4)
        jac[:, 4, :4] = p["beta"] * q.ravel()
        return bsr_matrix(
            (jac, np.arange(nf), np.arange(nf + 1)), shape=(5 * nf, 5 * nf)
        ).tocsc()

    times = np.unique(np.concatenate(([0.0], maturity)))
    solution = solve_ivp(
        rhs, (0, times[-1]), np.zeros(5 * nf, dtype=complex), method="BDF",
        t_eval=times, rtol=tolerance, atol=tolerance * 0.01, jac=jacobian,
    )
    if not solution.success or solution.t.size != times.size:
        raise ArithmeticError("Riccati integration failed: " + solution.message)
    z = solution.y[:, np.searchsorted(times, maturity)].T.reshape(-1, nf, 5)
    result = z[:, :, 4] + z[:, :, :4] @ p["v0"].ravel()
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("Nonfinite matrix transform.")
    return result


@lru_cache(maxsize=16)
def _quadrature(nodes, cutoff):
    edges = np.unique(np.asarray([0, 1, 4, 12, 35, 100, 250, cutoff], float))
    edges = edges[edges <= cutoff]
    degree = int(np.ceil(nodes / (edges.size - 1)))
    x, weights = leggauss(degree)
    half_width = np.diff(edges) / 2
    centres = (edges[1:] + edges[:-1]) / 2
    return ((centres[:, None] + half_width[:, None] * x).ravel(),
            (half_width[:, None] * weights).ravel())


def _quote_arrays(tau, strikes, theta, ym, yn):
    strikes = np.asarray(strikes, dtype=float)
    shape = strikes.shape
    values = []
    for index, item in enumerate((tau, theta, ym, yn)):
        value = np.asarray(item, dtype=float)
        # Maturities and yields vary down rows, while signs vary across
        # strikes, including the otherwise ambiguous square-grid case.
        if index != 1 and len(shape) == 2 and value.ndim == 1 and value.size == shape[0]:
            value = value[:, None]
        try:
            values.append(np.broadcast_to(value, shape))
        except ValueError as exc:
            raise ValueError("Quote inputs must broadcast to the strike array.") from exc
    time_, sign, domestic, foreign = values
    if (not all(np.all(np.isfinite(a)) for a in (strikes, *values))
            or np.any(strikes <= 0) or np.any(time_ <= 0)
            or np.any(np.abs(sign) != 1)):
        raise ValueError("Quotes require positive strikes/maturities, finite yields, theta +/-1.")
    return time_, strikes, sign, domestic, foreign


def matrix_prices(parameters, tau, strikes, theta, spot, ym, yn, settings=None,
                  return_diagnostics=False):
    """Price calls/puts by the same composite Lewis half-moment inversion."""
    config = _options(settings, {"nodes": 160, "cutoff": 600, "tolerance": 2e-7,
                                 "allowFormal": False})
    nodes = _positive_integer(config["nodes"], "nodes")
    cutoff = float(config["cutoff"])
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be positive and finite.")
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be positive and finite.")
    p = _parameters(parameters) if config["allowFormal"] else validate_matrix_diffusion(parameters)
    time_, strike, sign, ym, yn = _quote_arrays(tau, strikes, theta, ym, yn)
    u, weights = _quadrature(nodes, cutoff)
    unique_time, indices = np.unique(time_.ravel(), return_inverse=True)
    xi = np.concatenate(([0, -1j], u - 0.5j))
    logs = sum(matrix_log_transform(b, xi, unique_time, config["tolerance"]) for b in p)
    if np.max(np.abs(logs[:, :2].imag)) > 1e-6:
        raise ArithmeticError("Complex discount/forward moment.")
    cn, cm = logs[:, 0].real, logs[:, 1].real
    normalized = logs - cn[:, None] + 1j * xi * (cn - cm)[:, None]
    lp = (normalized[indices, 2:]
          + (-yn.ravel()[:, None] + 1j * (u - 0.5j)
             * (yn - ym).ravel()[:, None]) * time_.ravel()[:, None])
    phi = np.exp(lp)
    integrand = (np.exp(-1j * np.log(strike.ravel() / spot)[:, None] * u) * phi).real
    integral = integrand @ (weights / (u * u + 0.25))
    domestic_spot = spot * np.exp(-ym.ravel() * time_.ravel())
    foreign_strike = strike.ravel() * np.exp(-yn.ravel() * time_.ravel())
    calls = domestic_spot - np.sqrt(spot * strike.ravel()) * integral / np.pi
    prices = (calls - (sign.ravel() < 0) * (domestic_spot - foreign_strike)).reshape(strike.shape)
    if not np.all(np.isfinite(prices)):
        raise ArithmeticError("Nonfinite matrix prices.")
    diagnostics = {
        "maxTailTransform": float(np.max(np.abs(phi[:, -1]))),
        "discountMomentError": float(np.max(np.abs(normalized[:, 0]))),
        "forwardMomentError": float(np.max(np.abs(normalized[:, 1]))),
        "quadratureNodes": int(u.size), "cutoff": cutoff,
        "odeTolerance": config["tolerance"],
    }
    return (prices, diagnostics) if return_diagnostics else prices


def _black_prices(spot, strikes, tau, ym, yn, volatility, theta):
    v = np.asarray(volatility) * np.sqrt(tau)
    d1 = (np.log(spot / strikes) + (yn - ym) * tau) / v + v / 2
    d2 = d1 - v
    return theta * (spot * np.exp(-ym * tau) * ndtr(theta * d1)
                    - strikes * np.exp(-yn * tau) * ndtr(theta * d2))


def _implied_volatility(prices, spot, strikes, tau, ym, yn, theta):
    discounted_spot = spot * np.exp(-ym * tau)
    discounted_strikes = strikes * np.exp(-yn * tau)
    lower = np.maximum(theta * (discounted_spot - discounted_strikes), 0)
    upper = np.where(theta > 0, discounted_spot, discounted_strikes)
    valid = np.isfinite(prices) & (prices > lower) & (prices < upper)
    low, high = np.zeros_like(prices), np.full_like(prices, 3.0)
    valid &= prices <= _black_prices(spot, strikes, tau, ym, yn, high, theta)
    for _ in range(40):
        mid = (low + high) / 2
        estimate = _black_prices(spot, strikes, tau, ym, yn, mid, theta)
        low = np.where(estimate < prices, mid, low)
        high = np.where(estimate >= prices, mid, high)
    return np.where(valid, (low + high) / 2, np.nan)


def _market_arrays(market):
    strikes = np.asarray(market["strikes"], float)
    if strikes.ndim != 2 or strikes.shape[1] != 9:
        raise ValueError("The supplied calibration market must contain nine strikes per maturity.")
    theta = np.broadcast_to([-1, -1, -1, -1, -1, 1, 1, 1, 1], strikes.shape)
    return _quote_arrays(np.asarray(market["tau"])[:, None], strikes, theta,
                         np.asarray(market["eurYield"])[:, None],
                         np.asarray(market["usdYield"])[:, None])


def price_matrix(options=None, *, data=None):
    """Price all 135 quotes, with yield anchoring and IV/error diagnostics."""
    opts = _options(options, {"Parameters": None, "Settings": None, "OutputDir": None})
    dataset = load_example() if data is None else data
    p = dataset["parameters"] if opts["Parameters"] is None else opts["Parameters"]
    m = dataset["market"]
    t, k, th, ym, yn = _market_arrays(m)
    settings = FINE_SETTINGS if opts["Settings"] is None else opts["Settings"]
    values, diagnostics = matrix_prices(p, t, k, th, m["spot"], ym, yn, settings, True)
    iv = _implied_volatility(values, m["spot"], k, t, ym, yn, th)
    result = _json_safe({
        "date": m["date"], "parameters": p, "market": m, "prices": values,
        "impliedVolatility": iv, "diagnostics": diagnostics, "settings": settings,
        "method": "Composite Gauss-Legendre Lewis inversion; complex matrix Riccati BDF integration.",
    })
    _write_result(opts["OutputDir"], "prices.json", result)
    return result


def physical_parameters(parameters):
    """Unscaled, full physical entries used by MATLAB's elastic-net penalty."""
    p = _parameters(parameters)
    return np.concatenate([np.asarray(b[name]).ravel(order="F")
                           for b in p for name in (*MATRIX_FIELDS, "beta")])


def encode_parameters(parameters):
    """Encode the exact MATLAB bounds, Cholesky factors and spectral rho map."""
    p = validate_matrix_diffusion(parameters)
    _two_block_structure(p)
    values, scales, lower, upper = [], [], [], []
    for index, block in enumerate(p):
        l = np.linalg.cholesky(block["v0"])
        s = block["sigma"]
        v = [*block["kappa"].ravel(order="F"), s[0, 0], s[0, 1], s[1, 1],
             l[0, 0], l[1, 0], l[1, 1], block["beta"]]
        scale = [1] * 4 + [0.2] * 3 + [0.1] * 3 + [1]
        low = [-50] * 4 + [1e-4, -2, 1e-4, 1e-5, -1, 1e-5, 0.5]
        high = [50] * 4 + [2, 2, 2, 1, 1, 1, 0.5]
        if index == 0:
            v += [block[name][i, j] for name in ("rm", "rn")
                  for i, j in ((0, 0), (0, 1), (1, 1))]
            scale += [1] * 6
            low += [-3] * 6
            high += [3] * 6
        else:
            rr = block["rho"] / 0.995
            spectral_norm = np.linalg.norm(rr, 2)
            if spectral_norm >= 1:
                raise ValueError("Initial FX rho spectral norm must be strictly below 0.995.")
            raw = rr / np.sqrt(1 - spectral_norm**2)
            v += [*raw.ravel(order="F"), block["an"][0, 0], block["an"][0, 1], block["an"][1, 1]]
            scale += [max(1, np.linalg.norm(raw))] * 4 + [1] * 3
            low += [-np.inf] * 4 + [-3] * 3
            high += [np.inf] * 4 + [3] * 3
        values.extend(v)
        scales.extend(scale)
        lower.extend(low)
        upper.extend(high)
    scale = np.asarray(scales)
    x, lb, ub = np.asarray(values) / scale, np.asarray(lower) / scale, np.asarray(upper) / scale
    if np.any(x < lb) or np.any(x > ub):
        raise ValueError("Initial point outside optimiser bounds.")
    if np.max(np.abs(physical_parameters(decode_parameters(x, scale)) - physical_parameters(p))) >= 1e-10:
        raise ValueError("Parameter encoding is not price-preserving; verify block structure and symmetry.")
    return x, lb, ub, scale


def decode_parameters(x, scale):
    """Inverse of :func:`encode_parameters`, retaining the fixed block zeros."""
    value = np.asarray(x) * np.asarray(scale)
    if value.shape != (35,):
        raise ValueError("The two-block calibration requires 35 scaled coordinates.")
    result = []
    for index, offset in enumerate((0, 17)):
        z = value[offset:]
        l = np.array([[z[7], 0], [z[8], z[9]]])
        block = {"kappa": z[:4].reshape(2, 2, order="F"),
                 "sigma": np.array([[z[4], z[5]], [0, z[6]]]),
                 "v0": l @ l.T, "beta": z[10], "am": np.zeros((2, 2))}
        if index == 0:
            block.update(rho=np.zeros((2, 2)), an=np.zeros((2, 2)),
                         rm=np.array([[z[11], z[12]], [z[12], z[13]]]),
                         rn=np.array([[z[14], z[15]], [z[15], z[16]]]))
        else:
            raw = z[11:15].reshape(2, 2, order="F")
            block.update(rho=0.995 * raw / np.sqrt(1 + np.linalg.norm(raw, 2)**2),
                         an=np.array([[z[15], z[16]], [z[16], z[17]]]),
                         rm=np.zeros((2, 2)), rn=np.zeros((2, 2)))
        result.append(block)
    return result


def matrix_constraints(parameters):
    """MATLAB's c <= 0 stability and correlation constraints, block by block."""
    return np.asarray([value for b in _parameters(parameters) for value in
                       (1e-5 - np.trace(b["kappa"]), 1e-6 - np.linalg.det(b["kappa"]),
                        np.linalg.norm(b["rho"], 2) - 0.995)])


def _fit_metrics(parameters, market):
    t, k, th, ym, yn = _market_arrays(market)
    values, vols = [], []
    for settings in ({"nodes": 280, "cutoff": 900, "tolerance": 2e-9}, FINE_SETTINGS):
        prices = matrix_prices(parameters, t, k, th, market["spot"], ym, yn, settings)
        values.append(prices)
        vols.append(_implied_volatility(prices, market["spot"], k, t, ym, yn, th))
    target = np.asarray(market["marketPrice"])
    relative = (values[1] - target) / target
    error = 100 * vols[1] - np.asarray(market["marketIV"])
    physical = physical_parameters(parameters)
    penalty = 1e-4 * (0.5 * np.abs(physical).sum() + 0.5 * (physical * physical).sum())
    metrics = {
        "relativePriceMSE": np.mean(relative**2), "relativePriceRMSE": np.sqrt(np.mean(relative**2)),
        "elasticNetPenalty": penalty, "totalObjective": np.mean(relative**2) + penalty,
        "ivRMSEPP": np.sqrt(np.mean(error**2)), "ivMAEPP": np.mean(np.abs(error)),
        "maxIVErrorPP": np.max(np.abs(error)), "priceRMSE": np.sqrt(np.mean((values[1] - target)**2)),
        "gridPriceChange": np.max(np.abs(values[1] - values[0])),
        "gridIVChangePP": 100 * np.max(np.abs(vols[1] - vols[0])),
    }
    if not np.all(np.isfinite(vols[1])) or metrics["gridIVChangePP"] >= 1e-4:
        raise ArithmeticError("Independent matrix repricing did not converge.")
    return metrics, values[1], vols[1]


def calibrate_matrix(max_iterations=150, options=None, *, data=None):
    """Fit relative option prices plus the physical-parameter elastic net.

    Uses SciPy SLSQP with MATLAB's initial point, 33 free coordinates, bounds,
    nonlinear constraints and objective.  A limited run remains explicitly
    non-converged, and its best feasible parameters can be passed to simulation.
    """
    max_iterations = _positive_integer(max_iterations, "max_iterations")
    opts = _options(options, {"OutputDir": None, "UseParallel": False,
                              "MaxFunctionEvaluations": 16000, "InitialParameters": None})
    budget = _positive_integer(opts["MaxFunctionEvaluations"], "MaxFunctionEvaluations")
    if not isinstance(opts["UseParallel"], (bool, np.bool_)):
        raise ValueError("UseParallel must be boolean.")
    dataset = load_example() if data is None else data
    m = dataset["market"]
    initial = dataset["initialParameters"] if opts["InitialParameters"] is None else opts["InitialParameters"]
    x, lb, ub, scale = encode_parameters(initial)
    if np.max(matrix_constraints(initial)) > 1e-7:
        raise ValueError("Initial parameters violate the nonlinear stability constraints.")
    t, k, th, ym, yn = _market_arrays(m)
    target = np.asarray(m["marketPrice"], dtype=float)
    if target.shape != k.shape or not np.all(np.isfinite(target)) or np.any(target <= 0):
        raise ValueError("Calibration target prices must be finite, positive and match strikes.")
    free = np.flatnonzero(lb != ub)
    best, best_value, evaluations, iterations = x.copy(), np.inf, 0, 0
    history = []
    lock = threading.Lock()
    timer = time.perf_counter()

    class EvaluationLimit(Exception):
        pass

    def expand(z):
        expanded = x.copy()
        expanded[free] = z
        return expanded

    def objective(z):
        nonlocal best, best_value, evaluations
        with lock:
            if evaluations >= budget:
                raise EvaluationLimit()
            evaluations += 1
        full = expand(z)
        q = decode_parameters(full, scale)
        try:
            prices = matrix_prices(q, t, k, th, m["spot"], ym, yn, CALIBRATION_SETTINGS)
            physical = physical_parameters(q)
            value = (np.mean(((prices - target) / target)**2)
                     + 1e-4 * (0.5 * np.abs(physical).sum() + 0.5 * (physical**2).sum()))
            if not np.isfinite(value) or np.any(prices <= 0):
                value = 1e6
        except (ArithmeticError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            value = 1e6
        # Also retain feasible finite-difference trials. They use exactly the
        # same objective and never replace a better point with an infeasible one.
        if np.max(matrix_constraints(q)) <= 1e-7 and np.all(full >= lb) and np.all(full <= ub):
            with lock:
                if value < best_value:
                    best, best_value = full.copy(), float(value)
        return float(value)

    def constraints(z):
        return -matrix_constraints(decode_parameters(expand(z), scale))

    def callback(_z):
        nonlocal iterations
        iterations += 1
        with lock:
            if not history or best_value < history[-1][1]:
                history.append([iterations, best_value, time.perf_counter() - timer])
                checkpoint = {"parameters": decode_parameters(best, scale),
                              "objective": best_value, "history": history}
                _write_result(opts["OutputDir"], "checkpoint.json", checkpoint)

    initial_value = objective(x[free])
    pool_context = ThreadPoolExecutor() if opts["UseParallel"] else nullcontext(None)
    try:
        with pool_context as pool:
            settings = {"maxiter": max_iterations, "ftol": 1e-7,
                        "finite_diff_rel_step": 1e-4, "disp": False}
            if pool is not None:
                settings["workers"] = pool.map
            optimized = minimize(
                objective, x[free], method="SLSQP", jac="2-point",
                bounds=list(zip(lb[free], ub[free])),
                constraints={"type": "ineq", "fun": constraints},
                callback=callback, options=settings,
            )
        output = {"success": bool(optimized.success), "status": int(optimized.status),
                  "message": str(optimized.message), "iterations": int(optimized.nit),
                  "functionEvaluations": evaluations, "gradientEvaluations": int(optimized.njev)}
        exitflag = 1 if optimized.success else (0 if optimized.status == 9 else -1)
    except EvaluationLimit:
        output = {"success": False, "status": 9, "message": "Maximum function evaluations reached.",
                  "iterations": iterations, "functionEvaluations": evaluations}
        exitflag = 0
    p = decode_parameters(best, scale)
    max_constraint = float(np.max(matrix_constraints(p)))
    output.update(algorithm="SciPy SLSQP", useParallel=bool(opts["UseParallel"]),
                  maxConstraintViolation=max(0.0, max_constraint),
                  finiteDifferenceRelativeStep=1e-4, optimalityTolerance=1e-7,
                  note="SLSQP ftol is a combined stopping tolerance; MATLAB fmincon has separate tolerances. Best feasible evaluations are retained.")
    if best_value >= 1e6:
        raise ArithmeticError("No valid feasible matrix prices were obtained during calibration.")
    baseline, _, _ = _fit_metrics(dataset["ivBaselineParameters"], m)
    metrics, prices, iv = _fit_metrics(p, m)
    result = _json_safe({
        "parameters": p, "market": m, "alpha": 1e-4, "lambda": 0.5,
        "objective": best_value, "initialObjective": initial_value,
        "history": history, "seconds": time.perf_counter() - timer,
        "exitflag": exitflag, "output": output, "settings": CALIBRATION_SETTINGS,
        "penaltyConvention": PENALTY_CONVENTION, "baseline": baseline,
        "metrics": metrics, "prices": prices, "impliedVolatility": iv,
        "note": "Relative-price elastic-net fit on January matrix quotes/curves. Monte Carlo has not yet been run for these fitted parameters.",
    })
    _write_result(opts["OutputDir"], "calibration.json", result)
    return result


def _safe_z(error, standard_error):
    error, standard_error = np.broadcast_arrays(error, standard_error)
    result = np.full(error.shape, np.nan)
    np.divide(error, standard_error, out=result, where=standard_error > 0)
    result[(standard_error == 0) & (error == 0)] = 0
    return result


def simulate_matrix(options=None, *, data=None):
    """Exact beta=1/2 OU variance transitions with the same correlated FX noise.

    FX log returns use left-point Euler and rate integrals use the trapezoid
    rule, as in MATLAB.  The 40 independent batches quantify sampling error;
    time-discretisation bias is not included in their standard errors.
    """
    opts = _options(options, {"T": 1, "Steps": 250, "Paths": 20000,
                              "OutputDir": None, "Parameters": None,
                              "Blocks": 40, "Seed": 922761})
    steps = _positive_integer(opts["Steps"], "Steps")
    batches = _positive_integer(opts["Blocks"], "Blocks", 2)
    paths = _positive_integer(opts["Paths"], "Paths", 2 * batches)
    seed = _positive_integer(opts["Seed"], "Seed", 0)
    if seed + batches >= 2**32:
        raise ValueError("Seed + Blocks must fit an unsigned 32-bit MATLAB seed.")
    if paths % batches:
        raise ValueError("Paths must be divisible by Blocks (40 by default).")
    maturity = float(opts["T"])
    if not np.isfinite(maturity) or maturity <= 0:
        raise ValueError("T must be positive and finite.")
    dataset = load_example() if data is None else data
    p = validate_matrix_diffusion(dataset["parameters"] if opts["Parameters"] is None else opts["Parameters"])
    _two_block_structure(p)
    m = dataset["market"]
    rows = np.flatnonzero(np.abs(np.asarray(m["tau"]) - maturity) < 1e-10)
    if not rows.size:
        raise ValueError("T must match one of the supplied market maturities.")
    row = rows[0]
    ym, yn = float(m["eurYield"][row]), float(m["usdYield"][row])
    spot = float(m["spot"])
    strikes = np.asarray(m["strikes"][row])[::2]
    theta = np.asarray([-1, -1, 1, 1, 1])
    xi = np.arange(0, 101, 2)
    logs = sum(matrix_log_transform(b, np.concatenate(([0, -1j], xi)), [maturity], 1e-11)[0] for b in p)
    y0n, y0m = yn + logs[0].real / maturity, ym + logs[1].real / maturity
    cf = np.exp(logs[2:] + (-y0n + 1j * xi * (y0n - y0m)) * maturity)
    reference = matrix_prices(p, maturity, strikes, theta, spot, ym, yn, FINE_SETTINGS)
    dt = maturity / steps
    transitions, expected_v = [], []
    for block in p:
        f, g = -block["kappa"] / 2, block["sigma"].T / 2
        transition = expm(np.block([[f, g @ g.T], [np.zeros((2, 2)), -f.T]]) * dt)
        e = transition[:2, :2]
        covariance = transition[:2, 2:] @ e.T
        covariance = (covariance + covariance.T) / 2
        cross = expm(np.block([[f, g], [np.zeros((2, 4))]]) * dt)[:2, 2:]
        residual = covariance - cross @ cross.T / dt
        residual = (residual + residual.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        if eigenvalues.min() < -1e-10 * max(np.linalg.norm(covariance, 2), np.finfo(float).tiny):
            raise ArithmeticError("OU innovation covariance failure.")
        r = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        independent_cov = np.eye(2) - block["rho"].T @ block["rho"]
        independent_vals, independent_vecs = np.linalg.eigh(independent_cov)
        independent = independent_vecs @ np.diag(np.sqrt(np.maximum(independent_vals, 0))) @ independent_vecs.T
        transitions.append((e, cross / dt, r, np.linalg.cholesky(block["v0"]), independent))
        # A finite-horizon Van Loan covariance also works for singular kappa;
        # it equals MATLAB's stationary Lyapunov expression when that exists.
        full = expm(np.block([[f, g @ g.T], [np.zeros((2, 2)), -f.T]]) * maturity)
        e_t = full[:2, :2]
        cov_t = full[:2, 2:] @ e_t.T
        expected_v.extend((e_t @ block["v0"] @ e_t.T + 2 * cov_t).ravel(order="F"))
    n = paths // batches
    batch_cf = np.empty((batches, xi.size), dtype=complex)
    batch_prices = np.empty((batches, 5))
    batch_moments = np.empty((batches, 3))
    batch_variance = np.empty((batches, 8))
    timer = time.perf_counter()
    for batch in range(batches):
        # MT19937 is explicit and independently seeded per batch; NumPy's
        # normal transform and draw ordering differ from MATLAB's randn.
        rng = np.random.Generator(np.random.MT19937(seed + batch + 1))
        factors = [np.broadcast_to(q[3], (n, 2, 2)).copy() for q in transitions]
        rate_log, fx_log, integral = np.zeros((3, n))
        for _ in range(steps):
            for index, block in enumerate(p):
                e, j, r, _, independent = transitions[index]
                old = factors[index]
                brownian = rng.standard_normal((n, 2, 2)) * np.sqrt(dt)
                new = e @ old + j @ brownian + r @ rng.standard_normal((n, 2, 2))
                factors[index] = new
                if index == 0:
                    variance = 0.5 * (old @ old.transpose(0, 2, 1) + new @ new.transpose(0, 2, 1))
                    rate_log += np.einsum("ij,nji->n", block["rn"] - block["rm"], variance) * dt
                    integral += np.einsum("ij,nji->n", block["rn"], variance) * dt
                else:
                    ax = block["an"] @ old
                    correlated, independent_ax = block["rho"] @ ax, independent @ ax
                    variance = np.sum(ax * ax, axis=(1, 2))
                    price_noise = (np.sum(correlated * brownian, axis=(1, 2))
                                   + np.sqrt(np.sum(independent_ax**2, axis=(1, 2)) * dt)
                                   * rng.standard_normal(n))
                    fx_log += -0.5 * variance * dt + price_noise
        log_return = rate_log + fx_log + (y0n - y0m) * maturity
        discount = np.exp(-integral - y0n * maturity)
        terminal_spot = spot * np.exp(log_return)
        batch_cf[batch] = np.mean(discount[:, None] * np.exp(1j * log_return[:, None] * xi), axis=0)
        batch_prices[batch] = np.mean(discount[:, None] * np.maximum(theta * (terminal_spot[:, None] - strikes), 0), axis=0)
        batch_moments[batch] = [np.mean(discount), np.mean(discount * terminal_spot), np.mean(discount * log_return)]
        for index, factor in enumerate(factors):
            variance_mean = np.mean(factor @ factor.transpose(0, 2, 1), axis=0)
            batch_variance[batch, 4 * index:4 * (index + 1)] = variance_mean.ravel(order="F")
    if not all(np.all(np.isfinite(a)) for a in (batch_cf, batch_prices, batch_moments, batch_variance)):
        raise ArithmeticError("Monte Carlo produced nonfinite observations.")
    ecf, prices, moments, variance_mean = [a.mean(axis=0) for a in (batch_cf, batch_prices, batch_moments, batch_variance)]
    real_se, imag_se, price_se, moment_se, variance_se = [
        a.std(axis=0, ddof=1) / np.sqrt(batches)
        for a in (batch_cf.real, batch_cf.imag, batch_prices, batch_moments, batch_variance)]
    result = _json_safe({
        "date": m["date"], "T": maturity, "steps": steps, "paths": paths,
        "blocks": batches, "seed": seed, "seconds": time.perf_counter() - timer,
        "parameters": p, "xi": xi, "cfReal": cf.real, "cfImag": cf.imag,
        "ecfReal": ecf.real, "ecfImag": ecf.imag, "realSE": real_se, "imagSE": imag_se,
        "strikes": strikes, "theta": theta, "referencePrices": reference,
        "prices": prices, "priceSE": price_se, "discountForwardLogMoments": moments,
        "momentSE": moment_se, "expectedDiscount": np.exp(-yn * maturity),
        "expectedDiscountedSpot": spot * np.exp(-ym * maturity),
        "varianceMean": variance_mean, "varianceSE": variance_se,
        "expectedVarianceMean": expected_v,
        "method": "Exact two-column OU Wishart transitions, correlated log-price Euler, trapezoidal rate integral; no spectral projection.",
        "rng": "NumPy Generator(MT19937), Seed + 1 through Seed + Blocks; not pathwise identical to MATLAB randn.",
        "standardErrorConvention": "Sample standard deviation of independent batch means divided by sqrt(Blocks); excludes discretisation bias.",
        "maxRealZ": np.max(np.abs(_safe_z(ecf.real - cf.real, real_se))),
        "maxImagZ": np.max(np.abs(_safe_z(ecf.imag[1:] - cf.imag[1:], imag_se[1:]))),
        "priceZ": _safe_z(prices - reference, price_se),
        "varianceZ": _safe_z(variance_mean - expected_v, variance_se),
    })
    _write_result(opts["OutputDir"], "simulation.json", result)
    return result
