"""NumPy/SciPy counterpart of the public MATLAB independent-CIR FX model.

Parameters are rows ``[kappa, vbar, sigma, rho, v0, am, an, rm, rn]``.
Both the discount and discounted-spot moments match the supplied yield curves.
Calibration uses SciPy SLSQP with MATLAB's physical objective, bounds and scaled
variables; its stopping rules and search trajectory differ from fmincon SQP.
Simulation uses MT19937 with independent ``Seed + block`` streams. NumPy and
MATLAB transform random numbers differently, so paths are not bitwise identical.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if not np.isfinite(number) or number < minimum or number != int(number):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(number)


def _factor_count(factors):
    factors = _integer(factors, "factors")
    if factors not in (1, 2):
        raise ValueError("The examples support one or two factors.")
    return factors


def load_example(factors=1):
    """Return a fresh public vector example, with vols expressed as decimals."""
    factors = _factor_count(factors)
    return json.loads((ROOT / "data" / f"vector_N{factors}.json").read_text())


def _parameters(parameters, factors=None):
    # MATLAB jsonencode collapses an N=1 parameter row to a flat JSON array.
    p = np.asarray(parameters, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.ndim != 2 or p.shape[1] != 9 or p.shape[0] == 0:
        raise ValueError("parameters must have shape (N, 9).")
    if factors is not None and p.shape[0] != factors:
        raise ValueError("Parameter factor count does not match the example.")
    return p.copy()


def validate_vector_parameters(parameters, horizon):
    """Reject invalid CIR parameters and exploding required discounted moments."""
    p = _parameters(parameters)
    if not np.isscalar(horizon) or not np.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and nonnegative.")
    if not np.all(np.isfinite(p)):
        raise ValueError("All parameters must be finite.")
    if np.any((p[:, 0] <= 0) | (p[:, 1] < 0) | (p[:, 2] <= 0)
              | (abs(p[:, 3]) > 1) | (p[:, 4] < 0)):
        raise ValueError("Invalid CIR parameters or correlation.")
    if np.any(p[:, 5] != 0):
        raise ValueError("Only the am=0 model is supported.")
    for k, _, sigma, rho, _, _, a, rm, rn in p:
        for moment in (0, .5, 1):
            c = k - moment * rho * sigma * a
            b = (-moment**2 + moment) * a*a - 2*moment*(rn-rm) + 2*rn
            delta = c*c + sigma*sigma*b
            explosion = math.inf
            if b < 0:
                if delta < 0:
                    d = math.sqrt(-delta)
                    explosion = 2 * (math.pi/2 + math.atan(c/d)) / d
                elif c < 0:
                    if delta == 0:
                        explosion = -2/c
                    else:
                        d = math.sqrt(delta)
                        explosion = math.log((c-d)/(c+d))/d
            if horizon >= explosion:
                raise ValueError("Required discounted moment explodes before the horizon.")
    return p


def _raw_transform(p, u, tau):
    u = np.asarray(u, dtype=complex)
    shape = u.shape
    u = u.reshape(1, -1)
    k, vb, sigma, rho, v0, am, an, rm, rn = p.T[:, :, None]
    c = k - 1j*u*rho*sigma*(an-am)
    h = (an-am)**2*u*u + 1j*u*(an*an-am*am) - 2j*u*(rn-rm) + 2*rn
    d = np.sqrt(c*c + sigma*sigma*h)
    # Algebraically equivalent to the MATLAB little-trap formula. expm1 and
    # the removable d=0 limit also cover a deterministic zero factor safely.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        one_minus_e = -np.expm1(-d*tau)
        denominator = 2*d + (c-d)*one_minus_e
        aa = (c-d)*tau + 2*np.log(2*d/denominator)
        bb = -h*one_minus_e/denominator
        zero = abs(d) < 1e-12
        aa = np.where(zero, c*tau - 2*np.log1p(c*tau/2), aa)
        bb = np.where(zero, -h*tau/(2+c*tau), bb)
        term = (k*vb/(sigma*sigma))*aa + v0*bb
        term = np.where((vb == 0) & (v0 == 0), 0, term)
        out = np.sum(term, axis=0).reshape(shape)
    if not np.all(np.isfinite(out)):
        raise ValueError("The vector transform is not finite on this contour.")
    return out


def vector_log_transform(parameters, u, tau):
    """Unshifted discounted log transform at scalar or array Fourier arguments."""
    p = validate_vector_parameters(parameters, tau)
    if not np.all(np.isfinite(u)):
        raise ValueError("Fourier arguments must be finite.")
    return _raw_transform(p, u, tau)


def _surface_inputs(tau, strikes, theta, spot, ym, yn):
    k = np.asarray(strikes, dtype=float)
    values = []
    for value in (tau, theta, ym, yn):
        arr = np.asarray(value, dtype=float)
        values.append(arr)
    # MATLAB implicit expansion: maturity and yields run down the rows,
    # while option signs run across the strike columns.
    if k.ndim == 2:
        for i in (0, 2, 3):
            if values[i].ndim == 1 and values[i].size == k.shape[0]:
                values[i] = values[i][:, None]
    try:
        t, k, signs, y_m, y_n = np.broadcast_arrays(values[0], k, values[1], values[2], values[3])
    except ValueError as exc:
        raise ValueError("Maturities, strikes, signs and yields do not broadcast.") from exc
    if not np.isscalar(spot) or not np.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be finite and positive.")
    if k.size == 0 or any(not np.all(np.isfinite(a)) for a in (t, k, signs, y_m, y_n)):
        raise ValueError("Pricing inputs must be finite and nonempty.")
    if np.any(t <= 0) or np.any(k <= 0) or np.any(abs(signs) != 1):
        raise ValueError("Maturities and strikes must be positive; signs must be +/-1.")
    return t, k, signs, y_m, y_n


class _SurfacePricer:
    """Cache quote-dependent Fourier kernels across calibration evaluations."""
    def __init__(self, tau, strikes, theta, spot, ym, yn, step, cutoff):
        self.t, self.k, self.signs, self.ym, self.yn = _surface_inputs(tau, strikes, theta, spot, ym, yn)
        if not np.isscalar(step) or not np.isscalar(cutoff) or not np.isfinite(step) or not np.isfinite(cutoff) or step <= 0 or cutoff <= 0:
            raise ValueError("Quadrature step and cutoff must be finite and positive.")
        intervals = round(cutoff/step)
        if intervals < 2 or intervals % 2 or not math.isclose(intervals*step, cutoff, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Simpson quadrature needs an even integral cutoff/step.")
        self.u = np.arange(intervals+1)*step
        self.z = self.u - .5j
        weights = np.full(intervals+1, 2.)
        weights[1:-1:2] = 4
        weights[[0, -1]] = 1
        weights *= (step/3)/(self.u*self.u+.25)
        self.base = np.where(self.signs > 0, spot*np.exp(-self.ym*self.t), self.k*np.exp(-self.yn*self.t)).ravel()
        self.root = np.sqrt(spot*self.k).ravel()/np.pi
        self.groups = []
        for maturity in np.unique(self.t):
            ix = np.flatnonzero(self.t.ravel() == maturity)
            ym_i, yn_i = self.ym.ravel()[ix, None], self.yn.ravel()[ix, None]
            kernel = np.exp(-1j*np.log(self.k.ravel()[ix, None]/spot)*self.u
                            + (-yn_i+1j*self.z*(yn_i-ym_i))*maturity)*weights
            self.groups.append((float(maturity), ix, kernel))

    def __call__(self, parameters):
        p = validate_vector_parameters(parameters, float(np.max(self.t)))
        result = self.base.copy()
        for maturity, ix, kernel in self.groups:
            raw = _raw_transform(p, np.concatenate(([0, -1j], self.z)), maturity)
            matched = np.exp(raw[2:] - raw[0].real + 1j*self.z*(raw[0].real-raw[1].real))
            result[ix] -= self.root[ix]*np.real(kernel @ matched)
        if not np.all(np.isfinite(result)):
            raise ValueError("Nonfinite option prices.")
        return result.reshape(self.k.shape)


def vector_prices(parameters, tau, strikes, theta, spot, ym, yn, *, step=.05, cutoff=900):
    """Curve-matched half-moment prices, as an ndarray with broadcast input shape."""
    return _SurfacePricer(tau, strikes, theta, spot, ym, yn, step, cutoff)(parameters)


def _implied_vol(prices, spot, k, t, ym, yn, signs):
    t, k, signs, ym, yn = _surface_inputs(t, k, signs, spot, ym, yn)
    prices = np.broadcast_to(np.asarray(prices, dtype=float), k.shape)
    a, b = spot*np.exp(-ym*t), k*np.exp(-yn*t)
    lower, upper = np.maximum(signs*(a-b), 0), np.where(signs > 0, a, b)
    if np.any(~np.isfinite(prices) | (prices <= lower) | (prices >= upper)):
        raise ValueError("Prices outside strict Black-Scholes bounds.")
    lo, hi = np.zeros(k.shape), np.full(k.shape, 3.)
    def bs(vol):
        root = vol*np.sqrt(t)
        d1 = (np.log(spot/k) + (yn-ym)*t)/root + root/2
        return signs*(a*ndtr(signs*d1) - b*ndtr(signs*(d1-root)))
    if np.any(prices > bs(hi)):
        raise ValueError("Implied volatility exceeds the MATLAB inversion bracket [0, 3].")
    for _ in range(40):
        mid = (lo+hi)/2
        below = bs(mid) < prices
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return (lo+hi)/2


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_result(directory, filename, result):
    if directory is not None:
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / filename).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")


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


def _example_data(factors, data):
    factors = _factor_count(factors)
    d = load_example(factors) if data is None else data
    if not isinstance(d, dict):
        raise ValueError("data must be a loaded example dictionary.")
    p = _parameters(d["parameters"], factors)
    t, k, signs, ym, yn = _surface_inputs(d["tau"], d["strikes"], d["theta"], d["spot"], d["eurYield"], d["usdYield"])
    if k.ndim != 2:
        raise ValueError("Example strikes must be a maturity-by-quote matrix.")
    validate_vector_parameters(p, float(np.max(t)))
    return factors, d, p


def price_vector(factors=1, maturity=1, parameters=None, *, data=None, step=.025, cutoff=1800):
    """JSON-safe prices and IVs for one maturity, or the surface if maturity=None."""
    factors, d, saved = _example_data(factors, data)
    p = saved if parameters is None else _parameters(parameters, factors)
    t, k, ym, yn = (np.asarray(d[key], dtype=float) for key in ("tau", "strikes", "eurYield", "usdYield"))
    if maturity is not None:
        rows = np.flatnonzero(np.isclose(t, maturity, rtol=0, atol=1e-10))
        if rows.size == 0:
            raise ValueError("Choose a maturity present in the example.")
        row = rows[0]
        t, k, ym, yn = t[row], k[row], ym[row], yn[row]
    prices = vector_prices(p, t, k, d["theta"], d["spot"], ym, yn, step=step, cutoff=cutoff)
    iv = _implied_vol(prices, d["spot"], k, t, ym, yn, d["theta"])
    return _json_safe({"N": factors, "maturity": maturity, "tau": t, "parameters": p,
                       "strikes": k, "theta": d["theta"], "spot": d["spot"], "eurYield": ym,
                       "usdYield": yn, "prices": prices, "impliedVolatilityDecimal": iv,
                       "grid": {"step": step, "cutoff": cutoff}})


class _EvaluationLimit(Exception):
    pass


def calibrate_vector(factors=1, max_iterations=150, options=None, *, data=None):
    """Bounded physical-parameter elastic-net calibration on the unique quotes.

    Options: InitialParameters, Alpha, Lambda, MaxFunctionEvaluations,
    OutputDir (None disables writing), Display. Snake-case aliases are accepted.
    """
    factors, d, saved = _example_data(factors, data)
    max_iterations = _integer(max_iterations, "max_iterations")
    o = _options(options, {"InitialParameters": saved, "Alpha": 1e-4, "Lambda": .5,
                          "MaxFunctionEvaluations": 16000, "OutputDir": ROOT/"results"/f"vector_N{factors}", "Display": False})
    budget = _integer(o["MaxFunctionEvaluations"], "MaxFunctionEvaluations")
    alpha, lam = float(o["Alpha"]), float(o["Lambda"])
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("Alpha must be nonnegative and Lambda must be in [0, 1].")
    initial = _parameters(o["InitialParameters"], factors)
    validate_vector_parameters(initial, max(d["tau"]))
    market, market_iv = np.asarray(d["marketPrices"], float), np.asarray(d["marketIVDecimal"], float)
    shape = np.shape(d["strikes"])
    if market.shape != shape or market_iv.shape != shape or np.any(~np.isfinite(market)) or np.any(market <= 0) or np.any(~np.isfinite(market_iv)) or np.any(market_iv <= 0):
        raise ValueError("Market prices and decimal IVs must be finite, positive and match strikes.")
    maxvar = float(np.max(market_iv**2))
    lo = np.tile([.01, 0, .01, -1, 0, 0, 0, -1, -1], (factors, 1))
    hi = np.tile([2, maxvar, 1, 1, maxvar, 0, 2, 1, 1], (factors, 1))
    lo[0, 6] = hi[0, 6] = 1
    if np.any(initial < lo) or np.any(initial > hi):
        raise ValueError("Initial parameters are outside the stated bounds.")
    scales = np.ones_like(initial)
    scales[:, [1, 4]] = .01
    free = np.flatnonzero((lo < hi).ravel(order="F"))
    scale = scales.ravel(order="F")[free]
    x0 = initial.ravel(order="F")[free]/scale
    lb, ub = lo.ravel(order="F")[free]/scale, hi.ravel(order="F")[free]/scale
    pricer = _SurfacePricer(d["tau"], d["strikes"], d["theta"], d["spot"], d["eurYield"], d["usdYield"], .05, 900)
    def decode(x):
        flat = initial.ravel(order="F").copy()
        flat[free] = x*scale
        return flat.reshape(initial.shape, order="F")
    calls = 0
    iterations = 0
    cached_x, cached_value = None, None
    best_x, best_value = x0.copy(), math.inf
    def objective(x):
        nonlocal calls, cached_x, cached_value, best_x, best_value
        if cached_x is not None and np.array_equal(x, cached_x):
            return cached_value
        if calls >= budget:
            raise _EvaluationLimit
        calls += 1
        try:
            p = decode(x)
            prices = pricer(p)
            value = float(np.mean(((prices-market)/market)**2) + alpha*(lam*np.sum(abs(p))+(1-lam)*np.sum(p*p)))
            if np.any(prices <= 0) or not np.isfinite(value):
                value = 1e6
        except (ValueError, FloatingPointError, OverflowError):
            value = 1e6
        cached_x, cached_value = np.array(x, copy=True), value
        if value < best_value:
            best_x, best_value = np.array(x, copy=True), value
        return value
    def jacobian(x):
        base = objective(x)
        derivative = np.empty_like(x)
        for j in range(x.size):
            h = 1e-4*max(1., abs(x[j]))
            h = min(h, ub[j]-x[j]) if x[j]+h <= ub[j] else -min(h, x[j]-lb[j])
            z = np.array(x, copy=True)
            z[j] += h
            derivative[j] = (objective(z)-base)/h
        return derivative
    def callback(_):
        nonlocal iterations
        iterations += 1
    initial_value = objective(x0)
    if initial_value >= 1e6:
        raise ValueError("Initial parameters produce invalid prices.")
    start = time.perf_counter()
    try:
        fit = minimize(objective, x0, method="SLSQP", jac=jacobian,
                       bounds=list(zip(lb, ub)), callback=callback,
                       options={"ftol": 1e-7, "maxiter": max_iterations, "disp": bool(o["Display"])})
        fitted_x, value = fit.x, float(fit.fun)
        exitflag = 1 if fit.success else (0 if fit.status == 9 else -1)
        output = {"algorithm": "SciPy SLSQP", "status": int(fit.status), "message": str(fit.message), "iterations": int(fit.nit)}
    except _EvaluationLimit:
        fitted_x, value, exitflag = best_x, best_value, 0
        output = {"algorithm": "SciPy SLSQP", "status": None, "message": "Maximum function evaluations reached; returning best evaluated parameters.", "iterations": iterations}
    if not np.isfinite(value) or value > initial_value:
        fitted_x, value = x0, initial_value
        output["returnedInitialParameters"] = True
    output["functionEvaluations"] = calls
    fitted = decode(fitted_x)
    coarse = pricer(fitted)
    fine = vector_prices(fitted, d["tau"], d["strikes"], d["theta"], d["spot"], d["eurYield"], d["usdYield"], step=.025, cutoff=1800)
    grid_change = float(np.max(abs(fine-coarse)))
    if grid_change >= 1e-7:
        raise ValueError("Final price grid did not converge within 1e-7.")
    iv = _implied_vol(fine, d["spot"], d["strikes"], d["tau"], d["eurYield"], d["usdYield"], d["theta"])
    result = _json_safe({"N": factors, "parameters": fitted, "initialParameters": initial,
        "initialObjective": initial_value, "objective": value, "alpha": alpha, "lambda": lam,
        "exitflag": exitflag, "output": output, "seconds": time.perf_counter()-start,
        "settings": {"step": .05, "cutoff": 900, "finiteDifferenceStepSize": 1e-4,
                     "ftol": 1e-7, "maxIterations": max_iterations, "maxFunctionEvaluations": budget},
        "gridPriceChange": grid_change, "prices": fine, "impliedVolatilityDecimal": iv,
        "priceRMSE": np.sqrt(np.mean((fine-market)**2)), "ivRMSEDecimal": np.sqrt(np.mean((iv-market_iv)**2)),
        "note": "New bounded local optimisation on unique quotes. SLSQP and MATLAB fmincon SQP use different search and stopping rules; this is not the historical optimiser trajectory."})
    _write_result(o["OutputDir"], "calibration.json", result)
    return result


def _qe_step(v, k, b, sigma, dt, rng):
    e = np.exp(-k*dt)
    one_minus_e = -np.expm1(-k*dt)
    m = b+(v-b)*e
    s2 = v*sigma**2*e/k*one_minus_e + b*sigma**2/(2*k)*one_minus_e**2
    psi = s2/np.maximum(m*m, np.finfo(float).tiny)
    uniform = rng.random(v.shape)
    result = np.zeros_like(v)
    deterministic = psi == 0
    result[deterministic] = m[deterministic]
    low = (psi <= 1.5) & ~deterministic
    inv = 2/psi[low]
    b2 = inv-1 + np.sqrt(inv)*np.sqrt(inv-1)
    z = ndtri(np.clip(uniform[low], np.finfo(float).tiny, 1-np.finfo(float).eps))
    result[low] = m[low]/(1+b2)*(np.sqrt(b2)+z)**2
    high = psi > 1.5
    prob = (psi[high]-1)/(psi[high]+1)
    u = uniform[high]
    values = np.zeros_like(u)
    pos = u > prob
    values[pos] = np.log((1-prob[pos])/(1-u[pos]))*m[high][pos]/(1-prob[pos])
    result[high] = values
    return result


def simulate_vector(factors=1, options=None, *, data=None):
    """Batched full-truncation Euler (EM) or quadratic-exponential (QE) paths.

    Options: Parameters, T, Steps, Paths, Blocks, Scheme, Seed, OutputDir.
    Returns JSON-safe prices, sampling SEs, characteristic functions, moments,
    boundary counts and EM direct-spot diagnostics. QE direct-spot fields are
    null because that diagnostic exists only for Euler. Batch SEs use ddof=1.
    OutputDir=None disables JSON/NPZ writes.
    """
    factors, d, saved = _example_data(factors, data)
    o = _options(options, {"Parameters": saved, "T": 1., "Steps": 250, "Paths": 20000,
        "Blocks": 40, "Scheme": "QE", "Seed": 81473+10000*factors,
        "OutputDir": ROOT/"results"/f"vector_N{factors}"})
    steps, paths, blocks = (_integer(o[key], key, 2 if key == "Blocks" else 1) for key in ("Steps", "Paths", "Blocks"))
    seed = _integer(o["Seed"], "Seed", 0)
    if seed+blocks >= 2**32:
        raise ValueError("Seed + Blocks must fit an unsigned 32-bit MATLAB seed.")
    if paths % blocks:
        raise ValueError("Paths must be divisible by Blocks.")
    scheme = str(o["Scheme"]).upper()
    if scheme not in ("QE", "EM"):
        raise ValueError("Scheme must be EM or QE.")
    tau = float(o["T"])
    p = validate_vector_parameters(_parameters(o["Parameters"], factors), tau)
    rows = np.flatnonzero(np.isclose(d["tau"], tau, rtol=0, atol=1e-10))
    if tau <= 0 or rows.size == 0:
        raise ValueError("Choose a positive maturity present in the example.")
    row = rows[0]
    columns = [0, 2, 5, 8, 10]
    if np.shape(d["strikes"])[1] < 11:
        raise ValueError("Simulation examples require the eleven-quote smile layout.")
    strikes = np.asarray(d["strikes"], float)[row, columns]
    signs = np.asarray(d["theta"], float)[columns]
    spot, ym, yn = float(d["spot"]), float(d["eurYield"][row]), float(d["usdYield"][row])
    reference = vector_prices(p, tau, strikes, signs, spot, ym, yn, step=.025, cutoff=1800)
    xi = np.arange(0, 101, 2, dtype=float)
    raw = _raw_transform(p, np.concatenate(([0, -1j], xi)), tau)
    y0n, y0m = yn+raw[0].real/tau, ym+raw[1].real/tau
    cf = np.exp(raw[2:]+(-y0n+1j*xi*(y0n-y0m))*tau)
    k, b, sigma, rho, v0, am, an, rm, rn = p.T
    a = an-am
    n, dt = paths//blocks, tau/steps
    bp, direct_bp = np.zeros((blocks, strikes.size)), np.full((blocks, strikes.size), np.nan)
    bcf = np.zeros((blocks, xi.size), complex)
    direct_bcf = np.full((blocks, xi.size), complex(np.nan, np.nan))
    diagnostics = np.zeros((blocks, 7))
    terminal_var, boundary = np.zeros((blocks, factors)), np.zeros((blocks, factors))
    start = time.perf_counter()
    for block in range(blocks):
        rng = np.random.Generator(np.random.MT19937(seed+block+1))
        x, integ, direct_spot = np.zeros(n), np.zeros(n), np.full(n, spot)
        v = np.broadcast_to(v0, (n, factors)).copy()
        bad_spot, count = 0, np.zeros(factors)
        for _ in range(steps):
            vp = np.maximum(v, 0)
            if scheme == "QE":
                vn = _qe_step(vp, k, b, sigma, dt, rng)
                av = (vp+vn)/2
                dw = ((vn-vp-(b-av)*k*dt)*(rho/sigma)
                      + np.sqrt(av*dt)*np.sqrt(1-rho*rho)*rng.standard_normal((n, factors))) @ a
            else:
                z = rng.standard_normal((n, factors))
                w = z*rho + rng.standard_normal((n, factors))*np.sqrt(1-rho*rho)
                vn = v+(b-vp)*k*dt + np.sqrt(vp*dt)*sigma*z
                count += np.sum(vn < 0, axis=0)
                av = vp
                dw = (np.sqrt(vp*dt)*w) @ a
            mu = y0n-y0m + av @ (rn-rm)
            x += mu*dt - .5*(av @ (a*a))*dt + dw
            integ += (y0n + av @ rn)*dt
            if scheme == "EM":
                multiplier = 1+mu*dt+dw
                bad_spot += np.count_nonzero(multiplier <= 0)
                direct_spot *= multiplier
            v = vn
        discount, terminal = np.exp(-integ), spot*np.exp(x)
        if not np.all(np.isfinite(discount)) or not np.all(np.isfinite(terminal)):
            raise ValueError("Nonfinite simulated paths; check parameters and time resolution.")
        bp[block] = np.mean(discount[:, None]*np.maximum(signs*(terminal[:, None]-strikes), 0), axis=0)
        bcf[block] = np.mean(discount[:, None]*np.exp(1j*x[:, None]*xi), axis=0)
        diagnostics[block] = [np.mean(discount), np.mean(discount*terminal), np.nan, np.nan,
                              np.mean(discount**2), np.mean(terminal), np.mean(x)]
        if scheme == "EM":
            direct_bp[block] = np.mean(discount[:, None]*np.maximum(signs*(direct_spot[:, None]-strikes), 0), axis=0)
            if np.all(direct_spot > 0):
                direct_bcf[block] = np.mean(discount[:, None]*np.exp(1j*np.log(direct_spot[:, None]/spot)*xi), axis=0)
            diagnostics[block, 2:4] = [bad_spot/(n*steps), np.mean(abs(terminal-direct_spot))]
        terminal_var[block] = np.mean(np.maximum(v, 0), axis=0)
        boundary[block] = count/(n*steps)
    def mean(a):
        return np.mean(a, axis=0)
    def se(a):
        return np.std(a, axis=0, ddof=1)/np.sqrt(blocks)
    price, price_se, ecf = mean(bp), se(bp), mean(bcf)
    diag, diag_se = mean(diagnostics), se(diagnostics)
    direct_cf = mean(direct_bcf)
    settings = {"paths": paths, "blocks": blocks, "steps": steps, "scheme": scheme, "seed": seed}
    result = _json_safe({"N": factors, "parameters": p, "T": tau, "settings": settings,
        "strikes": strikes, "theta": signs, "referencePrices": reference, "prices": price, "priceSE": price_se,
        "ci95": [price-1.96*price_se, price+1.96*price_se], "xi": xi,
        "cfReal": cf.real, "cfImag": cf.imag, "ecfReal": ecf.real, "ecfImag": ecf.imag,
        "cfRealSE": se(bcf.real), "cfImagSE": se(bcf.imag), "ecfSE": se(bcf),
        "expectedDiscount": np.exp(-yn*tau), "expectedDiscountedSpot": spot*np.exp(-ym*tau),
        "discountMean": diag[0], "discountedSpotMean": diag[1],
        "discountSE": diag_se[0], "discountedSpotSE": diag_se[1], "diagnostics": diag, "diagnosticsSE": diag_se,
        "diagnosticNames": ["discount", "discountedSpot", "nonpositiveEulerSpotFraction", "absoluteLogVsDirectSpotDifference", "squaredDiscount", "spot", "logReturn"],
        "directPrice": mean(direct_bp), "directSE": se(direct_bp), "directECFReal": direct_cf.real, "directECFImag": direct_cf.imag,
        "pairedDifference": mean(direct_bp-bp), "pairedDifferenceSE": se(direct_bp-bp),
        "beta": k*b/sigma**2, "boundaryFraction": mean(boundary),
        "terminalVariance": mean(terminal_var), "terminalVarianceSE": se(terminal_var),
        "blockPrices": bp, "blockDirectPrices": direct_bp, "blockECFReal": bcf.real,
        "blockECFImag": bcf.imag, "blockDiagnostics": diagnostics, "blockTerminalVariance": terminal_var,
        "seconds": time.perf_counter()-start,
        "note": "Batch SEs measure sampling error, not time-step bias. NumPy and MATLAB random paths differ even with the same seed."})
    _write_result(o["OutputDir"], f"simulation_{scheme}.json", result)
    if o["OutputDir"] is not None:
        np.savez_compressed(Path(o["OutputDir"])/f"simulation_{scheme}.npz", blockPrices=bp,
            blockDirectPrices=direct_bp, blockECF=bcf, blockDirectECF=direct_bcf,
            blockDiagnostics=diagnostics, blockTerminalVariance=terminal_var, blockBoundaryFraction=boundary,
            parameters=p, cf=cf, xi=xi, referencePrices=reference)
    return result
