"""Compare Python with a fresh MATLAB export (see tests/exportLanguageParity.m)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from matrix_model import (load_example as load_matrix, matrix_log_transform,
                          matrix_prices, physical_parameters, simulate_matrix)
from vector_model import load_example as load_vector, vector_log_transform, vector_prices, simulate_vector
from vector_reference import log_transform

ROOT = Path(__file__).resolve().parents[1]


def check_close(name, actual, expected, tolerance, results):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.size != expected.size:
        raise AssertionError(f"{name}: different array sizes {actual.shape} and {expected.shape}")
    error = float(np.max(np.abs(actual.ravel() - expected.ravel())))
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f"{name}: maximum error {error:g} exceeds {tolerance:g}")
    results[name] = {"maxAbsoluteDifference": error, "tolerance": tolerance}


def check_samples(name, actual, actual_se, expected, expected_se, results):
    actual, expected, actual_se, expected_se = [np.asarray(x, dtype=float).ravel()
                                               for x in (actual, expected, actual_se, expected_se)]
    combined = np.hypot(actual_se, expected_se)
    error = np.abs(actual - expected)
    if not np.all(np.isfinite(error)) or np.any(error > 6 * combined + 1e-11):
        raise AssertionError(f"{name}: Monte Carlo difference exceeds six combined batch standard errors")
    z = np.divide(error, combined, out=np.zeros_like(error), where=combined > 1e-13)
    results[name] = {"maxCombinedStandardErrors": float(z.max()), "limit": 6}


def verify(fixture: Path, simulations=True):
    saved = json.loads(fixture.read_text(encoding="utf-8-sig"))
    results = {"matlabRuntime": saved["runtime"]}
    times = saved["maturities"]
    frequencies = np.asarray(saved["frequenciesReal"]) + 1j * np.asarray(saved["frequenciesImag"])
    for factors in (1, 2):
        name = f"vector_N{factors}"
        data, reference = load_vector(factors), saved[name]
        p = np.asarray(data["parameters"]).reshape(factors, 9)
        tau, ym, yn = [np.asarray(data[key])[:, None] for key in ("tau", "eurYield", "usdYield")]
        fine = vector_prices(p, tau, data["strikes"], data["theta"], data["spot"], ym, yn,
                             step=.025, cutoff=1800)
        check_close(name + ".prices", fine, reference["prices"], 3e-12, results)
        logs = np.asarray([vector_log_transform(p, frequencies, t) for t in times])
        independent = np.asarray([[log_transform(p, u, t) for u in frequencies] for t in times])
        check_close(name + ".independentLogReal", logs.real, independent.real, 3e-12, results)
        check_close(name + ".independentLogImag", logs.imag, independent.imag, 3e-12, results)
        check_close(name + ".logReal", logs.real, reference["logTransformReal"], 3e-12, results)
        check_close(name + ".logImag", logs.imag, reference["logTransformImag"], 3e-12, results)
        coarse = vector_prices(p, tau, data["strikes"], data["theta"], data["spot"], ym, yn)
        objective = np.mean(((coarse - data["marketPrices"]) / data["marketPrices"])**2)
        objective += 1e-4 * (.5 * np.abs(p).sum() + .5 * (p*p).sum())
        check_close(name + ".initialObjective", objective, reference["initialObjective"], 1e-11, results)
        if simulations:
            for scheme in ("EM", "QE"):
                result = simulate_vector(factors, {"Scheme": scheme, "OutputDir": None})
                target = reference[scheme]
                label = name + "." + scheme
                for field in ("referencePrices", "cfReal", "cfImag"):
                    check_close(label + "." + field, result[field], target[field], 3e-12, results)
                for field, se in (("prices", "priceSE"), ("ecfReal", "cfRealSE"), ("ecfImag", "cfImagSE"),
                                  ("terminalVariance", "terminalVarianceSE")):
                    check_samples(label + "." + field, result[field], result[se], target[field], target[se], results)
                check_samples(label + ".discountForward", result["diagnostics"][:2], result["diagnosticsSE"][:2],
                              target["diagnostics"][:2], target["diagnosticsSE"][:2], results)

    data, reference = load_matrix(), saved["matrix"]
    market = data["market"]
    tau, ym, yn = [np.asarray(market[key])[:, None] for key in ("tau", "eurYield", "usdYield")]
    theta = [-1, -1, -1, -1, -1, 1, 1, 1, 1]
    fine = matrix_prices(data["parameters"], tau, market["strikes"], theta, market["spot"], ym, yn,
                         {"nodes": 560, "cutoff": 1200, "tolerance": 1e-10})
    check_close("matrix.prices", fine, reference["prices"], 2e-8, results)
    logs = sum(matrix_log_transform(p, frequencies, times, 1e-10) for p in data["parameters"])
    check_close("matrix.logReal", logs.real, reference["logTransformReal"], 2e-8, results)
    check_close("matrix.logImag", logs.imag, reference["logTransformImag"], 2e-8, results)
    coarse = matrix_prices(data["initialParameters"], tau, market["strikes"], theta, market["spot"], ym, yn,
                           {"nodes": 210, "cutoff": 900, "tolerance": 1e-10})
    physical = physical_parameters(data["initialParameters"])
    objective = np.mean(((coarse - market["marketPrice"]) / market["marketPrice"])**2)
    objective += 1e-4 * (.5 * np.abs(physical).sum() + .5 * (physical*physical).sum())
    check_close("matrix.initialObjective", objective, reference["initialObjective"], 2e-8, results)
    if simulations:
        result, target = simulate_matrix({"OutputDir": None}), reference["simulation"]
        for field in ("referencePrices", "cfReal", "cfImag", "expectedVarianceMean", "expectedDiscount", "expectedDiscountedSpot"):
            check_close("matrix.simulation." + field, result[field], target[field], 2e-8, results)
        for field, se in (("prices", "priceSE"), ("ecfReal", "realSE"), ("ecfImag", "imagSE"),
                          ("varianceMean", "varianceSE"), ("discountForwardLogMoments", "momentSE")):
            check_samples("matrix.simulation." + field, result[field], result[se], target[field], target[se], results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", type=Path, default=ROOT / "results/language_parity/matlab.json")
    parser.add_argument("--no-simulations", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/language_parity/comparison.json")
    args = parser.parse_args()
    results = verify(args.matlab, not args.no_simulations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Passed {len(results)-1} MATLAB/Python comparisons; details: {args.output}")


if __name__ == "__main__":
    main()
