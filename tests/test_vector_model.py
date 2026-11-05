"""Deterministic MATLAB parity and statistical checks for the vector engine."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from vector_model import (calibrate_vector, load_example, price_vector,
                          simulate_vector, validate_vector_parameters,
                          vector_log_transform, vector_prices)
from vector_reference import log_transform


class VectorNumericalChecks(unittest.TestCase):
    def test_saved_prices_both_factors(self):
        for factors in (1, 2):
            with self.subTest(factors=factors):
                d = load_example(factors)
                report = price_vector(factors)
                np.testing.assert_allclose(report["prices"], d["validation"]["prices"], atol=2e-12, rtol=0)
                json.dumps(report, allow_nan=False)
                for u in (0, -1j, .5-.5j, 15-.5j, 100):
                    np.testing.assert_allclose(vector_log_transform(d["parameters"], u, 1),
                                               log_transform(d["parameters"], u, 1), atol=2e-13, rtol=2e-13)

    def test_surface_broadcast_and_put_call_parity(self):
        d = load_example(2)
        p, spot = d["parameters"], d["spot"]
        args = (p, d["tau"], d["strikes"])
        calls = vector_prices(*args, 1, spot, d["eurYield"], d["usdYield"])
        puts = vector_prices(*args, -1, spot, d["eurYield"], d["usdYield"])
        t, ym, yn = (np.asarray(d[key])[:, None] for key in ("tau", "eurYield", "usdYield"))
        expected = spot*np.exp(-ym*t)-np.asarray(d["strikes"])*np.exp(-yn*t)
        self.assertEqual(calls.shape, (17, 11))
        np.testing.assert_allclose(calls-puts, expected, atol=5e-15, rtol=0)
        surface = price_vector(2, maturity=None)
        self.assertEqual(np.shape(surface["impliedVolatilityDecimal"]), (17, 11))
        row = d["tau"].index(1)
        np.testing.assert_allclose(surface["prices"][row], price_vector(2)["prices"], atol=2e-14, rtol=0)
        # Square grids still mean maturity/yield rows and option-sign columns.
        times, square_strikes = [1., 2.], [[1.1, 1.4], [1.2, 1.5]]
        rate_m, rate_n, signs = [.01, .02], [.03, .04], [-1, 1]
        square = vector_prices(p, times, square_strikes, signs, spot, rate_m, rate_n)
        separate = [vector_prices(p, times[i], square_strikes[i], signs, spot,
                                  rate_m[i], rate_n[i]) for i in range(2)]
        np.testing.assert_allclose(square, separate, atol=2e-14, rtol=0)

    def test_curve_moments_and_removable_limits(self):
        for factors in (1, 2):
            d = load_example(factors)
            row = d["tau"].index(1)
            raw = vector_log_transform(d["parameters"], [0, -1j], 1)
            y0n = d["usdYield"][row]+raw[0].real
            y0m = d["eurYield"][row]+raw[1].real
            matched = np.exp(raw+(-y0n+1j*np.array([0, -1j])*(y0n-y0m)))
            np.testing.assert_allclose(matched, [np.exp(-d["usdYield"][row]), np.exp(-d["eurYield"][row])], atol=1e-14)
        # d=0 at u=0 has a finite Riccati limit before moment explosion.
        critical = [[1, .01, 1, 0, .02, 0, 1, 0, -.5]]
        expected = .01*(1-2*math.log(1.5)) + .02/3
        self.assertAlmostEqual(vector_log_transform(critical, 0, 1).real, expected, places=14)
        zero = [[1, 0, .2, -.5, 0, 0, 1, 0, 0]]
        np.testing.assert_array_equal(vector_log_transform(zero, [0, -1j, 10-.5j], 1), [0, 0, 0])

    def test_calibration_improves_physical_objective_and_preserves_bounds(self):
        for factors in (1, 2):
            with self.subTest(factors=factors):
                d = load_example(factors)
                r = calibrate_vector(factors, 1, {"OutputDir": None})
                p = np.asarray(r["parameters"])
                self.assertLessEqual(r["objective"], r["initialObjective"])
                self.assertLess(r["gridPriceChange"], 1e-7)
                self.assertEqual(p[0, 6], 1)
                np.testing.assert_array_equal(p[:, 5], np.zeros(factors))
                maxvar = np.max(np.asarray(d["marketIVDecimal"])**2)
                lower = np.tile([.01, 0, .01, -1, 0, 0, 0, -1, -1], (factors, 1))
                upper = np.tile([2, maxvar, 1, 1, maxvar, 0, 2, 1, 1], (factors, 1))
                self.assertTrue(np.all((p >= lower) & (p <= upper)))
                coarse = vector_prices(p, d["tau"], d["strikes"], d["theta"], d["spot"], d["eurYield"], d["usdYield"])
                market = np.asarray(d["marketPrices"])
                expected = np.mean(((coarse-market)/market)**2) + 1e-4*(.5*np.sum(abs(p))+.5*np.sum(p*p))
                self.assertAlmostEqual(r["objective"], expected, places=13)
                json.dumps(r, allow_nan=False)
        with tempfile.TemporaryDirectory() as folder:
            # A bounded budget is honoured even before the first SQP gradient.
            r = calibrate_vector(1, 2, {"OutputDir": folder, "MaxFunctionEvaluations": 1,
                                       "InitialParameters": load_example()["parameters"][0], "Alpha": 0})
            self.assertEqual(r["output"]["functionEvaluations"], 1)
            self.assertEqual(r["exitflag"], 0)
            self.assertEqual(r["objective"], r["initialObjective"])
            np.testing.assert_allclose(r["parameters"], load_example()["parameters"], atol=1e-17, rtol=0)
            self.assertEqual(json.loads((Path(folder)/"calibration.json").read_text())["parameters"], r["parameters"])

    def test_mc_prices_discount_forward_and_batch_statistics(self):
        for factors in (1, 2):
            for scheme in ("EM", "QE"):
                with self.subTest(factors=factors, scheme=scheme):
                    # The second low-Feller factor needs finer Euler stepping;
                    # SE measures sampling error and cannot absorb time bias.
                    steps = 1000 if (factors == 2 and scheme == "EM") else 100
                    r = simulate_vector(factors, {"Paths": 20000, "Blocks": 20, "Steps": steps,
                                                  "Scheme": scheme, "OutputDir": None})
                    error = abs(np.asarray(r["prices"])-r["referencePrices"])
                    self.assertTrue(np.all(error < 6*np.asarray(r["priceSE"])+1e-9))
                    for actual, expected, se in (("discountMean", "expectedDiscount", "discountSE"),
                                                ("discountedSpotMean", "expectedDiscountedSpot", "discountedSpotSE")):
                        self.assertLess(abs(r[actual]-r[expected]), 6*r[se]+1e-9)
                    batches = np.asarray(r["blockPrices"])
                    np.testing.assert_allclose(batches.mean(axis=0), r["prices"], atol=1e-15)
                    np.testing.assert_allclose(batches.std(axis=0, ddof=1)/math.sqrt(20), r["priceSE"], atol=1e-15)
                    self.assertAlmostEqual(r["ecfReal"][0], r["discountMean"], places=14)
                    self.assertEqual(r["ecfImag"][0], 0)
                    if scheme == "EM":
                        diff = np.asarray(r["blockDirectPrices"])-batches
                        np.testing.assert_allclose(diff.mean(axis=0), r["pairedDifference"], atol=1e-15)
                        np.testing.assert_allclose(diff.std(axis=0, ddof=1)/math.sqrt(20), r["pairedDifferenceSE"], atol=1e-15)
                    else:
                        self.assertTrue(all(v is None for v in r["directPrice"]))
                    json.dumps(r, allow_nan=False)

    def test_seed_fitted_parameters_outputs_and_zero_variance(self):
        d = load_example()
        p = np.asarray(d["parameters"])
        p[0, 4] *= 1.1
        options = {"paths": 200, "blocks": 4, "steps": 10, "seed": 1234,
                   "parameters": p.tolist(), "output_dir": None}
        a, b = simulate_vector(options=options), simulate_vector(options=options)
        self.assertEqual(a["blockPrices"], b["blockPrices"])
        self.assertEqual(a["parameters"], p.tolist())
        options["seed"] = 1235
        self.assertNotEqual(a["blockPrices"], simulate_vector(options=options)["blockPrices"])
        with tempfile.TemporaryDirectory() as folder:
            options["output_dir"] = folder
            report = simulate_vector(options=options)
            loaded = json.loads((Path(folder)/"simulation_QE.json").read_text())
            self.assertEqual(report, loaded)
            with np.load(Path(folder)/"simulation_QE.npz") as batches:
                np.testing.assert_array_equal(batches["blockPrices"], report["blockPrices"])
                self.assertEqual(batches["blockECF"].shape, (4, 51))
        options.update(parameters=[[1, 0, .2, -.5, 0, 0, 1, 0, 0]], output_dir=None)
        zero = simulate_vector(options=options)
        np.testing.assert_array_equal(zero["terminalVariance"], [0])
        self.assertAlmostEqual(zero["discountMean"], zero["expectedDiscount"], places=14)
        self.assertAlmostEqual(zero["discountedSpotMean"], zero["expectedDiscountedSpot"], places=14)

    def test_invalid_inputs_and_moment_domain(self):
        p = [[1, .01, .2, -.2, .01, 0, 1, 0, 0]]
        for step, cutoff in ((0, 900), (.1, .3), (float("nan"), 900)):
            with self.assertRaises(ValueError):
                vector_prices(p, 1, [1], [1], 1, 0, 0, step=step, cutoff=cutoff)
        for invalid in ([[1, .01, 0, 0, .01, 0, 1, 0, 0]],
                        [[1, .01, .2, 2, .01, 0, 1, 0, 0]],
                        [[1, .01, .2, 0, .01, 1, 1, 0, 0]],
                        [[.1, .01, 2, 0, .01, 0, 1, 0, -2]]):
            with self.assertRaises(ValueError):
                validate_vector_parameters(invalid, 10)
        for options in ({"Paths": 99, "Blocks": 10}, {"Blocks": 1}, {"Steps": 0},
                        {"Scheme": "wrong"}, {"Seed": -1}, {"T": .123}, {"TYPO": 3},
                        {"Seed": 1, "seed": 2}):
            with self.assertRaises(ValueError):
                simulate_vector(options={**options, "OutputDir": None})
        invalid_initial = np.asarray(load_example()["parameters"])
        invalid_initial[0, 6] = .9
        with self.assertRaises(ValueError):
            calibrate_vector(options={"InitialParameters": invalid_initial, "OutputDir": None})
        with self.assertRaises(ValueError):
            calibrate_vector(options={"Alpha": -1, "OutputDir": None})
        with self.assertRaises(ValueError):
            price_vector(maturity=.123)


if __name__ == "__main__":
    unittest.main()
