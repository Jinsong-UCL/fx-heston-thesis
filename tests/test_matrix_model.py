"""Independent Riccati, saved-surface, calibration and Monte Carlo checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from matrix_model import (  # noqa: E402
    FINE_SETTINGS, calibrate_matrix, decode_parameters, encode_parameters,
    load_example, matrix_constraints, matrix_log_transform, matrix_prices,
    physical_parameters, price_matrix, simulate_matrix, validate_matrix_diffusion,
)


class MatrixPricingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_example()

    def test_preserved_full_surface_and_moment_anchors(self):
        result = price_matrix()
        np.testing.assert_allclose(result["prices"], self.data["validation"]["prices"],
                                   rtol=0, atol=2e-9)
        self.assertEqual(np.asarray(result["prices"]).shape, (15, 9))
        self.assertLess(result["diagnostics"]["discountMomentError"], 1e-12)
        self.assertLess(result["diagnostics"]["forwardMomentError"], 1e-12)
        self.assertTrue(np.isfinite(result["impliedVolatility"]).all())
        json.dumps(result, allow_nan=False)

    def test_independent_block_exponential_noncommuting_matrices(self):
        # At short times, the independently linearised Riccati system is
        # well-conditioned and its scalar determinant has no branch ambiguity.
        for p in self.data["parameters"]:
            b = {name: np.asarray(value) for name, value in p.items()}
            a, q = b["an"] - b["am"], b["sigma"].T @ b["sigma"]
            frequencies, maturities = [0, 0.4, 2, 5], [0.01, 0.1, 0.25]
            ode = np.exp(matrix_log_transform(p, frequencies, maturities, 1e-11))
            exact = np.empty_like(ode)
            for i, maturity in enumerate(maturities):
                for j, xi in enumerate(frequencies):
                    c = b["kappa"] - 1j * xi * (b["sigma"].T @ b["rho"] @ a)
                    h = ((xi * xi + 1j * xi) * (a @ a)
                         - 2j * xi * (b["rn"] - b["rm"]) + 2 * b["rn"])
                    h = (h + h.T) / 2
                    flow = expm(np.block([[-c.T, -h], [-q, c]]) * maturity / 2)
                    u, v = flow[2:, 2:], flow[:2, 2:]
                    f = np.linalg.solve(u.T, v.T).T
                    exact[i, j] = np.exp(b["beta"] * np.trace(c) * maturity
                                          + np.trace(f @ b["v0"])) / np.linalg.det(u)
            np.testing.assert_allclose(ode, exact, rtol=0, atol=1e-9)

    def test_put_call_parity_and_unsorted_repeated_times(self):
        m = self.data["market"]
        time_ = np.array([1.0, 0.25, 1.0])
        strikes = np.array([1.3, 1.4, 1.5])
        calls = matrix_prices(self.data["parameters"], time_, strikes, 1,
                              m["spot"], 0.01, 0.02, FINE_SETTINGS)
        puts = matrix_prices(self.data["parameters"], time_, strikes, -1,
                             m["spot"], 0.01, 0.02, FINE_SETTINGS)
        np.testing.assert_allclose(calls - puts,
                                   m["spot"] * np.exp(-0.01 * time_) - strikes * np.exp(-0.02 * time_),
                                   rtol=0, atol=1e-15)
        zero = matrix_log_transform(self.data["parameters"][0], [0, 1, -1j], [0])
        np.testing.assert_array_equal(zero, np.zeros((1, 3)))

    def test_admissibility_is_enforced(self):
        for name, value in (("beta", 0.1), ("rho", [[1.1, 0], [0, 1.1]]),
                            ("v0", [[-1, 0], [0, 1]]), ("sigma", [[0, 0], [0, 0]]),
                            ("am", [[1, 0], [0, 0]]), ("an", [[0, 1], [0, 0]])):
            p = deepcopy(self.data["parameters"])
            p[0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_matrix_diffusion(p)

    def test_square_surface_uses_maturity_rows_and_sign_columns(self):
        p, spot = self.data["parameters"], self.data["market"]["spot"]
        strikes = np.array([[1.2, 1.4], [1.3, 1.5]])
        tau, ym, yn = np.array([0.25, 1]), np.array([0.01, 0.02]), np.array([0.03, 0.04])
        theta = np.array([-1, 1])
        implicit = matrix_prices(p, tau, strikes, theta, spot, ym, yn, FINE_SETTINGS)
        explicit = matrix_prices(p, tau[:, None], strikes, theta[None, :], spot,
                                 ym[:, None], yn[:, None], FINE_SETTINGS)
        np.testing.assert_allclose(implicit, explicit, rtol=0, atol=1e-14)


class MatrixCalibrationChecks(unittest.TestCase):
    def test_parameter_roundtrip_fixed_blocks_bounds_and_physical_penalty(self):
        data = load_example()
        p = data["initialParameters"]
        x, lower, upper, scale = encode_parameters(p)
        reconstructed = decode_parameters(x, scale)
        np.testing.assert_allclose(physical_parameters(reconstructed), physical_parameters(p),
                                   rtol=0, atol=1e-14)
        self.assertEqual(len(x), 35)
        self.assertEqual(np.count_nonzero(lower != upper), 33)
        self.assertTrue(np.all((x >= lower) & (x <= upper)))
        self.assertLessEqual(np.max(matrix_constraints(reconstructed)), 0)
        manual = sum(abs(float(value)) + float(value)**2
                     for block in p for name in ("kappa", "sigma", "rho", "v0", "am", "an", "rm", "rn", "beta")
                     for value in np.asarray(block[name]).ravel()) * 0.00005
        z = physical_parameters(p)
        self.assertAlmostEqual(manual, 0.00005 * (np.abs(z).sum() + (z**2).sum()), places=15)
        self.assertNotAlmostEqual(manual, 0.00005 * (np.abs(x).sum() + (x**2).sum()), places=5)

    def test_limited_fit_truthfully_reports_nonconvergence_and_can_simulate(self):
        # One evaluation is an intentional budget boundary, not an assertion
        # that an unconverged optimizer has calibrated the market.
        with tempfile.TemporaryDirectory() as directory:
            fit = calibrate_matrix(1, {"max_function_evaluations": 1, "output_dir": directory,
                                       "initial_parameters": load_example()["initialParameters"],
                                       "use_parallel": False})
            self.assertFalse(fit["output"]["success"])
            self.assertEqual(fit["exitflag"], 0)
            self.assertEqual(fit["output"]["functionEvaluations"], 1)
            self.assertAlmostEqual(fit["initialObjective"], 0.0146957644335, delta=2e-9)
            self.assertLessEqual(fit["objective"], fit["initialObjective"])
            self.assertLess(fit["metrics"]["gridIVChangePP"], 1e-4)
            self.assertTrue((Path(directory) / "calibration.json").exists())
            simulation = simulate_matrix({"Parameters": fit["parameters"], "Steps": 5, "Paths": 80})
            self.assertTrue(np.isfinite(simulation["prices"]).all())
            json.dumps(fit, allow_nan=False)
            json.dumps(simulation, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "Duplicate option"):
            calibrate_matrix(1, {"MaxFunctionEvaluations": 1, "max_function_evaluations": 2})


class MatrixSimulationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = {"Steps": 100, "Paths": 4000}
        cls.simulation = simulate_matrix(cls.options)

    def test_discount_forward_variance_and_prices_match_transform_statistically(self):
        r = self.simulation
        moments = np.asarray(r["discountForwardLogMoments"][:2])
        expected = np.array([r["expectedDiscount"], r["expectedDiscountedSpot"]])
        self.assertLess(np.max(np.abs((moments - expected) / np.asarray(r["momentSE"][:2]))), 4)
        self.assertLess(np.max(np.abs(r["varianceZ"])), 4)
        self.assertLess(np.max(np.abs(r["priceZ"])), 4)
        self.assertLess(r["maxRealZ"], 4)
        self.assertLess(r["maxImagZ"], 4)
        self.assertTrue(np.all(np.asarray(r["priceSE"]) > 0))
        self.assertEqual(r["blocks"], 40)
        self.assertEqual(r["seed"], 922761)

    def test_seed_repeatability_and_batch_controls(self):
        repeated = simulate_matrix({"steps": 100, "paths": 4000,
                                    "seed": 922761, "output_dir": None})
        for name in ("prices", "priceSE", "varianceMean", "ecfReal", "ecfImag"):
            self.assertEqual(self.simulation[name], repeated[name])
        changed = simulate_matrix({"Steps": 3, "Paths": 80, "Blocks": 4, "Seed": 123})
        self.assertEqual(changed["blocks"], 4)
        self.assertEqual(changed["seed"], 123)
        json.dumps(changed, allow_nan=False)
        for options in ({"Paths": 80, "paths": 80}, {"Seed": 1, "seed": 2},
                        {"OutputDir": None, "output_dir": "unused"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "Duplicate option"):
                simulate_matrix(options)

    def test_rejects_unsupported_structure_and_controls(self):
        for options in ({"Steps": 0}, {"Paths": 81}, {"Paths": 40}, {"T": 0.9},
                        {"Blocks": 1}, {"Seed": 2**32 - 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                simulate_matrix(options)
        for block, name, value in ((0, "beta", 0.75), (0, "an", [[1, 0], [0, 1]]),
                                   (1, "rn", [[1, 0], [0, 1]])):
            p = load_example()["parameters"]
            p[block][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                simulate_matrix({"Parameters": p, "Steps": 1, "Paths": 80})


if __name__ == "__main__":
    unittest.main()
