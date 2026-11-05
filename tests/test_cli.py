"""Exercise the public commands and saved-fit interchange on real calculations."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "python" / "run_heston.py"


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="cli-", dir=ROOT / "tmp")
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def command(self, *arguments, success=True):
        completed = subprocess.run([sys.executable, str(CLI), *map(str, arguments)],
                                   cwd=ROOT, text=True, capture_output=True, timeout=180)
        if success:
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        else:
            self.assertNotEqual(completed.returncode, 0)
        return completed

    def read(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_help_and_invalid_simulation(self):
        self.assertIn("calibrate", self.command("matrix", "--help").stdout)
        output = self.directory / "invalid"
        error = self.command("vector", "simulate", "--paths", 81, "--output-dir", output, success=False)
        self.assertIn("divisible", error.stderr.lower())
        self.assertFalse(output.exists())
        error = self.command("matrix", "price", "--maturity", 123, "--output-dir", output, success=False)
        self.assertIn("maturity", error.stderr.lower())

    def test_vector_fit_reprice_and_simulate_saved_json(self):
        fit_dir, price_dir, mc_dir = [self.directory / name for name in ("fit", "price", "mc")]
        self.command("vector", "calibrate", "--max-iterations", 1, "--max-evaluations", 2, "--output-dir", fit_dir)
        fit_file = fit_dir / "calibration.json"
        fit = self.read(fit_file)
        self.assertEqual(fit["exitflag"], 0)
        self.command("vector", "price", "--parameters", fit_file, "--all-maturities", "--output-dir", price_dir)
        prices = self.read(price_dir / "prices.json")
        np.testing.assert_allclose(prices["prices"], fit["prices"], atol=3e-12, rtol=0)
        # Raw one-row MATLAB JSON parameters must interchange as well.
        raw_file = self.directory / "matlab_parameters.json"
        raw_file.write_text(json.dumps(np.asarray(fit["parameters"]).ravel().tolist()), encoding="utf-8")
        self.command("vector", "simulate", "--parameters", raw_file, "--paths", 80, "--batches", 4,
                     "--steps", 5, "--seed", 17, "--scheme", "EM", "--output-dir", mc_dir)
        simulation = self.read(mc_dir / "simulation_EM.json")
        np.testing.assert_allclose(simulation["parameters"], fit["parameters"])
        self.assertTrue(np.all(np.asarray(simulation["priceSE"]) > 0))
        self.assertTrue(list(mc_dir.glob("*.npz")))

    def test_matrix_fit_reprice_and_simulate_saved_json(self):
        fit_dir, price_dir, mc_dir = [self.directory / name for name in ("fit", "price", "mc")]
        self.command("matrix", "calibrate", "--max-iterations", 1, "--max-evaluations", 2, "--output-dir", fit_dir)
        fit_file = fit_dir / "calibration.json"
        fit = self.read(fit_file)
        self.assertEqual(fit["exitflag"], 0)
        self.command("matrix", "price", "--parameters", fit_file, "--all-maturities", "--output-dir", price_dir)
        prices = self.read(price_dir / "prices.json")
        np.testing.assert_allclose(prices["prices"], fit["prices"], atol=2e-8, rtol=0)
        self.command("matrix", "simulate", "--parameters", fit_file, "--paths", 80,
                     "--steps", 5, "--output-dir", mc_dir)
        simulation = self.read(mc_dir / "simulation.json")
        self.assertEqual(simulation["parameters"], fit["parameters"])
        self.assertTrue(np.all(np.asarray(simulation["priceSE"]) > 0))
        self.command("matrix", "price", "--data", ROOT / "data/matrix.json", "--parameters", fit_file,
                     "--maturity", 1, "--output-dir", price_dir)
        self.assertEqual(np.asarray(self.read(price_dir / "prices.json")["prices"]).shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
