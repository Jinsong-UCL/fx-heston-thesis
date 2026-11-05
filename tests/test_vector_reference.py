"""Independent prices, measure matching, and error handling for the public inputs."""
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from vector_reference import half_moment_prices, log_transform, black_price, implied_vol


class VectorChecks(unittest.TestCase):
    def test_saved_prices_and_put_call_parity(self):
        for factors in (1, 2):
            data = json.loads((ROOT / f"data/vector_N{factors}.json").read_text())
            row = data["tau"].index(1)
            ym, yn = data["eurYield"][row], data["usdYield"][row]
            strikes, p, s = data["strikes"][row], data["parameters"], data["spot"]
            prices = half_moment_prices(p, 1, s, ym, yn, strikes, data["theta"], .025, 1800)
            self.assertLess(max(abs(a-b) for a, b in zip(prices, data["validation"]["prices"])), 2e-12)
            calls = half_moment_prices(p, 1, s, ym, yn, [strikes[5]], [1])
            puts = half_moment_prices(p, 1, s, ym, yn, [strikes[5]], [-1])
            self.assertAlmostEqual(calls[0]-puts[0], s*math.exp(-ym)-strikes[5]*math.exp(-yn), places=12)
            self.assertLess(abs(log_transform(p, 0, 1).imag), 1e-12)
            self.assertLess(abs(log_transform(p, -1j, 1).imag), 1e-12)

    def test_black_scholes_inversion(self):
        for sign in (-1, 1):
            price = black_price(1.2, 1.3, 2, .01, .03, .2, sign)
            self.assertAlmostEqual(implied_vol(price, 1.2, 1.3, 2, .01, .03, sign), .2, places=12)
        with self.assertRaises(ValueError):
            implied_vol(-.1, 1.2, 1.3, 2, .01, .03, 1)

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            half_moment_prices([[1,.01,.2,-.2,.01,0,1,0,0]], 1, 1, 0, 0, [1], [1], 0, 900)


if __name__ == "__main__":
    unittest.main()
