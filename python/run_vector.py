"""Price the saved vector examples independently, without MATLAB or third-party packages."""
import argparse
import json
from pathlib import Path

from vector_reference import half_moment_prices, implied_vol

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, choices=(1, 2), default=1)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=.025)
    parser.add_argument("--cutoff", type=float, default=1800)
    parser.add_argument("--parameters", type=Path, help="MATLAB calibration.json to price instead of saved parameters")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads((ROOT / f"data/vector_N{args.factors}.json").read_text())
    try:
        row = data["tau"].index(args.maturity)
    except ValueError:
        parser.error("Choose a maturity present in the example data.")
    p = data["parameters"]
    if args.parameters:
        p = json.loads(args.parameters.read_text())["parameters"]
        if p and isinstance(p[0], (int, float)):
            p = [p]  # MATLAB jsonencode collapses a one-row matrix.
        if len(p) != args.factors:
            parser.error("The supplied calibration has a different factor count.")
    k, signs = data["strikes"][row], data["theta"]
    ym, yn = data["eurYield"][row], data["usdYield"][row]
    prices = half_moment_prices(p, args.maturity, data["spot"], ym, yn, k, signs,
                                args.step, args.cutoff)
    iv = [implied_vol(c, data["spot"], strike, args.maturity, ym, yn, sign)
          for c, strike, sign in zip(prices, k, signs)]
    result = {"N": args.factors, "maturity": args.maturity, "parameters": p,
              "prices": prices, "impliedVolatilityDecimal": iv,
              "grid": {"step": args.step, "cutoff": args.cutoff}}
    output = args.output or ROOT / "results" / f"python_vector_N{args.factors}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved independent prices to {output}")


if __name__ == "__main__":
    main()
