"""Calibrate, price and simulate the vector or matrix FX Heston examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_parameters(path: Path, family: str):
    value = read_json(path)
    if isinstance(value, dict) and "parameters" in value:
        value = value["parameters"]
    if family == "vector" and isinstance(value, list) and value and isinstance(value[0], (int, float)):
        value = [value]  # MATLAB jsonencode collapses a one-row matrix.
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_subparsers(dest="family", required=True)
    for family in ("vector", "matrix"):
        model = models.add_parser(family, help=f"{family.capitalize()} Heston model")
        operations = model.add_subparsers(dest="operation", required=True)
        for operation in ("price", "calibrate", "simulate"):
            command = operations.add_parser(operation)
            command.add_argument("--data", type=Path, help="JSON input with the same schema as data/")
            command.add_argument("--output-dir", type=Path, help="default: results/python/<model>")
            if family == "vector":
                command.add_argument("--factors", type=int, choices=(1, 2), default=1)
            if operation == "calibrate":
                command.add_argument("--initial-parameters", type=Path, help="raw parameters or calibration.json")
                command.add_argument("--max-iterations", type=positive_int, default=150)
                command.add_argument("--max-evaluations", type=positive_int, default=16000)
                if family == "vector":
                    command.add_argument("--alpha", type=float, default=1e-4)
                    command.add_argument("--lambda", dest="elastic_lambda", type=float, default=.5)
                else:
                    command.add_argument("--parallel", action="store_true", help="parallel finite differences")
            else:
                command.add_argument("--parameters", type=Path, help="raw parameters or MATLAB/Python calibration.json")
                command.add_argument("--maturity", type=positive_float, default=1., help="must occur in the input data (default: 1)")
            if operation == "price":
                command.add_argument("--all-maturities", action="store_true", help="price the entire input surface")
                command.add_argument("--cutoff", type=positive_float, default=1800. if family == "vector" else 1200.)
                if family == "vector":
                    command.add_argument("--step", type=positive_float, default=.025)
                else:
                    command.add_argument("--nodes", type=positive_int, default=560)
                    command.add_argument("--tolerance", type=positive_float, default=1e-10)
            if operation == "simulate":
                command.add_argument("--paths", type=positive_int, default=20000)
                command.add_argument("--steps", type=positive_int, default=250)
                if family == "vector":
                    command.add_argument("--scheme", choices=("EM", "QE"), default="QE")
                    command.add_argument("--batches", type=positive_int, default=40)
                    command.add_argument("--seed", type=int)
    return parser


def run(args: argparse.Namespace) -> tuple[dict, Path]:
    name = f"vector_N{args.factors}" if args.family == "vector" else "matrix"
    output_dir = args.output_dir or ROOT / "results" / "python" / name
    data = read_json(args.data) if args.data else None
    options = {"OutputDir": str(output_dir)}
    parameter_path = getattr(args, "parameters", None)
    parameters = read_parameters(parameter_path, args.family) if parameter_path else None
    if parameters is not None:
        options["Parameters"] = parameters
    if args.operation == "calibrate":
        options["MaxFunctionEvaluations"] = args.max_evaluations
        if args.initial_parameters:
            options["InitialParameters"] = read_parameters(args.initial_parameters, args.family)
    elif args.operation == "simulate":
        options.update(T=args.maturity, Paths=args.paths, Steps=args.steps)

    if args.family == "vector":
        from vector_model import calibrate_vector, price_vector, simulate_vector

        if args.operation == "price":
            result = price_vector(args.factors, None if args.all_maturities else args.maturity,
                                  parameters, data=data, step=args.step, cutoff=args.cutoff)
        elif args.operation == "calibrate":
            options.update(Alpha=args.alpha, Lambda=args.elastic_lambda)
            result = calibrate_vector(args.factors, args.max_iterations, options, data=data)
        else:
            options.update(Scheme=args.scheme, Blocks=args.batches)
            if args.seed is not None:
                options["Seed"] = args.seed
            result = simulate_vector(args.factors, options, data=data)
    else:
        from matrix_model import calibrate_matrix, load_example, price_matrix, simulate_matrix

        if args.operation == "price":
            options["Settings"] = {"nodes": args.nodes, "cutoff": args.cutoff, "tolerance": args.tolerance}
            if not args.all_maturities:
                data = data or load_example()
                market = data["market"]
                rows = [i for i, t in enumerate(market["tau"]) if abs(float(t) - args.maturity) < 1e-10]
                if not rows:
                    raise ValueError("Choose a maturity present in the input data.")
                count = len(market["tau"])
                data = dict(data)
                data["market"] = {key: [value[rows[0]]] if isinstance(value, list) and len(value) == count else value
                                  for key, value in market.items()}
            result = price_matrix(options, data=data)
        elif args.operation == "calibrate":
            options["UseParallel"] = args.parallel
            result = calibrate_matrix(args.max_iterations, options, data=data)
        else:
            result = simulate_matrix(options, data=data)

    filename = "prices.json" if args.operation == "price" else "calibration.json" if args.operation == "calibrate" else (
        f"simulation_{args.scheme}.json" if args.family == "vector" else "simulation.json")
    output = output_dir / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result, output


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result, output = run(args)
    except ImportError as error:
        parser.exit(2, f"Dependency error: {error}. Run: python -m pip install -r requirements.txt\n")
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"Input/output error: {error}\n")
    print(f"Saved {args.family} {args.operation} result to {output}")
    if args.operation == "calibrate":
        print(f"Exit flag: {result.get('exitflag', 'see JSON')}; inspect solver output and fit diagnostics before using a fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
