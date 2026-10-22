import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from helpers import positive_scores, prepare_inputs, rank_customers
from tracking import load_selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--selection", type=Path,
        default=Path(__file__).resolve().parent / "results/selection.json",
    )
    args = parser.parse_args()
    if args.input_csv.resolve() == args.output_csv.resolve():
        parser.error("Input and output must be different files.")

    table = pd.read_csv(args.input_csv, keep_default_na=False)
    ids, X, _ = prepare_inputs(table, mode="predict")
    model, selection = load_selected(args.selection)
    probabilities = positive_scores(model, X) if len(X) else np.empty(0, dtype=float)
    ranked = rank_customers(ids, probabilities, selection["policy"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(ranked)} rows; selected {int(ranked['selected'].sum())}; "
          f"candidate={selection['candidate_id']}; run_id={selection['run_id']}")


if __name__ == "__main__":
    main()