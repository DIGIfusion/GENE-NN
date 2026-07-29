"""
Command-line inference from an existing CSV dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gene_nn.inference.common import (
    detect_model_type,
    predict_classreg,
    predict_multioutput,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GENE-NN surrogate inference on a CSV dataset."
    )

    parser.add_argument("--csv", required=True, help="Input CSV file.")

    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Model directory. For MH MLP, this contains model.pt and "
            "model_config.json. For MH class-regression, this contains "
            "pipeline_config.json."
        ),
    )

    parser.add_argument("--out-csv", required=True, help="Output CSV file.")

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device for PyTorch inference, for example cpu, cuda, or cuda:0. "
            "Default: auto."
        ),
    )

    parser.add_argument("--batch-size", type=int, default=4096, help="Inference batch size. Default: 4096.")

    parser.add_argument(
        "--keep-cols",
        nargs="*",
        default=None,
        help="Input columns to keep in the output. Default: keep all input columns.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    model_dir = Path(args.model_dir)
    out_csv = Path(args.out_csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_type = detect_model_type(model_dir)
    print(f"[info] model type: {model_type}")

    df = pd.read_csv(csv_path)

    if model_type == "mh-mlp":
        out_df = predict_multioutput(
            df=df,
            model_dir=model_dir,
            device=args.device,
            batch_size=args.batch_size,
            keep_cols=args.keep_cols,
        )

    elif model_type == "mh-classreg":
        out_df = predict_classreg(
            df=df,
            model_dir=model_dir,
            device=args.device,
            batch_size=args.batch_size,
            keep_cols=args.keep_cols,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"[done] wrote {len(out_df)} rows to {out_csv}")


if __name__ == "__main__":
    main()