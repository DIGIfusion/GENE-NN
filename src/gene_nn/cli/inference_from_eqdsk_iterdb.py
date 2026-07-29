"""
Command-line inference from EQDSK/ITERDB-derived input quantities.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from gene_nn.inference.common import (
    detect_model_type,
    load_model_feature_cols,
    predict_classreg,
    predict_multioutput,
    validate_user_parameters,
)
from gene_nn.utils.equilibrium import extract_scalars_from_eqdsk_iterdb


DEFAULT_KYMIN = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def build_input_dataframe(
    *,
    eqdsk_path: str | Path,
    iterdb_path: str | Path,
    x0_values: list[float],
    kymin_values: list[float],
) -> pd.DataFrame:
    """
    Extract local inputs from EQDSK/ITERDB and expand over kymin.
    """
    scalar_rows = extract_scalars_from_eqdsk_iterdb(
        eqdsk_path=eqdsk_path,
        iterdb_path=iterdb_path,
        rho0=x0_values,
    )

    rows = []

    for scalar_row in scalar_rows:
        for ky in kymin_values:
            row = dict(scalar_row)
            row["kymin"] = float(ky)
            rows.append(row)

    df = pd.DataFrame(rows)

    eqdsk_path = Path(eqdsk_path)
    iterdb_path = Path(iterdb_path)

    case_name = eqdsk_path.parent.name or eqdsk_path.stem

    df.insert(0, "case", case_name)
    df.insert(1, "eqdsk_path", str(eqdsk_path))
    df.insert(2, "iterdb_path", str(iterdb_path))

    return df


def parse_key_value_float(items: list[str] | None, *, arg_name: str) -> dict[str, float]:
    """
    Parse arguments like:
        --set beta=0.02 omegatorref=0.0
    """
    parsed = {}

    if not items:
        return parsed

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid {arg_name} entry {item!r}. Expected format name=value."
            )

        key, value = item.split("=", 1)
        key = key.strip()

        if not key:
            raise ValueError(f"Invalid {arg_name} entry {item!r}: empty key.")

        parsed[key] = float(value)

    return parsed


def parse_key_values_list(
    items: list[str] | None,
    *,
    arg_name: str,
) -> list[tuple[str, list[float]]]:
    """
    Parse arguments like:
        --scan beta=0.01,0.02,0.03
        --scale-scan beta=0.9,1.0,1.1
    """
    parsed = []

    if not items:
        return parsed

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid {arg_name} entry {item!r}. "
                "Expected format name=value1,value2,..."
            )

        key, values = item.split("=", 1)
        key = key.strip()

        if not key:
            raise ValueError(f"Invalid {arg_name} entry {item!r}: empty key.")

        vals = [float(v) for v in values.split(",") if v.strip()]

        if not vals:
            raise ValueError(f"Invalid {arg_name} entry {item!r}: no values found.")

        parsed.append((key, vals))

    return parsed


def apply_fixed_overrides(df: pd.DataFrame, overrides: dict[str, float]) -> pd.DataFrame:
    """
    Replace generated input columns by user-given fixed values.
    """
    out = df.copy()

    for key, value in overrides.items():
        out[key] = float(value)

    return out


def expand_parameter_scans(
    df: pd.DataFrame,
    *,
    absolute_scans: list[tuple[str, list[float]]],
    scale_scans: list[tuple[str, list[float]]],
) -> pd.DataFrame:
    """
    Expand dataframe over absolute and multiplicative parameter scans.
    """
    if not absolute_scans and not scale_scans:
        return df

    abs_keys = {key for key, _ in absolute_scans}
    scale_keys = {key for key, _ in scale_scans}
    overlap = abs_keys & scale_keys

    if overlap:
        raise ValueError(
            "The same parameter cannot be used in both --scan and --scale-scan "
            f"in the same command. Overlap: {sorted(overlap)}"
        )

    scan_specs = []

    for key, values in absolute_scans:
        scan_specs.append(("absolute", key, values))

    for key, values in scale_scans:
        if key not in df.columns:
            raise KeyError(
                f"Cannot scale-scan {key!r}: column is not present in the input dataframe. "
                f"Available columns: {list(df.columns)}"
            )

        scan_specs.append(("scale", key, values))

    rows = []

    for _, base_row in df.iterrows():
        choices = [values for _, _, values in scan_specs]

        for combo in product(*choices):
            row = base_row.copy()

            for (kind, key, _), value in zip(scan_specs, combo):
                if kind == "absolute":
                    row[key] = float(value)
                    row[f"{key}_scan_value"] = float(value)

                elif kind == "scale":
                    original = float(base_row[key])
                    row[key] = original * float(value)
                    row[f"{key}_scale_factor"] = float(value)
                    row[f"{key}_unscaled"] = original

                else:
                    raise ValueError(f"Unknown scan kind: {kind}")

            rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GENE-NN surrogate inference from EQDSK + ITERDB. "
            "The script extracts local scalar inputs, expands over kymin, "
            "optionally overrides or scans input parameters, and runs the saved "
            "surrogate model."
        )
    )

    parser.add_argument("--eqdsk-path", required=True, help="Path to EQDSK file.")
    parser.add_argument("--iterdb-path", required=True, help="Path to ITERDB file.")

    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Model directory. For MH MLP, this contains model.pt and model_config.json. "
            "For MH class-regression, this contains pipeline_config.json."
        ),
    )

    parser.add_argument(
        "--x0",
        type=float,
        nargs="+",
        required=True,
        help="Radial locations in rho_tor_norm.",
    )

    parser.add_argument(
        "--kymin",
        type=float,
        nargs="+",
        default=DEFAULT_KYMIN,
        help="kymin values. Default: 0.1, 0.2, ..., 1.0.",
    )

    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help=(
            "Fixed parameter overrides, format name=value. "
            "Example: --set beta=0.02 omegatorref=0.0"
        ),
    )

    parser.add_argument(
        "--scan",
        nargs="*",
        default=None,
        help=(
            "Absolute parameter scans, format name=v1,v2,v3. "
            "Example: --scan beta=0.01,0.02,0.03"
        ),
    )

    parser.add_argument(
        "--scale-scan",
        nargs="*",
        default=None,
        help=(
            "Multiplicative scans around current input values, format name=f1,f2,f3. "
            "Example: --scale-scan beta=0.9,1.0,1.1,1.2"
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

    parser.add_argument("--batch-size", type=int, default=4096)

    parser.add_argument(
        "--keep-cols",
        nargs="*",
        default=None,
        help="Input columns to keep in the output. Default: keep all input columns.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eqdsk_path = Path(args.eqdsk_path)
    iterdb_path = Path(args.iterdb_path)
    model_dir = Path(args.model_dir)
    out_csv = Path(args.out_csv)

    if not eqdsk_path.exists():
        raise FileNotFoundError(f"EQDSK file not found: {eqdsk_path}")

    if not iterdb_path.exists():
        raise FileNotFoundError(f"ITERDB file not found: {iterdb_path}")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_type = detect_model_type(model_dir)
    model_feature_cols = load_model_feature_cols(model_dir, model_type)

    print(f"[info] model type: {model_type}")

    fixed_overrides = parse_key_value_float(args.set, arg_name="--set")
    absolute_scans = parse_key_values_list(args.scan, arg_name="--scan")
    scale_scans = parse_key_values_list(args.scale_scan, arg_name="--scale-scan")

    df = build_input_dataframe(
        eqdsk_path=eqdsk_path,
        iterdb_path=iterdb_path,
        x0_values=args.x0,
        kymin_values=args.kymin,
    )

    override_keys = list(fixed_overrides.keys())
    absolute_scan_keys = [key for key, _ in absolute_scans]
    scale_scan_keys = [key for key, _ in scale_scans]

    validate_user_parameters(
        df,
        override_keys,
        model_feature_cols=model_feature_cols,
        arg_name="--set",
    )

    validate_user_parameters(
        df,
        absolute_scan_keys,
        model_feature_cols=model_feature_cols,
        arg_name="--scan",
    )

    validate_user_parameters(
        df,
        scale_scan_keys,
        model_feature_cols=model_feature_cols,
        arg_name="--scale-scan",
    )

    df = apply_fixed_overrides(df, fixed_overrides)
    df = expand_parameter_scans(
        df,
        absolute_scans=absolute_scans,
        scale_scans=scale_scans,
    )

    print(f"[info] input rows: {len(df)}")
    print(f"[info] x0 values: {len(args.x0)}")
    print(f"[info] kymin values: {len(args.kymin)}")

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