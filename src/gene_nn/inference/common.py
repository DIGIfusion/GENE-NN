from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from gene_nn.models import load_model


ALIASES = {
    "omn_e": "omn",
    "omt_e": "omt",
}

SUPPORTED_TARGET_TRANSFORMS = {"raw", "log", "log1p", "signed_log1p"}


def load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_model_type(model_dir: str | Path) -> str:
    """
    Detect whether model_dir contains a multi-head MLP or a
    classification + regression pipeline.
    """
    model_dir = Path(model_dir)

    if (model_dir / "pipeline_config.json").exists():
        return "mh-classreg"

    if (
        (model_dir / "model_config.json").exists()
        and (model_dir / "model.pt").exists()
        and (model_dir / "preprocessor.joblib").exists()
    ):
        return "mh-mlp"

    raise FileNotFoundError(
        f"Could not identify model type in {model_dir}. Expected either:\n"
        "  - pipeline_config.json for class-regression model\n"
        "  - model.pt, model_config.json, preprocessor.joblib for multi-head MLP"
    )


def load_model_feature_cols(model_dir: str | Path, model_type: str) -> list[str]:
    """
    Load the input feature names expected by a saved model.
    """
    model_dir = Path(model_dir)

    if model_type == "mh-mlp":
        config_path = model_dir / "model_config.json"
    elif model_type == "mh-classreg":
        config_path = model_dir / "pipeline_config.json"
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if "feature_cols" not in config:
        raise KeyError(f"Missing 'feature_cols' in {config_path}")

    return list(config["feature_cols"])


def validate_user_parameters(
    df: pd.DataFrame,
    keys: list[str],
    *,
    model_feature_cols: list[str],
    arg_name: str,
) -> None:
    """
    Validate user-given --set/--scan/--scale-scan parameter names.

    Parameters must exist in the generated input dataframe. If a parameter exists
    but is not used by the selected model, inference can continue, but the user
    is warned that changing it will not affect predictions.
    """
    if not keys:
        return

    missing = [key for key in keys if key not in df.columns]

    if missing:
        raise KeyError(
            f"Invalid {arg_name} parameter(s): {missing}. "
            f"Available input columns: {list(df.columns)}"
        )

    unused = [key for key in keys if key not in model_feature_cols]

    if unused:
        print(
            f"[warning] {arg_name} parameter(s) {unused} are not used by this model. "
            "Changing them will not affect the predictions."
        )


def make_base_output_df(
    df: pd.DataFrame,
    keep_cols: list[str] | None,
) -> pd.DataFrame:
    if keep_cols is None:
        return df.copy()

    keep = [col for col in keep_cols if col in df.columns]
    missing = [col for col in keep_cols if col not in df.columns]

    if missing:
        print(f"[warning] requested keep-cols not found and skipped: {missing}")

    return df.loc[:, keep].copy()


def build_X_from_df(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    Build the model input matrix from a dataframe.

    The output columns follow feature_cols exactly. A small set of aliases is
    allowed for compatibility with other datasets.
    """
    selected_cols = []
    missing = []

    for col in feature_cols:
        if col in df.columns:
            selected_cols.append(col)
        elif col in ALIASES and ALIASES[col] in df.columns:
            selected_cols.append(ALIASES[col])
        else:
            missing.append(col)

    if missing:
        raise KeyError(
            "Input dataframe is missing required model feature columns.\n"
            f"Missing features: {missing}\n"
            f"Model requires: {feature_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    return (
        df.loc[:, selected_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32, copy=True)
    )


def valid_feature_rows(X: np.ndarray) -> np.ndarray:
    valid_mask = np.isfinite(X).all(axis=1)

    if not valid_mask.any():
        raise ValueError(
            "No valid rows found: required feature columns contain missing "
            "or non-numeric values."
        )

    n_bad = int((~valid_mask).sum())

    if n_bad > 0:
        print(
            f"[warning] {n_bad} rows contain missing or invalid required inputs; "
            "predictions for those rows will be NaN."
        )

    return valid_mask


@torch.no_grad()
def predict_numpy(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    device: str | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Run batched PyTorch inference and return predictions as a NumPy array.
    """
    dev = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model.eval()
    model.to(dev)

    X = np.asarray(X, dtype=np.float32)
    outputs = []

    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(dev)
        yb = model(xb).detach().cpu().numpy()
        outputs.append(yb)

    if not outputs:
        return np.zeros((0, 0), dtype=np.float32)

    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def predict_classes_and_probs(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    device: str | None = None,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict class labels and softmax probabilities from classifier logits.
    """
    logits = predict_numpy(
        model,
        X,
        device=device,
        batch_size=batch_size,
    )

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred_class = np.argmax(probs, axis=1)

    return pred_class, probs


def load_target_preprocessing(
    model_dir: str | Path,
    config: dict[str, Any],
) -> tuple[list[str], dict[str, str], Any | None]:
    """
    Load target columns, target transforms, and optional target scaler.

    Target transforms are read from model_config.json by default. A separate
    target_transform_config.json is also supported for compatibility with older
    saved runs.
    """
    model_dir = Path(model_dir)
    target_cols = list(config.get("target_cols", []))

    transform_config = load_json_if_exists(model_dir / "target_transform_config.json")

    if transform_config is not None:
        transform_map = transform_config.get("target_transforms", {})
    else:
        transform_map = config.get("target_transforms", {})

    transform_map = {col: transform_map.get(col, "raw") for col in target_cols}

    for col, kind in transform_map.items():
        if kind not in SUPPORTED_TARGET_TRANSFORMS:
            raise ValueError(
                f"Unsupported target transform {kind!r} for target {col!r}. "
                f"Supported transforms: {sorted(SUPPORTED_TARGET_TRANSFORMS)}"
            )

    scaler_path = model_dir / "target_scaler.joblib"
    target_scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    return target_cols, transform_map, target_scaler


def inverse_one_target_transform(values: np.ndarray, kind: str) -> np.ndarray:
    """
    Convert one target from transformed space back to raw physical space.
    """
    values = np.asarray(values, dtype=np.float32)

    if kind == "raw":
        return values

    if kind == "log":
        return np.exp(values)

    if kind == "log1p":
        return np.expm1(values)

    if kind == "signed_log1p":
        return np.sign(values) * np.expm1(np.abs(values))

    raise ValueError(f"Unknown target transform: {kind!r}")


def inverse_target_preprocessing(
    y_model: np.ndarray,
    *,
    target_cols: list[str],
    transform_map: dict[str, str],
    target_scaler: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert model outputs into transformed/unscaled and raw physical target space.

    Returns
    -------
    y_transformed
        Predictions after inverse target scaling, before inverse target transforms.

    y_raw
        Predictions in raw physical target space.
    """
    y_model = np.asarray(y_model, dtype=np.float32)

    if y_model.ndim == 1:
        y_model = y_model.reshape(-1, 1)

    if target_scaler is not None:
        y_transformed = target_scaler.inverse_transform(y_model)
    else:
        y_transformed = y_model.copy()

    y_raw = y_transformed.copy()

    for j, col in enumerate(target_cols):
        kind = transform_map.get(col, "raw")
        y_raw[:, j] = inverse_one_target_transform(y_transformed[:, j], kind)

    return y_transformed.astype(np.float32), y_raw.astype(np.float32)


def predict_multioutput(
    *,
    df: pd.DataFrame,
    model_dir: str | Path,
    device: str | None = None,
    batch_size: int = 4096,
    keep_cols: list[str] | None = None,
    save_model_space: bool = False,
    save_transformed_space: bool = False,
) -> pd.DataFrame:
    model_dir = Path(model_dir)

    model, config, preprocessor = load_model(model_dir, map_location=device or "cpu")

    feature_cols = list(config["feature_cols"])
    target_cols, transform_map, target_scaler = load_target_preprocessing(
        model_dir,
        config,
    )

    if not target_cols:
        output_dim = int(config.get("output_dim", 1))
        target_cols = [f"output_{i}" for i in range(output_dim)]

    print(f"[model] multi-head MLP: {model_dir}")
    print(f"[model] features: {feature_cols}")
    print(f"[model] targets: {target_cols}")

    X_raw = build_X_from_df(df, feature_cols)
    valid_mask = valid_feature_rows(X_raw)

    X_valid = X_raw[valid_mask]

    if preprocessor is not None:
        X_valid = preprocessor.transform(X_valid).astype(np.float32)

    y_model_valid = predict_numpy(
        model,
        X_valid,
        device=device,
        batch_size=batch_size,
    )

    y_transformed_valid, y_raw_valid = inverse_target_preprocessing(
        y_model_valid,
        target_cols=target_cols,
        transform_map=transform_map,
        target_scaler=target_scaler,
    )

    pred_raw = np.full((len(df), len(target_cols)), np.nan, dtype=np.float32)
    pred_transformed = np.full((len(df), len(target_cols)), np.nan, dtype=np.float32)
    pred_model = np.full((len(df), len(target_cols)), np.nan, dtype=np.float32)

    pred_raw[valid_mask, :] = y_raw_valid
    pred_transformed[valid_mask, :] = y_transformed_valid
    pred_model[valid_mask, :] = y_model_valid

    out_parts = [make_base_output_df(df, keep_cols)]

    out_parts.append(
        pd.DataFrame(
            pred_raw,
            index=df.index,
            columns=[f"pred_{target}" for target in target_cols],
        )
    )

    if save_transformed_space:
        out_parts.append(
            pd.DataFrame(
                pred_transformed,
                index=df.index,
                columns=[f"pred_transformed_{target}" for target in target_cols],
            )
        )

    if save_model_space:
        out_parts.append(
            pd.DataFrame(
                pred_model,
                index=df.index,
                columns=[f"pred_model_{target}" for target in target_cols],
            )
        )

    return pd.concat(out_parts, axis=1)


def predict_classreg(
    *,
    df: pd.DataFrame,
    model_dir: str | Path,
    device: str | None = None,
    batch_size: int = 4096,
    keep_cols: list[str] | None = None,
    save_model_space: bool = False,
    save_transformed_space: bool = False,
) -> pd.DataFrame:
    model_dir = Path(model_dir)

    pipeline_config_path = model_dir / "pipeline_config.json"

    with pipeline_config_path.open("r", encoding="utf-8") as f:
        pipeline_config = json.load(f)

    classifier_dir = model_dir / pipeline_config.get("classifier_dir", "classifier")
    regressors_dir = model_dir / pipeline_config.get("regressors_dir", "regressors")

    classifier, classifier_config, classifier_preprocessor = load_model(
        classifier_dir,
        map_location=device or "cpu",
    )

    feature_cols = list(pipeline_config["feature_cols"])
    classifier_feature_cols = list(classifier_config["feature_cols"])

    if classifier_feature_cols != feature_cols:
        raise ValueError(
            "Classifier feature columns do not match pipeline feature columns. "
            f"Pipeline: {feature_cols}. Classifier: {classifier_feature_cols}."
        )

    X_raw = build_X_from_df(df, feature_cols)
    valid_mask = valid_feature_rows(X_raw)

    X_valid_raw = X_raw[valid_mask]
    X_valid_classifier = classifier_preprocessor.transform(X_valid_raw).astype(np.float32)

    pred_class_valid, probs_valid = predict_classes_and_probs(
        classifier,
        X_valid_classifier,
        device=device,
        batch_size=batch_size,
    )

    n_classes = probs_valid.shape[1]

    first_regressor_dir = None

    for class_id in range(n_classes):
        candidate = regressors_dir / f"class_{class_id}"

        if candidate.exists():
            first_regressor_dir = candidate
            break

    if first_regressor_dir is None:
        raise FileNotFoundError(f"No class regressors found in {regressors_dir}")

    _, first_regressor_config, _ = load_model(
        first_regressor_dir,
        map_location=device or "cpu",
    )
    target_cols, _, _ = load_target_preprocessing(
        first_regressor_dir,
        first_regressor_config,
    )

    if not target_cols:
        output_dim = int(first_regressor_config.get("output_dim", 1))
        target_cols = [f"output_{i}" for i in range(output_dim)]

    print(f"[model] class-regression model: {model_dir}")
    print(f"[model] features: {feature_cols}")
    print(f"[model] targets: {target_cols}")

    class_names = pipeline_config.get("class_names")

    if class_names is not None:
        print(f"[model] classes: {class_names}")

    pred_raw = np.full((len(df), len(target_cols)), np.nan, dtype=np.float32)
    pred_transformed = np.full((len(df), len(target_cols)), np.nan, dtype=np.float32)
    pred_model = np.full((len(df), len(target_cols)), np.nan, dtype=np.float32)

    valid_indices = np.where(valid_mask)[0]

    for class_id in range(n_classes):
        regressor_dir = regressors_dir / f"class_{class_id}"

        if not regressor_dir.exists():
            print(f"[warning] missing regressor for class {class_id}; skipping")
            continue

        class_mask = pred_class_valid == class_id

        if class_mask.sum() == 0:
            continue

        regressor, regressor_config, regressor_preprocessor = load_model(
            regressor_dir,
            map_location=device or "cpu",
        )

        regressor_feature_cols = list(regressor_config["feature_cols"])

        if regressor_feature_cols != feature_cols:
            raise ValueError(
                f"Feature columns for class {class_id} do not match the classifier. "
                f"Classifier: {feature_cols}. Regressor: {regressor_feature_cols}."
            )

        regressor_target_cols, transform_map, target_scaler = load_target_preprocessing(
            regressor_dir,
            regressor_config,
        )

        if not regressor_target_cols:
            output_dim = int(regressor_config.get("output_dim", 1))
            regressor_target_cols = [f"output_{i}" for i in range(output_dim)]

        if regressor_target_cols != target_cols:
            raise ValueError(
                f"Target columns for class {class_id} do not match the first regressor. "
                f"First regressor: {target_cols}. "
                f"This regressor: {regressor_target_cols}."
            )

        X_sub_raw = X_valid_raw[class_mask]
        X_sub = regressor_preprocessor.transform(X_sub_raw).astype(np.float32)

        y_model = predict_numpy(
            regressor,
            X_sub,
            device=device,
            batch_size=batch_size,
        )

        y_transformed, y_raw = inverse_target_preprocessing(
            y_model,
            target_cols=target_cols,
            transform_map=transform_map,
            target_scaler=target_scaler,
        )

        row_indices = valid_indices[class_mask]

        pred_raw[row_indices, :] = y_raw
        pred_transformed[row_indices, :] = y_transformed
        pred_model[row_indices, :] = y_model

    class_target = pipeline_config.get("class_target", "class")
    class_col = f"class_{class_target}"

    class_values = np.full(len(df), np.nan, dtype=np.float32)
    class_values[valid_mask] = pred_class_valid

    prob_cols = [f"prob_class_{i}" for i in range(n_classes)]
    prob_values = np.full((len(df), n_classes), np.nan, dtype=np.float32)
    prob_values[valid_mask, :] = probs_valid

    out_parts = [make_base_output_df(df, keep_cols)]

    out_parts.append(
        pd.DataFrame(
            class_values,
            index=df.index,
            columns=[class_col],
        )
    )

    if class_names is not None:
        class_name_values = np.full(len(df), None, dtype=object)

        for class_id, class_name in enumerate(class_names):
            class_name_values[valid_indices[pred_class_valid == class_id]] = class_name

        out_parts.append(
            pd.DataFrame(
                class_name_values,
                index=df.index,
                columns=[f"{class_col}_name"],
            )
        )

    out_parts.append(
        pd.DataFrame(
            prob_values,
            index=df.index,
            columns=prob_cols,
        )
    )

    out_parts.append(
        pd.DataFrame(
            pred_raw,
            index=df.index,
            columns=[f"pred_{target}" for target in target_cols],
        )
    )

    if save_transformed_space:
        out_parts.append(
            pd.DataFrame(
                pred_transformed,
                index=df.index,
                columns=[f"pred_transformed_{target}" for target in target_cols],
            )
        )

    if save_model_space:
        out_parts.append(
            pd.DataFrame(
                pred_model,
                index=df.index,
                columns=[f"pred_model_{target}" for target in target_cols],
            )
        )

    return pd.concat(out_parts, axis=1)