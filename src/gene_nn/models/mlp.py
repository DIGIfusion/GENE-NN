from __future__ import annotations

import json
from pathlib import Path

import joblib
import torch
import torch.nn as nn


def _make_hidden_layers(
    input_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    dropout: float,
) -> tuple[list[nn.Module], int]:
    layers = []
    dims = [int(input_dim)] + [int(h) for h in hidden_dims]

    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())

        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))

    return layers, dims[-1]


class MLP(nn.Module):
    """
    Standard feed-forward MLP.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (64, 32, 16),
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_layers, last_dim = _make_hidden_layers(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=float(dropout),
        )

        self.network = nn.Sequential(
            *hidden_layers,
            nn.Linear(last_dim, int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiHeadMLP(nn.Module):
    """
    MLP with a shared trunk and one regression head per target.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        head_hidden_dims: tuple[int, ...] | list[int] = (32,),
        dropout: float = 0.0,
        target_cols: list[str] | None = None,
    ):
        super().__init__()

        self.output_dim = int(output_dim)
        self.target_cols = target_cols

        trunk_layers, trunk_out_dim = _make_hidden_layers(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=float(dropout),
        )

        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.ModuleList()

        for _ in range(self.output_dim):
            head_layers, head_out_dim = _make_hidden_layers(
                input_dim=trunk_out_dim,
                hidden_dims=head_hidden_dims,
                dropout=float(dropout),
            )

            self.heads.append(
                nn.Sequential(
                    *head_layers,
                    nn.Linear(head_out_dim, 1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        outputs = [head(z) for head in self.heads]
        return torch.cat(outputs, dim=1)


def load_model(
    model_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    """
    Load a saved MLP or MultiHeadMLP model bundle.

    Expected files:
        model.pt
        model_config.json
        preprocessor.joblib

    Returns
    -------
    model
        Loaded PyTorch model in eval mode.

    config
        Model configuration dictionary.

    preprocessor
        Input preprocessor loaded from preprocessor.joblib.
    """
    model_dir = Path(model_dir)

    config_path = model_dir / "model_config.json"
    weights_path = model_dir / "model.pt"
    preprocessor_path = model_dir / "preprocessor.joblib"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_arch = config.get("model_arch", "mlp")

    if model_arch == "mlp":
        model = MLP(
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            hidden_dims=tuple(config["hidden_dims"]),
            dropout=float(config.get("dropout", 0.0)),
        )

    elif model_arch == "multihead_mlp":
        model = MultiHeadMLP(
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            hidden_dims=tuple(config["hidden_dims"]),
            head_hidden_dims=tuple(config.get("head_hidden_dims", [32])),
            dropout=float(config.get("dropout", 0.0)),
            target_cols=config.get("target_cols"),
        )

    else:
        raise ValueError(f"Unknown model_arch in model_config.json: {model_arch!r}")

    state_dict = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()

    preprocessor = joblib.load(preprocessor_path)

    return model, config, preprocessor