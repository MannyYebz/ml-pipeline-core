"""Feedforward neural network for binary churn classification."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

from src.config import CONFIG, ModelConfig


class ChurnClassifier(nn.Module):
    """Feedforward binary classifier with batch norm and dropout.

    The network is a stack of ``Linear -> BatchNorm1d -> ReLU -> Dropout``
    blocks ending in a single-logit linear head. The forward pass returns raw
    logits; use :meth:`predict_proba` to obtain sigmoid probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        dropout: float | None = None,
        use_batch_norm: bool | None = None,
    ) -> None:
        """Build the network.

        Args:
            input_dim: Number of input features (matches ``Preprocessor.output_dim``).
            hidden_dims: Sizes of the hidden layers. Must contain at least two
                values so the final architecture has at least three layers
                (two hidden + one output). Defaults to ``CONFIG.model.hidden_dims``.
            dropout: Dropout probability applied after each hidden block.
                Defaults to ``CONFIG.model.dropout``.
            use_batch_norm: Toggle batch normalization in hidden layers.
                Defaults to ``CONFIG.model.use_batch_norm``.

        Raises:
            ValueError: If ``hidden_dims`` has fewer than 2 entries.
        """
        super().__init__()
        cfg: ModelConfig = CONFIG.model
        hidden = list(hidden_dims if hidden_dims is not None else cfg.hidden_dims)
        drop_p = cfg.dropout if dropout is None else dropout
        bn = cfg.use_batch_norm if use_batch_norm is None else use_batch_norm

        if len(hidden) < 2:
            raise ValueError(
                "hidden_dims must define at least 2 hidden layers "
                f"(received {len(hidden)})"
            )

        self.input_dim: int = int(input_dim)
        self.hidden_dims: List[int] = hidden
        self.dropout: float = float(drop_p)
        self.use_batch_norm: bool = bool(bn)

        layers: List[nn.Module] = []
        prev = self.input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if drop_p > 0.0:
                layers.append(nn.Dropout(p=drop_p))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming-uniform initialization to all ``Linear`` layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw output logits.

        Args:
            x: Input batch of shape ``(batch, input_dim)``.

        Returns:
            Logits of shape ``(batch,)``.
        """
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid probabilities for a batch.

        Args:
            x: Input batch of shape ``(batch, input_dim)``.

        Returns:
            Probabilities in ``[0, 1]`` of shape ``(batch,)``.
        """
        self.eval()
        return torch.sigmoid(self.forward(x))

    def config_dict(self) -> dict:
        """Serialize the architecture for checkpointing.

        Returns:
            A JSON-serializable dictionary that can be passed back to
            :meth:`from_config` to reconstruct an identically-shaped network.
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
        }

    @classmethod
    def from_config(cls, config: dict) -> "ChurnClassifier":
        """Construct a model from a dictionary produced by :meth:`config_dict`.

        Args:
            config: Architecture dictionary.

        Returns:
            A new ``ChurnClassifier`` with the requested shape.
        """
        return cls(
            input_dim=int(config["input_dim"]),
            hidden_dims=list(config["hidden_dims"]),
            dropout=float(config["dropout"]),
            use_batch_norm=bool(config["use_batch_norm"]),
        )
