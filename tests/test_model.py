"""Unit tests for the PyTorch model and supporting components."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dataset import ChurnDataset, build_dataloader
from src.evaluate import evaluate
from src.model import ChurnClassifier


@pytest.fixture()
def model() -> ChurnClassifier:
    """A small ``ChurnClassifier`` suitable for fast unit tests."""
    return ChurnClassifier(input_dim=20, hidden_dims=[16, 8, 4], dropout=0.2)


def test_forward_output_shape_matches_batch(model: ChurnClassifier) -> None:
    """``forward`` should return logits of shape ``(batch,)``."""
    x = torch.randn(8, model.input_dim)
    logits = model(x)
    assert logits.shape == (8,)


def test_layer_count_meets_minimum() -> None:
    """The network must contain at least three ``Linear`` layers."""
    net = ChurnClassifier(input_dim=10, hidden_dims=[8, 8])
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) >= 3


def test_dropout_and_batchnorm_present() -> None:
    """Regularization layers should be wired into the default architecture."""
    net = ChurnClassifier(input_dim=10, hidden_dims=[8, 8], dropout=0.3, use_batch_norm=True)
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in net.modules())
    has_bn = any(isinstance(m, torch.nn.BatchNorm1d) for m in net.modules())
    assert has_dropout and has_bn


def test_predict_proba_in_unit_interval(model: ChurnClassifier) -> None:
    """``predict_proba`` should always return values in ``[0, 1]``."""
    x = torch.randn(32, model.input_dim)
    probs = model.predict_proba(x)
    assert probs.shape == (32,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_config_roundtrip_preserves_architecture(model: ChurnClassifier) -> None:
    """``from_config(model.config_dict())`` must reproduce the same architecture."""
    cfg = model.config_dict()
    rebuilt = ChurnClassifier.from_config(cfg)
    assert rebuilt.config_dict() == cfg
    assert sum(p.numel() for p in rebuilt.parameters()) == sum(
        p.numel() for p in model.parameters()
    )


def test_invalid_hidden_dims_rejected() -> None:
    """A single-layer architecture should be rejected at construction."""
    with pytest.raises(ValueError):
        ChurnClassifier(input_dim=10, hidden_dims=[8])


def test_state_dict_roundtrip(model: ChurnClassifier) -> None:
    """Weights should survive a save/load round-trip via ``state_dict``."""
    sd = model.state_dict()
    other = ChurnClassifier.from_config(model.config_dict())
    other.load_state_dict(sd)
    x = torch.randn(4, model.input_dim)
    model.eval()
    other.eval()
    torch.testing.assert_close(model(x), other(x))


def test_dataset_indexing_and_length() -> None:
    """The dataset should expose ``__len__`` and ``__getitem__`` correctly."""
    X = np.random.randn(10, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=10)
    ds = ChurnDataset(X, y)
    assert len(ds) == 10
    feat, label = ds[0]
    assert feat.shape == (4,)
    assert label.dtype == torch.float32


def test_dataset_rejects_length_mismatch() -> None:
    """Mismatched feature/label arrays should raise at construction."""
    with pytest.raises(ValueError):
        ChurnDataset(np.zeros((5, 3), dtype=np.float32), np.zeros(4, dtype=np.int64))


def test_model_can_overfit_tiny_batch() -> None:
    """The training step should be able to drive loss down on a toy batch.

    This is a smoke test that catches gross wiring bugs (e.g. detached graph,
    inverted labels) without depending on real data.
    """
    torch.manual_seed(0)
    net = ChurnClassifier(input_dim=8, hidden_dims=[16, 8], dropout=0.0, use_batch_norm=False)
    x = torch.randn(64, 8)
    w = torch.randn(8)
    y = (x @ w > 0).float()

    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    initial = float(loss_fn(net(x), y).item())
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
    final = float(loss_fn(net(x), y).item())
    assert final < initial * 0.5, f"Loss did not decrease meaningfully ({initial} -> {final})"


def test_evaluate_returns_full_metric_suite() -> None:
    """``evaluate`` should populate every metric field with finite values."""
    torch.manual_seed(0)
    X = np.random.randn(48, 6).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    loader = build_dataloader(ChurnDataset(X, y), batch_size=16, shuffle=False)

    net = ChurnClassifier(input_dim=6, hidden_dims=[8, 8], dropout=0.0, use_batch_norm=False)
    result = evaluate(net, loader, torch.device("cpu"))

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert 0.0 <= result.auc_roc <= 1.0
    assert result.loss >= 0.0
    assert len(result.confusion_matrix) == 2 and len(result.confusion_matrix[0]) == 2
