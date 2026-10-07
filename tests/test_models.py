"""
Unit Tests — Deep Learning Models
"""

import pytest
import torch
import numpy as np

from src.models.lstm_model import SpectralLSTM
from src.models.cnn_lstm_model import CNNLSTMModel


class TestSpectralLSTM:
    """Tests for the SpectralLSTM model."""

    def test_output_shape(self):
        model = SpectralLSTM(input_dim=12, hidden_dim=64, num_classes=3)
        x = torch.randn(8, 30, 12)  # batch=8, seq=30, features=12
        out = model(x)
        assert out.shape == (8, 3)

    def test_single_sample(self):
        model = SpectralLSTM(input_dim=12, hidden_dim=64, num_classes=3)
        x = torch.randn(1, 30, 12)
        out = model(x)
        assert out.shape == (1, 3)

    def test_no_nan_in_output(self):
        model = SpectralLSTM(input_dim=12, hidden_dim=128, num_classes=3)
        x = torch.randn(4, 30, 12)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_predict_proba_sums_to_one(self):
        model = SpectralLSTM(input_dim=12, hidden_dim=64, num_classes=3)
        model.eval()
        x = torch.randn(4, 30, 12)
        probs = model.predict_proba(x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_gradient_flow(self):
        model = SpectralLSTM(input_dim=12, hidden_dim=64, num_classes=3)
        x = torch.randn(4, 30, 12)
        y = torch.tensor([0, 1, 2, 0])
        out = model(x)
        loss = torch.nn.CrossEntropyLoss()(out, y)
        loss.backward()
        # Check that at least some gradients are non-zero
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_bidirectional_mode(self):
        model = SpectralLSTM(input_dim=12, hidden_dim=64, num_classes=3,
                              bidirectional=True)
        x = torch.randn(4, 30, 12)
        out = model(x)
        assert out.shape == (4, 3)

    def test_different_input_dims(self):
        for dim in [6, 10, 12, 20]:
            model = SpectralLSTM(input_dim=dim, hidden_dim=32, num_classes=3)
            x = torch.randn(2, 30, dim)
            out = model(x)
            assert out.shape == (2, 3)


class TestCNNLSTMModel:
    """Tests for the CNN-LSTM hybrid model."""

    def test_output_shape(self):
        model = CNNLSTMModel(n_bands=12, seq_len=30, num_classes=3)
        x = torch.randn(8, 30, 12)
        out = model(x)
        assert out.shape == (8, 3)

    def test_single_sample(self):
        model = CNNLSTMModel(n_bands=12, seq_len=30, num_classes=3)
        x = torch.randn(1, 30, 12)
        out = model(x)
        assert out.shape == (1, 3)

    def test_no_nan_in_output(self):
        model = CNNLSTMModel(n_bands=12, seq_len=30, num_classes=3)
        x = torch.randn(4, 30, 12)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_attention_weights(self):
        model = CNNLSTMModel(n_bands=12, seq_len=30, num_classes=3)
        x = torch.randn(4, 30, 12)
        out, attn = model(x, return_attention=True)
        assert out.shape == (4, 3)
        assert attn.shape == (4, 30)
        # Attention weights should sum to ~1 per sample
        sums = attn.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_gradient_flow(self):
        model = CNNLSTMModel(n_bands=12, seq_len=30, num_classes=3)
        x = torch.randn(4, 30, 12)
        y = torch.tensor([0, 1, 2, 0])
        out = model(x)
        loss = torch.nn.CrossEntropyLoss()(out, y)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_custom_conv_channels(self):
        model = CNNLSTMModel(n_bands=12, seq_len=30, num_classes=3,
                              conv_channels=[16, 32, 64])
        x = torch.randn(4, 30, 12)
        out = model(x)
        assert out.shape == (4, 3)

    def test_predict_proba(self):
        model = CNNLSTMModel(n_bands=12, seq_len=30, num_classes=3)
        model.eval()
        x = torch.randn(2, 30, 12)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 3)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
