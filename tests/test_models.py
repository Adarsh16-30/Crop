"""
Unit Tests — Deep Learning Models and Factory
"""

import pytest
import torch

from src.models.cnn_lstm_model import CNNLSTMModel
from src.models.factory import build_model, build_model_from_config
from src.models.lstm_model import SpectralLSTM


def _has_gradients(model, x):
    loss = torch.nn.CrossEntropyLoss()(model(x), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    return all(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestSpectralLSTM:

    def test_output_shape(self):
        assert SpectralLSTM(input_dim=12, hidden_dim=64)(torch.randn(8, 30, 12)).shape == (8, 3)

    def test_predict_proba_sums_to_one(self):
        model = SpectralLSTM(input_dim=12, hidden_dim=64).eval()
        probs = model.predict_proba(torch.randn(4, 30, 12))
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_all_parameters_receive_gradients(self):
        assert _has_gradients(SpectralLSTM(input_dim=12, hidden_dim=32), torch.randn(4, 30, 12))

    def test_bidirectional_mode(self):
        model = SpectralLSTM(input_dim=12, hidden_dim=32, bidirectional=True)
        assert model(torch.randn(4, 30, 12)).shape == (4, 3)

    @pytest.mark.parametrize('dim', [6, 12, 20])
    def test_different_input_dims(self, dim):
        assert SpectralLSTM(input_dim=dim, hidden_dim=16)(torch.randn(2, 30, dim)).shape == (2, 3)


class TestCNNLSTMModel:

    def test_output_shape(self):
        assert CNNLSTMModel(n_bands=12, seq_len=30)(torch.randn(8, 30, 12)).shape == (8, 3)

    def test_attention_weights_sum_to_one(self):
        out, attn = CNNLSTMModel(n_bands=12, seq_len=30)(torch.randn(4, 30, 12),
                                                         return_attention=True)
        assert out.shape == (4, 3) and attn.shape == (4, 30)
        assert torch.allclose(attn.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_all_parameters_receive_gradients(self):
        assert _has_gradients(CNNLSTMModel(n_bands=12, seq_len=30), torch.randn(4, 30, 12))

    def test_without_attention_uses_last_step(self):
        model = CNNLSTMModel(n_bands=12, seq_len=5, attention=False)
        assert model.attention is None
        out, weights = model(torch.randn(3, 5, 12), return_attention=True)
        assert out.shape == (3, 3) and torch.equal(weights[:, -1], torch.ones(3))

    def test_widths_scale_with_hidden_size(self):
        small = CNNLSTMModel(n_bands=12, seq_len=5, lstm_hidden=16)
        large = CNNLSTMModel(n_bands=12, seq_len=5, lstm_hidden=128)
        assert small.conv_channels == [8, 16]
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        assert count(small) * 20 < count(large)

    def test_custom_conv_channels(self):
        model = CNNLSTMModel(n_bands=12, seq_len=30, conv_channels=[16, 32, 64])
        assert model(torch.randn(4, 30, 12)).shape == (4, 3)


class TestFactory:

    @pytest.mark.parametrize('model_type,cls,attention', [
        ('lstm', SpectralLSTM, None), ('cnn_lstm', CNNLSTMModel, False),
        ('cnn_lstm_attention', CNNLSTMModel, True)])
    def test_build_model(self, model_type, cls, attention):
        model = build_model(model_type, input_dim=12, hidden_dim=16)
        assert isinstance(model, cls)
        if attention is not None:
            assert model.use_attention is attention

    def test_attention_config_round_trip(self):
        config = {'model_type': 'cnn_lstm_attention', 'input_dim': 12, 'seq_len': 5,
                  'hidden_dim': 16, 'num_layers': 1, 'dropout': 0.5, 'conv_channels': [8, 16]}
        model = build_model_from_config(config)
        build_model_from_config(config).load_state_dict(model.state_dict())
        assert model.use_attention and model.conv_channels == [8, 16]

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            build_model('transformer', input_dim=12)

    def test_config_round_trip_reproduces_architecture(self):
        config = {'model_type': 'lstm', 'input_dim': 12, 'seq_len': 30, 'hidden_dim': 24,
                  'num_layers': 3, 'dropout': 0.1}
        model = build_model_from_config(config)
        assert model.lstm.hidden_size == 24 and model.lstm.num_layers == 3
        clone = build_model_from_config(config)
        clone.load_state_dict(model.state_dict())  # identical parameter shapes
