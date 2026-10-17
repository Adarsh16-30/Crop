"""
Model factory: build any supported model from a type name or from the
`config` dict stored inside a checkpoint.
"""

import torch.nn as nn

from src.constants import NUM_CLASSES
from src.models.cnn_lstm_model import CNNLSTMModel
from src.models.lstm_model import SpectralLSTM

MODEL_TYPES = ('lstm', 'cnn_lstm', 'cnn_lstm_attention')
DISPLAY_NAMES = {'lstm': 'LSTM', 'cnn_lstm': 'CNN-LSTM',
                 'cnn_lstm_attention': 'CNN-LSTM-Attention'}
DEFAULT_NUM_LAYERS = {'lstm': 2, 'cnn_lstm': 1, 'cnn_lstm_attention': 1}


def build_model(
    model_type: str,
    input_dim: int,
    seq_len: int = 30,
    num_classes: int = NUM_CLASSES,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    num_layers: int | None = None,
    conv_channels: list[int] | None = None,
) -> nn.Module:
    """Build a model by type name (see MODEL_TYPES)."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Use one of {MODEL_TYPES}.")
    if num_layers is None:
        num_layers = DEFAULT_NUM_LAYERS[model_type]

    if model_type == 'lstm':
        return SpectralLSTM(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            num_classes=num_classes, dropout=dropout
        )
    return CNNLSTMModel(
        n_bands=input_dim, seq_len=seq_len, num_classes=num_classes,
        conv_channels=conv_channels, lstm_hidden=hidden_dim, lstm_layers=num_layers,
        dropout=dropout, attention=model_type == 'cnn_lstm_attention'
    )


def build_model_from_config(config: dict) -> nn.Module:
    """Rebuild the exact architecture described by a checkpoint's config."""
    return build_model(
        model_type=config['model_type'],
        input_dim=config['input_dim'],
        seq_len=config['seq_len'],
        num_classes=config.get('num_classes', NUM_CLASSES),
        hidden_dim=config['hidden_dim'],
        dropout=config.get('dropout', 0.0),
        num_layers=config.get('num_layers'),
        conv_channels=config.get('conv_channels'),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
