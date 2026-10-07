"""
CNN-LSTM Hybrid Model for Spectral Disease Detection
=====================================================
1D CNN extracts local spectral patterns within each timestep,
LSTM captures temporal progression of disease indicators.

Architecture:
    Input (batch, seq_len, n_bands)
    → Permute to (batch, n_bands, seq_len)
    → Conv1d layers for spectral feature extraction
    → Permute back to (batch, seq_len, conv_features)
    → LSTM for temporal modeling
    → Linear classifier
"""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model combining convolutional spectral feature
    extraction with recurrent temporal modeling.

    Parameters
    ----------
    n_bands : int
        Number of input spectral features per timestep.
    seq_len : int
        Number of timesteps in input sequence.
    num_classes : int
        Number of output disease classes.
    conv_channels : list[int]
        Number of output channels for each Conv1d layer.
    lstm_hidden : int
        LSTM hidden state size.
    lstm_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_bands: int = 12,
        seq_len: int = 30,
        num_classes: int = 3,
        conv_channels: list[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]

        self.n_bands = n_bands
        self.seq_len = seq_len

        # Build CNN layers dynamically
        cnn_layers = []
        in_channels = n_bands
        for out_channels in conv_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            ])
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Attention mechanism (optional — improves focus on critical timesteps)
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for CNN and classifier layers."""
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                         nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, n_bands).
        return_attention : bool
            If True, also return attention weights.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        torch.Tensor (optional)
            Attention weights of shape (batch, seq_len) if return_attention=True.
        """
        batch_size = x.size(0)

        # CNN: extract spectral features at each timestep
        # (batch, seq_len, n_bands) → (batch, n_bands, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # (batch, conv_out, seq_len) → (batch, seq_len, conv_out)
        x = x.permute(0, 2, 1)

        # LSTM: capture temporal progression
        lstm_out, (hn, cn) = self.lstm(x)
        # lstm_out: (batch, seq_len, lstm_hidden)

        # Temporal attention
        attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, lstm_hidden)

        # Classify
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights.squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities via softmax."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
