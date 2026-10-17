"""
CNN-LSTM and CNN-LSTM-Attention Models
======================================
A 1-D CNN over time extracts local spectral-temporal patterns, an LSTM
models the temporal progression, and the sequence is pooled into one
vector for the classifier:

    Input (batch, seq_len, n_features)
    -> Conv1d blocks over the time axis (BatchNorm, ReLU, dropout)
    -> LSTM
    -> pooling:  attention=True   learned temporal attention: a weighted
                                  sum of all LSTM outputs (CNN-LSTM-Attention)
                 attention=False  last LSTM output (plain CNN-LSTM, the ablation)
    -> classifier head

The attention weights say which acquisitions the model relied on; get them
with forward(x, return_attention=True).

Layer widths scale with `lstm_hidden` (conv channels hidden/2 and hidden,
head hidden/2) so that a small configuration really is small. That matters
on the real data, which has a few hundred sequences.
"""

import torch
import torch.nn as nn


def default_conv_channels(lstm_hidden: int) -> list[int]:
    return [max(8, lstm_hidden // 2), lstm_hidden]


class CNNLSTMModel(nn.Module):
    """
    Parameters
    ----------
    n_bands : int
        Input features per timestep.
    seq_len : int
        Timesteps per sequence.
    num_classes : int
        Output classes.
    conv_channels : list[int] or None
        Output channels of each Conv1d block (default: scaled from lstm_hidden).
    lstm_hidden : int
        LSTM hidden size.
    lstm_layers : int
        LSTM layers.
    dropout : float
        Dropout in the head (half of it after each conv block).
    attention : bool
        Temporal attention pooling (True) or last-timestep pooling (False).
    """

    def __init__(
        self,
        n_bands: int = 12,
        seq_len: int = 30,
        num_classes: int = 3,
        conv_channels: list[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        attention: bool = True,
    ):
        super().__init__()
        conv_channels = list(conv_channels or default_conv_channels(lstm_hidden))
        self.n_bands = n_bands
        self.seq_len = seq_len
        self.conv_channels = conv_channels
        self.use_attention = attention

        blocks, in_ch = [], n_bands
        for out_ch in conv_channels:
            blocks += [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                       nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout * 0.5)]
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        self.lstm = nn.LSTM(input_size=conv_channels[-1], hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)

        head = max(8, lstm_hidden // 2)
        self.attention = nn.Sequential(nn.Linear(lstm_hidden, head), nn.Tanh(),
                                       nn.Linear(head, 1)) if attention else None
        self.classifier = nn.Sequential(nn.Linear(lstm_hidden, head), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(head, num_classes))
        self._init_weights()

    def _init_weights(self):
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        x : (batch, seq_len, n_bands) -> logits (batch, num_classes)
        [, attention weights (batch, seq_len); a one-hot on the
        last step when attention is off].
        """
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)                                  # (B, T, H)
        if self.attention is not None:
            weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
            context = (lstm_out * weights).sum(dim=1)
        else:
            context = lstm_out[:, -1]
            weights = torch.zeros_like(lstm_out[..., :1])
            weights[:, -1] = 1.0
        logits = self.classifier(context)
        if return_attention:
            return logits, weights.squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)
