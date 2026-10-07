"""
LSTM Model for Temporal Spectral Disease Prediction
====================================================
2-layer LSTM that processes spectral time-series sequences
and classifies crop health state.

Input:  (batch, timesteps, n_features)
Output: (batch, num_classes)
"""

import torch
import torch.nn as nn


class SpectralLSTM(nn.Module):
    """
    LSTM-based temporal model for spectral sequence classification.

    The model processes a time-series of spectral readings and predicts
    one of three disease classes (Healthy, Early Stress, Active Disease)
    using the final hidden state of a multi-layer LSTM.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep (raw bands + indices).
    hidden_dim : int
        LSTM hidden state dimensionality.
    num_layers : int
        Number of stacked LSTM layers.
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout rate between LSTM layers and in classifier.
    bidirectional : bool
        Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        classifier_input = hidden_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights using Xavier uniform."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, timesteps, input_dim).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        # LSTM forward
        out, (hn, cn) = self.lstm(x)

        # Use final hidden state from last layer
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            final_hidden = torch.cat(
                (hn[-2, :, :], hn[-1, :, :]), dim=1
            )
        else:
            final_hidden = hn[-1, :, :]

        # Classify
        logits = self.classifier(final_hidden)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities via softmax."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
