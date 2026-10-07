"""
Performance Tests — Model Quality and API Latency Thresholds
=============================================================
These tests verify that trained models meet minimum performance
requirements. They require trained model checkpoints to exist.
"""

import os
import time

import pytest
import numpy as np


# Skip all tests if no processed data exists
pytestmark = pytest.mark.skipif(
    not os.path.exists('data/processed/X_test.npy'),
    reason="Processed test data not found. Run the pipeline first."
)


class TestModelPerformance:
    """Performance threshold tests for trained models."""

    @pytest.fixture(autouse=True)
    def load_test_data(self):
        """Load test data once for all tests."""
        self.X_test = np.load('data/processed/X_test.npy')
        self.y_test = np.load('data/processed/y_test.npy')
        if os.path.exists('data/processed/X_test_agg.npy'):
            self.X_test_agg = np.load('data/processed/X_test_agg.npy')
            self.y_test_agg = np.load('data/processed/y_test_agg.npy')

    @pytest.mark.skipif(
        not os.path.exists('models/checkpoints/rf_model.joblib'),
        reason="RF model not trained yet"
    )
    def test_rf_accuracy_threshold(self):
        """Random Forest should achieve ≥ 85% accuracy."""
        from src.models.classical_models import load_model
        from sklearn.metrics import accuracy_score

        rf = load_model('models/checkpoints/rf_model.joblib')
        y_pred = rf.predict(self.X_test_agg)
        acc = accuracy_score(self.y_test_agg, y_pred)
        assert acc >= 0.85, f"RF accuracy {acc:.4f} < 0.85 threshold"

    @pytest.mark.skipif(
        not os.path.exists('models/checkpoints/lstm_best.pt'),
        reason="LSTM model not trained yet"
    )
    def test_lstm_accuracy_threshold(self):
        """LSTM should achieve ≥ 85% accuracy."""
        import torch
        from src.models.lstm_model import SpectralLSTM
        from sklearn.metrics import accuracy_score

        device = torch.device('cpu')
        model = SpectralLSTM(input_dim=self.X_test.shape[2], hidden_dim=128,
                              num_layers=2, num_classes=3)
        checkpoint = torch.load('models/checkpoints/lstm_best.pt',
                                 map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        x = torch.tensor(self.X_test, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x)
            y_pred = logits.argmax(dim=1).numpy()

        acc = accuracy_score(self.y_test, y_pred)
        assert acc >= 0.85, f"LSTM accuracy {acc:.4f} < 0.85 threshold"

    @pytest.mark.skipif(
        not os.path.exists('models/checkpoints/lstm_best.pt'),
        reason="LSTM model not trained yet"
    )
    def test_early_stress_f1_threshold(self):
        """Early Stress (class 1) F1 should be ≥ 0.75."""
        import torch
        from src.models.lstm_model import SpectralLSTM
        from sklearn.metrics import f1_score

        device = torch.device('cpu')
        model = SpectralLSTM(input_dim=self.X_test.shape[2], hidden_dim=128,
                              num_layers=2, num_classes=3)
        checkpoint = torch.load('models/checkpoints/lstm_best.pt',
                                 map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        x = torch.tensor(self.X_test, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(x).argmax(dim=1).numpy()

        # Binary F1 for class 1
        y_test_binary = (self.y_test == 1).astype(int)
        y_pred_binary = (y_pred == 1).astype(int)
        f1_class1 = f1_score(y_test_binary, y_pred_binary)

        assert f1_class1 >= 0.75, \
            f"Early Stress F1 {f1_class1:.4f} < 0.75 threshold"

    @pytest.mark.skipif(
        not os.path.exists('models/checkpoints/lstm_best.pt'),
        reason="LSTM model not trained yet"
    )
    def test_active_disease_precision_threshold(self):
        """Active Disease (class 2) precision should be ≥ 0.80."""
        import torch
        from src.models.lstm_model import SpectralLSTM
        from sklearn.metrics import precision_score

        device = torch.device('cpu')
        model = SpectralLSTM(input_dim=self.X_test.shape[2], hidden_dim=128,
                              num_layers=2, num_classes=3)
        checkpoint = torch.load('models/checkpoints/lstm_best.pt',
                                 map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        x = torch.tensor(self.X_test, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(x).argmax(dim=1).numpy()

        y_test_binary = (self.y_test == 2).astype(int)
        y_pred_binary = (y_pred == 2).astype(int)
        prec_class2 = precision_score(y_test_binary, y_pred_binary,
                                       zero_division=0)

        assert prec_class2 >= 0.80, \
            f"Active Disease precision {prec_class2:.4f} < 0.80 threshold"


class TestAPILatency:
    """API response time tests."""

    @pytest.mark.skipif(
        not os.path.exists('models/checkpoints/lstm_best.pt'),
        reason="Model not trained yet"
    )
    def test_api_response_under_200ms(self):
        """API /predict should respond in < 200ms."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)

        payload = {
            "sequence": [[0.35, 0.55, 0.65, 0.80, 0.20, 0.15,
                           0.45, 0.60, 0.90, 0.78, 0.25, 1.03]] * 30
        }

        # Warm-up request
        client.post("/predict", json=payload)

        # Timed request
        start = time.time()
        response = client.post("/predict", json=payload)
        elapsed_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            assert elapsed_ms < 200, \
                f"API response time {elapsed_ms:.1f}ms > 200ms threshold"
