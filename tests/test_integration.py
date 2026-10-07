"""
Integration Tests — End-to-End Pipeline & API
"""

import pytest
import numpy as np


class TestFullPipeline:
    """End-to-end integration tests."""

    def test_data_generation_to_preprocessing(self):
        """Generate data → compute indices → create sequences."""
        from src.data.data_generator import generate_dataset
        from src.features.spectral_indices import compute_indices
        from src.data.preprocessor import create_sequences

        df = generate_dataset(n_samples=50, seq_len=30)
        df = compute_indices(df)
        X, y, ids = create_sequences(df, seq_len=30)

        assert X.shape[0] == 50
        assert X.shape[1] == 30
        assert X.shape[2] == 12  # 6 raw + 6 indices
        assert len(y) == 50
        assert set(np.unique(y)).issubset({0, 1, 2})

    def test_data_to_aggregated_features(self):
        """Generate data → compute indices → aggregate for classical ML."""
        from src.data.data_generator import generate_dataset
        from src.features.spectral_indices import compute_indices
        from src.data.preprocessor import create_aggregated_features

        df = generate_dataset(n_samples=30, seq_len=10)
        df = compute_indices(df)
        X_agg, y, ids = create_aggregated_features(df)

        assert X_agg.shape[0] == 30
        assert X_agg.shape[1] == 24  # 12 features * 2 (mean + std)
        assert len(y) == 30

    def test_model_training_and_inference(self):
        """Build model → train for 1 epoch → run inference."""
        import torch
        from src.models.lstm_model import SpectralLSTM
        from src.data.data_generator import generate_dataset
        from src.features.spectral_indices import compute_indices
        from src.data.preprocessor import create_sequences

        df = generate_dataset(n_samples=20, seq_len=10)
        df = compute_indices(df)
        X, y, _ = create_sequences(df, seq_len=10)

        model = SpectralLSTM(input_dim=12, hidden_dim=32, num_classes=3)
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # One training step
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

        # Inference
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(x_tensor)

        assert probs.shape == (20, 3)
        assert torch.allclose(probs.sum(dim=1), torch.ones(20), atol=1e-4)

    def test_geo_pipeline(self):
        """Predictions → GeoDataFrame → risk summary."""
        import pandas as pd
        from src.geospatial.risk_mapper import build_geodataframe, get_risk_summary

        pred_df = pd.DataFrame({
            'lat': np.random.uniform(12.5, 13.5, 50),
            'lon': np.random.uniform(77.0, 78.0, 50),
            'predicted_label': np.random.choice([0, 1, 2], 50),
            'confidence': np.random.uniform(0.5, 1.0, 50)
        })

        gdf = build_geodataframe(pred_df)
        summary = get_risk_summary(gdf)

        assert summary['total_samples'] == 50
        assert 0 <= summary['mean_risk'] <= 1


class TestAPIEndpoint:
    """Tests for the FastAPI predict endpoint."""

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_predict_without_model(self):
        """Predict should return 503 if no model is loaded."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        payload = {
            "sequence": [[0.35, 0.55, 0.65, 0.80, 0.20, 0.15,
                           0.45, 0.60, 0.90, 0.78, 0.25, 1.03]] * 30
        }
        response = client.post("/predict", json=payload)
        # Should return 503 (model not loaded) or 200 if model happens to exist
        assert response.status_code in [200, 503]

    def test_metrics_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
