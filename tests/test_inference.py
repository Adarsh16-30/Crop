"""
Inference & Training Tests — checkpoint format, train/serve parity,
calibration, OOD detection, evaluation statistics.
"""

import numpy as np
import pytest
import torch

from src.evaluation.evaluate import bootstrap_ci, compute_metrics, mcnemar_test
from src.inference.pipeline import FeaturePipeline
from src.inference.predictor import Predictor, load_checkpoint_dict
from src.training.train import compute_class_weights, fit_temperature

from .conftest import SEQ_LEN


class TestCheckpoint:

    def test_loads_with_weights_only(self, tiny_checkpoint):
        ckpt = load_checkpoint_dict(tiny_checkpoint)  # would raise on pickled objects
        cfg = ckpt['config']
        assert cfg['model_type'] == 'lstm' and cfg['seq_len'] == SEQ_LEN
        assert cfg['scaler']['feature_names'][-1] == 'CRI'
        assert cfg['temperature'] > 0
        assert set(cfg['versions']) == {'torch', 'numpy', 'sklearn'}
        assert 'val_macro_f1' in ckpt

    def test_predictor_rebuilds_architecture_from_config(self, tiny_checkpoint):
        predictor = Predictor(tiny_checkpoint, device=torch.device('cpu'))
        assert predictor.model.lstm.hidden_size == 16
        assert predictor.seq_len == SEQ_LEN


class TestTrainServeParity:

    def test_raw_api_path_equals_offline_path(self, tiny_checkpoint, tiny_data,
                                              raw_test_sequences):
        """The bug this guards against: serving raw values to a model trained on scaled ones."""
        predictor = Predictor(tiny_checkpoint, device=torch.device('cpu'))
        offline = predictor.predict_proba_scaled(tiny_data['X_test'])
        online = np.array([r['probabilities'] for r in predictor.predict_raw(raw_test_sequences)])
        np.testing.assert_allclose(online, offline, atol=1e-5)

    def test_feature_pipeline_matches_preprocessed_arrays(self, tiny_data, raw_test_sequences):
        pipeline = FeaturePipeline.from_dict(tiny_data['scaler_params'])
        np.testing.assert_allclose(pipeline.transform_raw(raw_test_sequences),
                                   tiny_data['X_test'], atol=1e-4)


class TestInputValidation:

    @pytest.fixture
    def predictor(self, tiny_checkpoint):
        return Predictor(tiny_checkpoint, device=torch.device('cpu'))

    def test_wrong_length_rejected(self, predictor):
        with pytest.raises(ValueError, match='exactly'):
            predictor.predict_raw(np.full((SEQ_LEN + 5, 6), 0.4))

    def test_wrong_band_count_rejected(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict_raw(np.full((SEQ_LEN, 5), 0.4))

    def test_nan_rejected(self, predictor):
        x = np.full((SEQ_LEN, 6), 0.4)
        x[3, 2] = np.nan
        with pytest.raises(ValueError):
            predictor.predict_raw(x)

    def test_absurd_input_flagged_out_of_distribution(self, predictor, raw_test_sequences):
        normal = predictor.predict_raw(raw_test_sequences[0])[0]
        weird = np.tile([0.01, 0.99, 0.01, 0.99, 0.99, 0.01], (SEQ_LEN, 1))
        assert not normal['out_of_distribution']
        assert predictor.predict_raw(weird)[0]['out_of_distribution']

    def test_result_fields(self, predictor, raw_test_sequences):
        r = predictor.predict_raw(raw_test_sequences[0])[0]
        assert sum(r['probabilities']) == pytest.approx(1.0, abs=1e-5)
        assert 0 <= r['risk_score'] <= 1
        assert r['confidence'] == max(r['probabilities'])


class TestTrainingHelpers:

    def test_class_weights_from_true_labels(self):
        w = compute_class_weights(np.array([0] * 6 + [1] * 3 + [2] * 1), torch.device('cpu'))
        np.testing.assert_allclose(w.numpy(), [10 / 18, 10 / 9, 10 / 3], rtol=1e-6)

    def test_temperature_fixes_overconfidence(self):
        torch.manual_seed(0)
        labels = torch.randint(0, 3, (500,))
        logits = torch.randn(500, 3)
        logits[torch.arange(500), labels] += 1.0
        logits *= 8  # badly overconfident
        T = fit_temperature(logits, labels)
        nll = torch.nn.CrossEntropyLoss()
        assert T > 1
        assert nll(logits / T, labels) < nll(logits, labels)


class TestEvaluationStats:

    def test_bootstrap_ci_contains_point_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 300)
        pred = np.where(rng.random(300) < 0.8, y, (y + 1) % 3)
        lo, hi = bootstrap_ci(y, pred)
        assert lo < (pred == y).mean() < hi

    def test_mcnemar(self):
        y = np.zeros(100, dtype=int)
        a = np.zeros(100, dtype=int)
        b = np.zeros(100, dtype=int)
        b[:30] = 1  # b wrong on 30 samples a gets right
        assert mcnemar_test(y, a, b)['p_value'] < 0.001
        assert mcnemar_test(y, a, a)['p_value'] == 1.0

    def test_compute_metrics_has_cis(self):
        y = np.array([0, 1, 2] * 20)
        m = compute_metrics(y, y, None, 'perfect')
        assert m['accuracy'] == 1.0 and m['accuracy_ci_low'] == 1.0
        assert m['f1_early_stress'] == 1.0
