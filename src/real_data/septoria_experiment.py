"""
Septoria experiment: nested, pot-grouped cross-validation
=========================================================
Twelve pots are far too few for a fixed train / validation / test split
(a 2-pot test set would be pure luck), so every model is evaluated by an
outer pot-grouped CV and tuned only inside each outer training fold:

    outer loop  StratifiedGroupKFold over pots -> held-out pots are scored once
    inner loop  src.training.tuning (grouped CV on the training pots):
                RF / XGBoost grids; LSTM / CNN-LSTM / CNN-LSTM-Attention size,
                regularization and epoch count; out-of-fold temperature
    per fold    feature scaler fitted on the training pots only; QUBO / top-K
                selection on the training pots only

Reported (reports/septoria/):
    model_comparison.csv     out-of-fold metrics, 95% CIs by resampling POTS
    tuning.csv               every inner configuration of every outer fold
    feature_selection.csv    QUBO vs. top-K vs. all hyperspectral features
    presymptomatic_test.csv  controls vs. inoculated-but-not-yet-symptomatic
                             leaves only, with a pot-level permutation test
    fold_predictions.csv     every out-of-fold prediction
"""

import argparse
import json
import logging
import os
import tempfile

import numpy as np
import pandas as pd

from src.constants import NUM_CLASSES
from src.data.preprocessor import create_aggregated_features, create_sequences
from src.features.spectral_indices import compute_indices, get_feature_columns
from src.real_data import septoria as sep
from src.real_data.common import reports_dir
from src.training import tuning

logger = logging.getLogger(__name__)

SEQUENCES_CSV = os.path.join('data', 'real', 'septoria', 'septoria_sequences.csv')
# Fixed, strongly regularized RF for the permutation test (tuning inside every
# permutation would cost ~50k forest fits; the SAME pipeline scores the real
# and the permuted labels, which is what makes the test valid).
PRESYMPTOMATIC_RF = {'max_depth': 3, 'min_samples_leaf': 5, 'max_features': 'sqrt'}
DL_MODELS = ('lstm', 'cnn_lstm', 'cnn_lstm_attention')
# Smaller grids than the CDM run: ~150 training windows per outer fold.
RF_GRID = {'max_depth': [2, 3, 5], 'min_samples_leaf': [3, 8], 'max_features': ['sqrt', 0.5]}
XGB_GRID = {'max_depth': [1, 2], 'learning_rate': [0.05], 'n_estimators': [50, 150],
            'min_child_weight': [2, 5], 'reg_lambda': [1.0, 10.0]}
DL_GRID = {'hidden_dim': [8, 16], 'dropout': [0.3, 0.5], 'weight_decay': [1e-2, 1e-1],
           'noise_std': [0.1]}
# Per-leaf hyperspectral features (beyond the six project bands) offered to
# the classical models and the feature selectors.
EXTRA_FEATURES = ['R450', 'R531', 'R550', 'R570', 'R670', 'R680', 'R715', 'R725', 'R760',
                  'R900', 'R970', 'R1000', 'R1200', 'R1450', 'R1650', 'R1940', 'R2100',
                  'R2300', 'ndvi_p10', 'ndvi_std', 'ndre_p10', 'ndre_std']
# SWIR-only runs (VNIR not extracted yet): the SWIR project bands, the other
# SWIR bands, the project SWIR ratio and the depths of the two water bands.
SWIR_FEATURES = ['swir_1610', 'swir_2190', 'R1000', 'R1200', 'R1450', 'R1650', 'R1940',
                 'R2100', 'R2300', 'SWIR_ratio', 'water_1450', 'water_1940']


def feature_sets(sensors: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """(per-step features for the sequence models, pool for classical models / selectors)."""
    if set(sensors) == {'VNIR', 'SWIR'}:
        return get_feature_columns(), get_feature_columns() + EXTRA_FEATURES
    if tuple(sensors) == ('SWIR',):
        return SWIR_FEATURES, SWIR_FEATURES
    raise ValueError(f"Unsupported camera set {sensors}: use ('SWIR',) or both")


def add_derived(df: pd.DataFrame, sensors: tuple[str, ...]) -> pd.DataFrame:
    """Indices computed per row from the extracted bands."""
    df = df.copy()
    if 'VNIR' in sensors:
        df = compute_indices(df, inplace=True)
    df['SWIR_ratio'] = df['swir_1610'] / df['swir_2190'].clip(lower=1e-3)
    df['water_1450'] = df['R1450'] / df['R1200'].clip(lower=1e-3)
    df['water_1940'] = df['R1940'] / df['R1650'].clip(lower=1e-3)
    return df


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def run_dir(sensors) -> str:
    """reports/septoria/<vnir_swir|swir>/: runs with different cameras never overwrite."""
    return reports_dir('septoria', '_'.join(s.lower() for s in sep.SENSORS if s in sensors))


def build_dataset(seq_len: int = 4, sensors=sep.SENSORS, out: str | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """observations, onsets, sequences (long format); also written to disk."""
    obs = sep.build_observations(sensors=sensors)
    if obs.empty:
        raise FileNotFoundError("No extracted leaf spectra; run python -m src.real_data "
                                "septoria-fetch first")
    onsets = sep.visible_onset(obs)
    state = sep.observation_state(obs, onsets)
    normal = sep.visibly_normal(obs, onsets)
    seqs = sep.build_leaf_sequences(obs, state, seq_len=seq_len, normal=normal)
    out = out or run_dir(sensors)
    obs.assign(state=state, visibly_normal=normal).to_csv(
        os.path.join(out, 'observations.csv'), index=False)
    onsets.to_csv(os.path.join(out, 'symptom_onset.csv'), index=False)
    flags = onsets.attrs['flags']
    flags.to_csv(os.path.join(out, 'symptom_flags.csv'), index=False)
    # How often each group looks abnormal per date: the controls' rate is the
    # non-disease change (senescence, handling damage) the rule cannot separate.
    flags.groupby(['dpi', 'infected'])['abnormal'].mean().unstack().rename(
        columns={False: 'control_abnormal_rate', True: 'inoculated_abnormal_rate'}).to_csv(
        os.path.join(out, 'abnormal_rate_by_date.csv'))
    os.makedirs(os.path.dirname(SEQUENCES_CSV), exist_ok=True)
    seqs.to_csv(SEQUENCES_CSV, index=False)
    return obs.assign(state=state), onsets, seqs


def arrays(seqs: pd.DataFrame, seq_len: int, sensors=sep.SENSORS) -> dict:
    """Unscaled per-step features, labels, pots and ids, one row per window."""
    primary, pool = feature_sets(tuple(sensors))
    df = add_derived(seqs, tuple(sensors))
    X, y, ids = create_sequences(df, seq_len, primary)
    Xh, _, _ = create_sequences(df, seq_len, pool, sample_ids=ids)
    meta = df.groupby('sample_id').first().loc[ids]
    return {'X': X, 'Xh': Xh, 'y': y, 'ids': ids, 'pot': meta['site_id'].to_numpy(),
            'end_dpi': df.groupby('sample_id')['dpi'].last().loc[ids].to_numpy(),
            'treatment': meta['treatment'].to_numpy(), 'project': primary, 'hyper': pool,
            'normal': meta['window_normal'].astype(bool).to_numpy()}


def scale(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Standardize per feature with statistics of the training windows only."""
    flat = train.reshape(-1, train.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return [((a - mean) / std).astype(np.float32) for a in (train, *others)]


def full_proba(model, X: np.ndarray) -> np.ndarray:
    """predict_proba with one column per class, even if a class was absent in training."""
    out = np.zeros((len(X), NUM_CLASSES))
    out[:, model.classes_.astype(int)] = model.predict_proba(X)
    return out


def agg_names(features: list[str]) -> list[str]:
    return [f'{f}_{s}' for s in ('mean', 'std', 'slope', 'delta') for f in features]


# ---------------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------------

def outer_folds(y, pots, n_splits, seed):
    return tuning.grouped_folds(y, pots, n_splits=n_splits, seed=seed)


def run(seq_len: int = 4, n_outer: int = 6, n_inner: int = 4, max_features: int = 8,
        epochs: int = 80, patience: int = 12, seed: int = 42, models=DL_MODELS,
        n_permutations: int = 200, sensors=sep.SENSORS) -> dict:
    import torch

    from src.models.classical_models import fit, make_random_forest, make_xgboost
    from src.quantum.integration import select_qubo, select_top_k
    from src.training.train import fit_temperature, load_checkpoint, train_from_arrays

    out_dir = run_dir(sensors)
    _, onsets, seqs = build_dataset(seq_len, sensors=sensors, out=out_dir)
    d = arrays(seqs, seq_len, sensors=sensors)
    y, pots = d['y'], d['pot']
    logger.info("Septoria: %d windows from %d leaves / %d pots | classes %s",
                len(y), seqs['leaf_id'].nunique(), len(np.unique(pots)),
                np.bincount(y, minlength=NUM_CLASSES).tolist())
    device = torch.device('cpu')
    preds = {name: np.full(len(y), -1) for name in
             ('Random Forest', 'XGBoost', 'RF (hyperspectral)', 'RF (QUBO 8)', 'RF (top-K 8)',
              *[m for m in models])}
    probs = {name: np.zeros((len(y), NUM_CLASSES)) for name in preds}
    fold_of = np.full(len(y), -1)
    tables, selections = [], []

    for k, (tr, te) in enumerate(outer_folds(y, pots, n_outer, seed)):
        logger.info("Outer fold %d: test pots %s", k, sorted(set(pots[te])))
        fold_of[te] = k
        Xtr, Xte = scale(d['X'][tr], d['X'][te])
        Htr, Hte = scale(d['Xh'][tr], d['Xh'][te])
        Atr, Ate = create_aggregated_features(Xtr), create_aggregated_features(Xte)
        HAtr, HAte = create_aggregated_features(Htr), create_aggregated_features(Hte)

        # classical, project features
        for name, factory, grid in (('Random Forest', make_random_forest, RF_GRID),
                                    ('XGBoost', make_xgboost, XGB_GRID)):
            params, table = tuning.tune_classical(name, factory, grid, Atr, y[tr], pots[tr],
                                                  n_splits=n_inner, seed=seed)
            tables.append(table.assign(outer_fold=k))
            model = fit(factory(seed=seed, **params), Atr, y[tr])
            probs[name][te] = full_proba(model, Ate)
        # classical, hyperspectral pool + feature selection (selection on train pots only)
        params, table = tuning.tune_classical('Random Forest', make_random_forest, RF_GRID,
                                              HAtr, y[tr], pots[tr], n_splits=n_inner, seed=seed)
        tables.append(table.assign(model='RF (hyperspectral)', outer_fold=k))
        rf_all = fit(make_random_forest(seed=seed, **params), HAtr, y[tr])
        probs['RF (hyperspectral)'][te] = full_proba(rf_all, HAte)
        qubo = select_qubo(HAtr, y[tr], max_features, rf_params=params, seed=seed)
        topk = select_top_k(rf_all.feature_importances_, max_features)
        for name, idx in (('RF (QUBO 8)', qubo), ('RF (top-K 8)', topk)):
            m = fit(make_random_forest(seed=seed, **params), HAtr[:, idx], y[tr])
            probs[name][te] = full_proba(m, HAte[:, idx])
            selections.append({'outer_fold': k, 'method': name,
                               'features': [agg_names(d['hyper'])[i] for i in idx]})

        # sequence models
        scaler = {'mean': [0.0] * Xtr.shape[2], 'scale': [1.0] * Xtr.shape[2],
                  'feature_names': d['project'], 'seq_len': seq_len}
        for mt in models:
            params, table, info = tuning.tune_sequence_model(
                mt, Xtr, y[tr], pots[tr], scaler, grid=DL_GRID, n_splits=n_inner,
                epochs=epochs, patience=patience, batch_size=16, seed=seed, device=device)
            tables.append(table.assign(outer_fold=k))
            temperature = max(1.0, fit_temperature(torch.from_numpy(info['oof_logits']),
                                                   torch.from_numpy(info['oof_labels'])))
            with tempfile.TemporaryDirectory() as tmp:
                hist = train_from_arrays(
                    mt, {'X_train': Xtr, 'y_train': y[tr], 'X_val': Xtr, 'y_val': y[tr]},
                    scaler, device, epochs=info['epochs'], batch_size=16,
                    weight_decay=params['weight_decay'], hidden_dim=params['hidden_dim'],
                    dropout=params['dropout'], patience=patience, seed=seed,
                    checkpoint_dir=tmp, log_dir=tmp, noise_std=params['noise_std'],
                    early_stopping=False, temperature=temperature)
                # Not the deployment Predictor: it only accepts the field
                # models' feature list, and these lab features differ.
                net = load_checkpoint(hist['checkpoint_path'], device)
                logits = torch.from_numpy(tuning._logits(net, Xte, device))
                probs[mt][te] = torch.softmax(logits / temperature, dim=1).numpy()

    for name in probs:
        preds[name] = probs[name].argmax(axis=1)
    pd.DataFrame({'sample_id': d['ids'], 'pot': pots, 'end_dpi': d['end_dpi'], 'label': y,
                  'outer_fold': fold_of, **{f'pred_{n}': p for n, p in preds.items()}}
                 ).to_csv(os.path.join(out_dir, 'fold_predictions.csv'), index=False)
    pd.concat(tables, ignore_index=True).to_csv(os.path.join(out_dir, 'tuning.csv'),
                                                index=False)
    pd.DataFrame(selections).to_csv(os.path.join(out_dir, 'feature_selection.csv'),
                                    index=False)

    comparison = pd.DataFrame([summarise(name, y, preds[name], pots, seed) for name in preds])
    comparison.to_csv(os.path.join(out_dir, 'model_comparison.csv'), index=False)
    pre = presymptomatic_test(d, n_outer, seed, n_permutations)
    pre.to_csv(os.path.join(out_dir, 'presymptomatic_test.csv'), index=False)
    logger.info("Septoria out-of-fold results:\n%s", comparison.round(3).to_string(index=False))
    logger.info("Pre-symptomatic test:\n%s", pre.round(3).to_string(index=False))
    return {'comparison': comparison, 'presymptomatic': pre, 'onsets': onsets}


def summarise(name: str, y, pred, pots, seed: int, n_boot: int = 2000) -> dict:
    """Out-of-fold metrics; CIs by resampling whole pots (the unit of independence)."""
    from sklearn.metrics import accuracy_score, f1_score

    def f1(a, b):
        return f1_score(a, b, average='macro', labels=list(range(NUM_CLASSES)), zero_division=0)

    rng = np.random.default_rng(seed)
    unique = np.unique(pots)
    members = {p: np.flatnonzero(pots == p) for p in unique}
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([members[p] for p in rng.choice(unique, len(unique))])
        boot.append(f1(y[idx], pred[idx]))
    per_class = f1_score(y, pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    return {'Model': name, 'accuracy': accuracy_score(y, pred), 'f1_macro': f1(y, pred),
            'f1_macro_ci_low': float(np.percentile(boot, 2.5)),
            'f1_macro_ci_high': float(np.percentile(boot, 97.5)),
            'f1_healthy': per_class[0], 'f1_presymptomatic': per_class[1],
            'f1_symptomatic': per_class[2]}


def presymptomatic_test(d: dict, n_outer: int, seed: int,
                        n_permutations: int) -> pd.DataFrame:
    """
    The question the dataset is for: before visible symptoms, do the spectra
    separate inoculated from control leaves? Uses only windows of control
    leaves and pre-symptomatic inoculated leaves in which every observation
    looks normal (no chlorosis / necrosis, see sep.visible_onset), on end
    dates where both groups occur, a fixed regularized hyperspectral RF
    (PRESYMPTOMATIC_RF), pot-grouped CV -> ROC-AUC. Null distribution: treatment labels permuted
    across POTS (all windows of a pot keep one label), same pipeline.
    """
    from sklearn.metrics import roc_auc_score

    from src.models.classical_models import fit, make_random_forest

    # like with like: windows in which EVERY observation looks normal, both groups
    keep = np.isin(d['y'], [0, 1]) & d['normal']
    both = [dpi for dpi in np.unique(d['end_dpi'][keep])
            if set(d['y'][keep & (d['end_dpi'] == dpi)]) == {0, 1}]
    keep &= np.isin(d['end_dpi'], both)
    X, y, pots = d['Xh'][keep], (d['y'][keep] == 1).astype(int), d['pot'][keep]

    def oof_auc(labels):
        score = np.zeros(len(labels))
        for tr, te in tuning.grouped_folds(labels, pots, n_outer, seed):
            Xtr, Xte = scale(X[tr], X[te])
            Atr, Ate = create_aggregated_features(Xtr), create_aggregated_features(Xte)
            model = fit(make_random_forest(seed=seed, **PRESYMPTOMATIC_RF), Atr, labels[tr])
            score[te] = model.predict_proba(Ate)[:, 1]
        return roc_auc_score(labels, score)

    observed = oof_auc(y)
    pot_label = pd.Series(y, index=pots).groupby(level=0).first()
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        shuffled = pd.Series(rng.permutation(pot_label.to_numpy()), index=pot_label.index)
        null.append(oof_auc(shuffled.loc[pots].to_numpy()))
    null = np.array(null)
    return pd.DataFrame([{
        'comparison': 'control vs inoculated, before visible-symptom onset',
        'end_dpis': json.dumps([int(v) for v in both]), 'n_windows': int(len(y)),
        'n_presymptomatic': int(y.sum()), 'n_pots': int(len(np.unique(pots))),
        'roc_auc': observed, 'null_auc_mean': float(null.mean()) if len(null) else np.nan,
        'permutation_p': float((1 + (null >= observed).sum()) / (1 + len(null)))
        if len(null) else np.nan}])


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
    p = argparse.ArgumentParser(description='Septoria nested pot-grouped CV experiment')
    p.add_argument('--seq-len', type=int, default=4)
    p.add_argument('--outer-folds', type=int, default=6)
    p.add_argument('--inner-folds', type=int, default=4)
    p.add_argument('--max-features', type=int, default=8)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--permutations', type=int, default=200)
    p.add_argument('--models', nargs='+', default=list(DL_MODELS), choices=DL_MODELS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sensors', nargs='+', default=list(sep.SENSORS), choices=list(sep.SENSORS))
    a = p.parse_args(argv)
    run(seq_len=a.seq_len, n_outer=a.outer_folds, n_inner=a.inner_folds,
        max_features=a.max_features, epochs=a.epochs, seed=a.seed, models=a.models,
        n_permutations=a.permutations, sensors=tuple(a.sensors))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
