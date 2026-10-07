"""
Model Evaluation and Comparison
================================
Shared metrics for every model, bootstrap confidence intervals, pairwise
McNemar significance tests, and plots (confusion matrix, ROC, training
curves, comparison chart with CI error bars).
"""

import itertools
import logging
import os

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize  # noqa: E402

from src.constants import LABEL_NAMES  # noqa: E402

logger = logging.getLogger(__name__)

TARGET_NAMES = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
LABELS = sorted(LABEL_NAMES)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric=accuracy_score,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI for a metric over test samples."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[i] = metric(y_true[idx], y_pred[idx])
    return (float(np.quantile(scores, alpha / 2)),
            float(np.quantile(scores, 1 - alpha / 2)))


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0)


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """
    Exact McNemar test: do two models make significantly different errors
    on the same test samples?
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_right, b_right = pred_a == y_true, pred_b == y_true
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    discordant = only_a + only_b
    p_value = 1.0 if discordant == 0 else binomtest(only_a, discordant, 0.5).pvalue
    return {'only_a_correct': only_a, 'only_b_correct': only_b, 'p_value': float(p_value)}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    model_name: str
) -> dict:
    """Metrics shared by classical and deep models, with 95% bootstrap CIs."""
    report = classification_report(y_true, y_pred, labels=LABELS, target_names=TARGET_NAMES,
                                   output_dict=True, zero_division=0)
    acc_ci = bootstrap_ci(y_true, y_pred, accuracy_score)
    f1_ci = bootstrap_ci(y_true, y_pred, macro_f1)
    results = {
        'model_name': model_name,
        'n_test': int(len(y_true)),
        'accuracy': accuracy_score(y_true, y_pred),
        'accuracy_ci_low': acc_ci[0], 'accuracy_ci_high': acc_ci[1],
        'f1_macro': macro_f1(y_true, y_pred),
        'f1_macro_ci_low': f1_ci[0], 'f1_macro_ci_high': f1_ci[1],
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        'predictions': np.asarray(y_pred),
        'probabilities': y_proba,
    }
    for name in TARGET_NAMES:
        key = name.lower().replace(' ', '_')
        results[f'f1_{key}'] = report[name]['f1-score']
        results[f'precision_{key}'] = report[name]['precision']
        results[f'recall_{key}'] = report[name]['recall']

    logger.info("%s | acc %.4f [%.4f, %.4f] | macro-F1 %.4f [%.4f, %.4f] | n=%d",
                model_name, results['accuracy'], *acc_ci, results['f1_macro'], *f1_ci,
                len(y_true))
    logger.info("\n%s", classification_report(y_true, y_pred, labels=LABELS,
                                              target_names=TARGET_NAMES, zero_division=0))
    return results


def evaluate_dl_model(predictor, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = 'DL Model') -> dict:
    """Evaluate a loaded src.inference.predictor.Predictor on standardized X_test."""
    probs = predictor.predict_proba_scaled(X_test)
    return compute_metrics(y_test, probs.argmax(axis=1), probs, model_name)


def plot_confusion_matrix(cm, model_name: str, output_path: str,
                          class_names: list[str] | None = None) -> None:
    """Plot and save a confusion matrix heatmap."""
    class_names = class_names or TARGET_NAMES
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{model_name} — Confusion Matrix', fontsize=14)
    _save(fig, output_path)


def plot_roc_curves(y_test: np.ndarray, y_proba: np.ndarray, model_name: str,
                    output_path: str, class_names: list[str] | None = None) -> None:
    """Plot and save one-vs-rest ROC curves."""
    class_names = class_names or TARGET_NAMES
    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, color in zip(range(len(class_names)), ['#2ecc71', '#f39c12', '#e74c3c']):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} — ROC Curves (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    _save(fig, output_path)


def plot_training_curves(log_csv_path: str, output_path: str, model_name: str = 'Model') -> None:
    """Plot loss, accuracy and validation macro-F1 from the training log."""
    df = pd.read_csv(log_csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='#3498db', lw=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='#e74c3c', lw=2)
    axes[0].set_title(f'{model_name} — Loss')
    axes[1].plot(df['epoch'], df['train_acc'], label='Train Acc', color='#3498db', lw=2)
    axes[1].plot(df['epoch'], df['val_acc'], label='Val Acc', color='#e74c3c', lw=2)
    if 'val_macro_f1' in df:
        axes[1].plot(df['epoch'], df['val_macro_f1'], label='Val macro-F1',
                     color='#8e44ad', lw=2, ls='--')
    axes[1].set_title(f'{model_name} — Accuracy / F1')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    _save(fig, output_path)


def compare_models(
    results_list: list[dict],
    y_test: np.ndarray,
    output_csv: str = 'reports/model_comparison.csv',
    output_plot: str = 'reports/model_comparison.png',
    output_significance: str = 'reports/model_significance.csv'
) -> pd.DataFrame:
    """
    Comparison table (with 95% CIs), pairwise McNemar tests, and a bar chart
    with CI error bars. All models must be evaluated on the same y_test.
    """
    metrics = ['n_test', 'accuracy', 'accuracy_ci_low', 'accuracy_ci_high',
               'f1_macro', 'f1_macro_ci_low', 'f1_macro_ci_high', 'f1_weighted',
               'precision_macro', 'recall_macro',
               'f1_healthy', 'f1_early_stress', 'f1_active_disease']
    df = pd.DataFrame([{'Model': r['model_name'], **{m: r.get(m) for m in metrics}}
                       for r in results_list])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Model comparison saved: %s\n%s", output_csv, df.to_string(index=False))

    sig_rows = []
    for a, b in itertools.combinations(results_list, 2):
        test = mcnemar_test(y_test, a['predictions'], b['predictions'])
        sig_rows.append({'model_a': a['model_name'], 'model_b': b['model_name'], **test,
                         'significant_at_0.05': test['p_value'] < 0.05})
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(output_significance, index=False)
    logger.info("Pairwise McNemar tests saved: %s\n%s", output_significance,
                sig.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35
    for i, (metric, color) in enumerate((('accuracy', '#3498db'), ('f1_macro', '#2ecc71'))):
        values = df[metric].to_numpy(dtype=float)
        err = np.vstack([values - df[f'{metric}_ci_low'], df[f'{metric}_ci_high'] - values])
        bars = ax.bar(x + i * width, values, width, yerr=err, capsize=4,
                      label=metric.replace('_', ' ').title(), color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8, color='white')
    ax.set_ylabel('Score (95% bootstrap CI)')
    ax.set_title(f'Model Comparison (n_test = {len(y_test)})')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(df['Model'], rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    _save(fig, output_plot)
    return df


def _save(fig, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved %s", output_path)
