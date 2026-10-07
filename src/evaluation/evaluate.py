"""
Model Evaluation and Comparison
================================
Evaluates trained models, generates confusion matrices, ROC curves,
training curve plots, and a comparison summary across all models.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import label_binarize

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.data_generator import LABEL_NAMES


def evaluate_dl_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    model_name: str = 'DL Model',
    batch_size: int = 64
) -> dict:
    """
    Evaluate a PyTorch deep learning model on test data.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    X_test : np.ndarray
        Test features of shape (n_samples, seq_len, n_features).
    y_test : np.ndarray
        True labels.
    device : torch.device
        Compute device.
    model_name : str
        Name for display.
    batch_size : int
        Inference batch size.

    Returns
    -------
    dict
        Evaluation results with metrics, predictions, and probabilities.
    """
    model = model.to(device)
    model.eval()

    X_test_arr = np.ascontiguousarray(X_test, dtype=np.float32)
    y_test_arr = np.ascontiguousarray(y_test, dtype=np.int64)

    dataset = TensorDataset(
        torch.from_numpy(X_test_arr),
        torch.from_numpy(y_test_arr)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=torch.cuda.is_available())

    all_preds = []
    all_probs = []

    use_amp = device.type == 'cuda'
    with torch.inference_mode():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(X_batch)
            else:
                logits = model(X_batch)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
    report = classification_report(y_test, y_pred, target_names=target_names,
                                   output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred,
        'probabilities': y_proba,
    }

    # Per-class metrics
    for label, name in LABEL_NAMES.items():
        if name in report:
            results[f'f1_{name.lower().replace(" ", "_")}'] = report[name]['f1-score']
            results[f'precision_{name.lower().replace(" ", "_")}'] = report[name]['precision']
            results[f'recall_{name.lower().replace(" ", "_")}'] = report[name]['recall']

    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy:       {results['accuracy']:.4f}")
    print(f"  F1 (macro):     {results['f1_macro']:.4f}")
    print(f"  F1 (weighted):  {results['f1_weighted']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")

    return results


def plot_confusion_matrix(
    cm: list | np.ndarray,
    model_name: str,
    output_path: str,
    class_names: list[str] | None = None
) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm : array-like
        Confusion matrix.
    model_name : str
        Title prefix.
    output_path : str
        Path to save PNG.
    class_names : list[str] or None
        Class label names.
    """
    if class_names is None:
        class_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{model_name} — Confusion Matrix', fontsize=14)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {output_path}")


def plot_roc_curves(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    output_path: str,
    class_names: list[str] | None = None
) -> None:
    """
    Plot and save one-vs-rest ROC curves.

    Parameters
    ----------
    y_test : np.ndarray
        True labels.
    y_proba : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    model_name : str
        Title prefix.
    output_path : str
        Path to save PNG.
    class_names : list[str] or None
        Class label names.
    """
    if class_names is None:
        class_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]

    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} — ROC Curves (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved: {output_path}")


def plot_training_curves(
    log_csv_path: str,
    output_path: str,
    model_name: str = 'Model'
) -> None:
    """
    Plot training and validation loss/accuracy curves from CSV log.

    Parameters
    ----------
    log_csv_path : str
        Path to training log CSV.
    output_path : str
        Path to save PNG.
    model_name : str
        Title prefix.
    """
    df = pd.read_csv(log_csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='#3498db', lw=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='#e74c3c', lw=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} — Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(df['epoch'], df['train_acc'], label='Train Acc', color='#3498db', lw=2)
    axes[1].plot(df['epoch'], df['val_acc'], label='Val Acc', color='#e74c3c', lw=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} — Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {output_path}")


def compare_models(
    results_list: list[dict],
    output_csv: str = 'reports/model_comparison.csv',
    output_plot: str = 'reports/model_comparison.png'
) -> pd.DataFrame:
    """
    Create a comparison table and bar chart across all trained models.

    Parameters
    ----------
    results_list : list[dict]
        List of evaluation result dicts from evaluate_model / evaluate_dl_model.
    output_csv : str
        Path to save comparison CSV.
    output_plot : str
        Path to save comparison bar chart.

    Returns
    -------
    pd.DataFrame
        Comparison dataframe.
    """
    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro',
               'f1_healthy', 'f1_early_stress', 'f1_active_disease']

    rows = []
    for result in results_list:
        row = {'Model': result['model_name']}
        for metric in metrics:
            row[metric] = result.get(metric, None)
        rows.append(row)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nModel comparison saved: {output_csv}")
    print(df.to_string(index=False))

    # Bar chart
    plot_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.25
    colors = ['#3498db', '#2ecc71', '#f39c12']

    for i, metric in enumerate(plot_metrics):
        values = df[metric].values
        bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(),
                      color=colors[i])
        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(df['Model'].values, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved: {output_plot}")

    return df
