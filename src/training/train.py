"""
Unified Training Script for Deep Learning Models
=================================================
Trains SpectralLSTM / CNNLSTMModel with:
- reproducible seeding (python, numpy, torch, cuDNN)
- inverse-frequency class-weighted loss (computed from y_train directly;
  no resampling on top, so imbalance is corrected exactly once)
- ReduceLROnPlateau + early stopping on validation macro-F1
- temperature scaling fitted on the validation set
- self-describing checkpoints (weights + architecture + scaler + calibration)
  that load with torch.load(weights_only=True)
"""

import argparse
import csv
import json
import logging
import os
import random
import time

import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.constants import LABEL_NAMES, NUM_CLASSES
from src.inference.predictor import load_checkpoint_dict
from src.models.factory import MODEL_TYPES, build_model, build_model_from_config  # noqa: F401

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        return torch.device('cuda')
    logger.info("Using CPU")
    return torch.device('cpu')


def compute_class_weights(y: np.ndarray, device: torch.device,
                          num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights from the true training labels."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes).astype(float)
    weights = np.where(counts > 0, len(y) / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders. Training data is shuffled with a seeded generator."""
    def to_ds(X, y):
        return TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
        )

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(to_ds(X_train, y_train), batch_size=batch_size,
                              shuffle=True, generator=generator, pin_memory=pin_memory)
    val_loader = DataLoader(to_ds(X_val, y_val), batch_size=batch_size,
                            shuffle=False, pin_memory=pin_memory)
    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    noise_std: float = 0.0) -> tuple[float, float]:
    """
    Train for one epoch. Returns (mean loss, accuracy).

    `noise_std` adds Gaussian noise to the (standardized) inputs of every
    batch: a cheap regularizer against memorizing a small training set.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        if noise_std > 0:
            X_batch = X_batch + noise_std * torch.randn_like(X_batch)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += X_batch.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader and return (logits, labels) on CPU."""
    model.eval()
    logits, labels = [], []
    for X_batch, y_batch in loader:
        logits.append(model(X_batch.to(device)).cpu())
        labels.append(y_batch)
    return torch.cat(logits), torch.cat(labels)


def validate(model, val_loader, criterion, device) -> dict:
    """Validation loss, accuracy and macro-F1."""
    logits, labels = collect_logits(model, val_loader, device)
    loss = criterion(logits.to(device), labels.to(device)).item()
    preds = logits.argmax(dim=1).numpy()
    y = labels.numpy()
    return {
        'loss': loss,
        'acc': float((preds == y).mean()),
        'macro_f1': float(f1_score(y, preds, average='macro', zero_division=0)),
    }


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """
    Temperature scaling (Guo et al., 2017): find T > 0 minimizing the NLL
    of softmax(logits / T) on held-out data. Does not change argmax.
    """
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter,
                                  line_search_fn='strong_wolfe')
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.exp().item())
    if not np.isfinite(temperature):
        logger.warning("Temperature fit diverged; leaving the model uncalibrated (T=1)")
        return 1.0
    return float(np.clip(temperature, 0.05, 20.0))


def library_versions() -> dict:
    # str(): torch.__version__ is a TorchVersion object, which weights_only
    # loading (correctly) refuses to unpickle.
    return {'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'sklearn': str(sklearn.__version__)}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    y_train: np.ndarray,
    model_config: dict,
    epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoint_dir: str = 'models/checkpoints',
    model_name: str = 'model',
    log_dir: str = 'reports',
    noise_std: float = 0.0,
    early_stopping: bool = True,
    temperature: float | None = None,
    extra_config: dict | None = None
) -> dict:
    """
    Full training loop with early stopping on validation macro-F1.

    With `early_stopping=False` it trains exactly `epochs` epochs and keeps
    the final weights (validation metrics are only logged). `temperature`,
    if given, is stored instead of fitting one on the validation set, e.g.
    a temperature fitted on out-of-fold predictions. `extra_config` is
    merged into the checkpoint config.

    Parameters
    ----------
    y_train : np.ndarray
        True training labels, used for class weights.
    model_config : dict
        Plain-typed description stored in the checkpoint: model_type,
        input_dim, seq_len, hidden_dim, num_layers, dropout, scaler.

    Returns
    -------
    dict
        Per-epoch history plus best metrics, temperature and checkpoint path.
    """
    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    class_weights = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=max(1, patience // 2)
    )

    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc',
                               'val_macro_f1', 'lr')}
    best = {'macro_f1': -1.0, 'loss': float('inf'), 'acc': 0.0, 'epoch': 0}
    epochs_without_improvement = 0
    best_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
    log_path = os.path.join(log_dir, f'{model_name}_training_log.csv')

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss',
                                'val_acc', 'val_macro_f1', 'lr'])

    logger.info("Training %s on %s | epochs=%d lr=%g | class weights=%s",
                model_name, device, epochs, learning_rate,
                np.round(class_weights.cpu().numpy(), 3).tolist())
    start_time = time.time()

    def save(epoch: int, val: dict, temperature: float = 1.0) -> None:
        config = dict(model_config, **(extra_config or {}), temperature=temperature,
                      label_names={str(k): v for k, v in LABEL_NAMES.items()},
                      num_classes=NUM_CLASSES, versions=library_versions())
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': config,
            'val_loss': val['loss'],
            'val_acc': val['acc'],
            'val_macro_f1': val['macro_f1'],
            'model_name': model_name,
        }, best_path)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                noise_std=noise_std)
        val = validate(model, val_loader, criterion, device)
        scheduler.step(val['macro_f1'])
        lr = optimizer.param_groups[0]['lr']

        for key, value in (('train_loss', train_loss), ('train_acc', train_acc),
                           ('val_loss', val['loss']), ('val_acc', val['acc']),
                           ('val_macro_f1', val['macro_f1']), ('lr', lr)):
            history[key].append(value)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{train_loss:.6f}', f'{train_acc:.4f}',
                                    f"{val['loss']:.6f}", f"{val['acc']:.4f}",
                                    f"{val['macro_f1']:.4f}", f'{lr:.6f}'])

        if epoch % 5 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d | train loss %.4f acc %.4f | val loss %.4f "
                        "acc %.4f F1 %.4f | lr %.6f | %.1fs",
                        epoch, epochs, train_loss, train_acc, val['loss'], val['acc'],
                        val['macro_f1'], lr, time.time() - start_time)

        improved = (not early_stopping or val['macro_f1'] > best['macro_f1'] or
                    (val['macro_f1'] == best['macro_f1'] and val['loss'] < best['loss']))
        if improved:
            best = {**val, 'epoch': epoch}
            epochs_without_improvement = 0
            save(epoch, val)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (no val macro-F1 gain for %d epochs)",
                            epoch, patience)
                break

    # Calibrate the best model on the validation set (unless a temperature
    # was supplied) and re-save.
    ckpt = load_checkpoint_dict(best_path, device)
    model.load_state_dict(ckpt['model_state_dict'])
    if temperature is None:
        val_logits, val_labels = collect_logits(model, val_loader, device)
        temperature = fit_temperature(val_logits, val_labels)
    save(best['epoch'], best, temperature=temperature)

    total_time = time.time() - start_time
    logger.info("Training complete in %.1fs | best epoch %d | val macro-F1 %.4f | "
                "temperature %.3f | %s", total_time, best['epoch'], best['macro_f1'],
                temperature, best_path)

    history.update({
        'best_epoch': best['epoch'],
        'best_val_macro_f1': best['macro_f1'],
        'best_val_loss': best['loss'],
        'best_val_acc': best['acc'],
        'temperature': temperature,
        'total_time': total_time,
        'checkpoint_path': best_path,
    })
    return history


def load_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Rebuild a model from a self-describing checkpoint (weights_only load)."""
    ckpt = load_checkpoint_dict(checkpoint_path, device)
    model = build_model_from_config(ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    logger.info("Loaded %s (epoch %s, val macro-F1 %.4f)", checkpoint_path,
                ckpt.get('epoch'), ckpt.get('val_macro_f1', float('nan')))
    return model


def train_from_arrays(
    model_type: str,
    data: dict,
    scaler_params: dict,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    patience: int = 10,
    seed: int = 42,
    checkpoint_dir: str = 'models/checkpoints',
    log_dir: str = 'reports',
    noise_std: float = 0.0,
    early_stopping: bool = True,
    temperature: float | None = None,
    extra_config: dict | None = None
) -> dict:
    """Build, train and calibrate one model from preprocessed split arrays."""
    set_seed(seed)
    X_train, y_train = data['X_train'], data['y_train']
    seq_len, input_dim = X_train.shape[1], X_train.shape[2]

    model = build_model(model_type, input_dim=input_dim, seq_len=seq_len,
                        hidden_dim=hidden_dim, dropout=dropout)
    model_config = {
        'model_type': model_type,
        'input_dim': int(input_dim),
        'seq_len': int(seq_len),
        'hidden_dim': int(hidden_dim),
        'num_layers': int(model.num_layers if model_type == 'lstm'
                          else model.lstm.num_layers),
        'dropout': float(dropout),
        'conv_channels': [int(c) for c in getattr(model, 'conv_channels', [])] or None,
        'noise_std': float(noise_std),
        'scaler': {k: scaler_params[k] for k in ('mean', 'scale', 'feature_names', 'seq_len')},
        'seed': int(seed),
    }
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, data['X_val'], data['y_val'], batch_size=batch_size, seed=seed
    )
    logger.info("Model %s | parameters: %s", model_type,
                f"{sum(p.numel() for p in model.parameters()):,}")
    return train_model(
        model, train_loader, val_loader, device, y_train=y_train,
        model_config=model_config, epochs=epochs, learning_rate=learning_rate,
        weight_decay=weight_decay, patience=patience, checkpoint_dir=checkpoint_dir,
        model_name=model_type, log_dir=log_dir, noise_std=noise_std,
        early_stopping=early_stopping, temperature=temperature, extra_config=extra_config
    )


def main():
    """CLI entrypoint for model training."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Train crop disease detection model')
    parser.add_argument('--model', type=str, default='lstm', choices=list(MODEL_TYPES))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints')
    args = parser.parse_args()

    data = {k: np.load(os.path.join(args.data_dir, f'{k}.npy'))
            for k in ('X_train', 'y_train', 'X_val', 'y_val')}
    with open(os.path.join(args.data_dir, 'scaler_params.json')) as f:
        scaler_params = json.load(f)

    train_from_arrays(
        args.model, data, scaler_params, get_device(), epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr, weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim, dropout=args.dropout, patience=args.patience,
        seed=args.seed, checkpoint_dir=args.checkpoint_dir
    )


if __name__ == '__main__':
    main()
