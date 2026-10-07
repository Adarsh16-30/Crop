"""
Unified Training Script for Deep Learning Models
=================================================
Handles training for SpectralLSTM and CNNLSTMModel with:
- CUDA GPU support
- Class-weighted loss
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping
- Checkpoint saving
- Metrics logging to CSV
"""

import argparse
import csv
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.models.lstm_model import SpectralLSTM
from src.models.cnn_lstm_model import CNNLSTMModel


def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def compute_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for balanced loss.

    Parameters
    ----------
    y : np.ndarray
        Label array.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Weight tensor of shape (num_classes,).
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = total / (len(classes) * counts)
    weight_tensor = torch.zeros(int(classes.max()) + 1, dtype=torch.float32)
    for c, w in zip(classes, weights):
        weight_tensor[int(c)] = w
    return weight_tensor.to(device)


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from numpy arrays."""
    X_train_arr = np.ascontiguousarray(X_train, dtype=np.float32)
    X_val_arr = np.ascontiguousarray(X_val, dtype=np.float32)
    y_train_arr = np.ascontiguousarray(y_train, dtype=np.int64)
    y_val_arr = np.ascontiguousarray(y_val, dtype=np.int64)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_arr),
        torch.from_numpy(y_train_arr)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_arr),
        torch.from_numpy(y_val_arr)
    )

    class_counts = Counter(y_train_arr.tolist())
    sample_weights = np.array(
        [1.0 / class_counts[int(label)] for label in y_train_arr],
        dtype=np.float32
    )
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               sampler=sampler, drop_last=False,
                               pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, pin_memory=pin_memory)

    return train_loader, val_loader


def build_model(
    model_type: str,
    input_dim: int,
    seq_len: int = 30,
    num_classes: int = 3,
    hidden_dim: int = 128,
    dropout: float = 0.3
) -> nn.Module:
    """
    Build a model by type name.

    Parameters
    ----------
    model_type : str
        'lstm' or 'cnn_lstm'.
    input_dim : int
        Number of input features.
    seq_len : int
        Sequence length (used by CNN-LSTM).
    num_classes : int
        Number of output classes.
    hidden_dim : int
        Hidden dimension for LSTM.
    dropout : float
        Dropout rate.

    Returns
    -------
    nn.Module
        Constructed model.
    """
    if model_type == 'lstm':
        return SpectralLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=num_classes,
            dropout=dropout
        )
    elif model_type == 'cnn_lstm':
        return CNNLSTMModel(
            n_bands=input_dim,
            seq_len=seq_len,
            num_classes=num_classes,
            lstm_hidden=hidden_dim,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm' or 'cnn_lstm'.")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """
    Train for one epoch.

    Returns
    -------
    tuple[float, float]
        Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Validate model on validation set.

    Returns
    -------
    tuple[float, float]
        Average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 10,
    checkpoint_dir: str = 'models/checkpoints',
    model_name: str = 'model',
    log_dir: str = 'reports'
) -> dict:
    """
    Full training loop with early stopping and LR scheduling.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    device : torch.device
        Compute device.
    epochs : int
        Maximum number of epochs.
    learning_rate : float
        Initial learning rate.
    patience : int
        Early stopping patience (epochs without improvement).
    checkpoint_dir : str
        Directory to save model checkpoints.
    model_name : str
        Name prefix for saved files.
    log_dir : str
        Directory to save training logs.

    Returns
    -------
    dict
        Training history with losses and accuracies per epoch.
    """
    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Compute class weights from training data
    all_labels = []
    for _, y_batch in train_loader:
        all_labels.extend(y_batch.numpy().tolist())
    class_weights = compute_class_weights(np.array(all_labels), device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                    weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1_proxy': [],
        'lr': []
    }

    best_val_loss = float('inf')
    best_val_acc = float('-inf')
    epochs_without_improvement = 0
    best_checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')

    # CSV log
    log_path = os.path.join(log_dir, f'{model_name}_training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"  Device: {device} | Epochs: {epochs} | LR: {learning_rate}")
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_f1_proxy = (2 * val_acc * train_acc) / (val_acc + train_acc + 1e-8)

        # LR scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_proxy'].append(val_f1_proxy)
        history['lr'].append(current_lr)

        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{train_loss:.6f}', f'{train_acc:.4f}',
                           f'{val_loss:.6f}', f'{val_acc:.4f}', f'{current_lr:.6f}'])

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        # Early stopping check
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1_proxy': val_f1_proxy,
                'model_name': model_name,
            }, best_checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")

    history['best_val_loss'] = best_val_loss
    history['best_val_acc'] = best_val_acc
    history['total_time'] = total_time
    history['best_epoch'] = epoch - epochs_without_improvement

    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch {checkpoint['epoch']}, val_acc={checkpoint.get('val_acc', float('nan')):.4f}, "
            f"val_loss={checkpoint['val_loss']:.4f})")
    return model


def main():
    """CLI entrypoint for model training."""
    parser = argparse.ArgumentParser(description='Train crop disease detection model')
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'cnn_lstm'],
                        help='Model type (default: lstm)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='LSTM hidden dimension (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Processed data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                        help='Checkpoint save directory')
    args = parser.parse_args()

    device = get_device()

    # Load processed data
    print(f"Loading data from {args.data_dir}/...")
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    print(f"Data loaded: X_train={X_train.shape}, X_val={X_val.shape}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=args.batch_size
    )

    # Build model
    model = build_model(
        model_type=args.model,
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Parameters: {total_params:,}")

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model
    )


if __name__ == '__main__':
    main()
