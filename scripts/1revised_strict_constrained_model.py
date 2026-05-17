"""
Strictly Constrained Neural Network for Multi-label Biomedical Prediction.

This module implements a regularized neural network with monotonic constraints
for predicting multi-task biomedical outcomes. The model enforces domain-specific
physical constraints (e.g., positive-only responses, bounded outputs) through
architecture design rather than post-hoc correction.

"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_squared_error, r2_score,
    mean_absolute_error
)
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 导入统计工具模块
try:
    from stats_utils import (
        calculate_confidence_interval,
        calculate_p_value_correlation,
        calculate_metrics_p_values,
        calculate_cv_metrics_p_values,
        save_p_values_to_json
    )
except ImportError:
    print("Warning: stats_utils module not found, using inline implementations")

# Configure logging for reproducible experiment tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Hyperparameter and training configuration container.
    
    All parameters are explicitly documented for reproducibility.
    """
    # Architecture
    input_dim: int = 23  # BSG included as input feature (23 features total)
    hidden_dim1: int = 512
    hidden_dim2: int = 256
    hidden_dim3: int = 128
    hidden_dim4: int = 64
    hidden_dim5: int = 32
    output_dim: int = 3
    num_constraints: int = 10
    
    # Training
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 1000
    batch_size: int = 32
    early_stopping_patience: int = 100
    
    # Regularization
    dropout_rate: float = 0.3
    l2_lambda: float = 1e-5
    
    # Learning rate scheduling
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 30
    lr_min: float = 1e-6
    
    # Paths
    checkpoint_dir: str = str(Path(__file__).parent.parent / "results" / "1forward" / "model_checkpoints")
    log_dir: str = str(Path(__file__).parent.parent / "results" / "1forward" / "training_logs")
    result_dir: str = str(Path(__file__).parent.parent / "results" / "1forward")
    
    # Reproducibility
    random_seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# REPRODUCIBILITY UTILITIES
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across numpy, torch, and python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility.")


def create_timestamp() -> str:
    """Generate timestamp string for experiment tracking."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# DATA HANDLING
# =============================================================================

# Feature definitions
INPUT_FEATURES = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'prolif1', 'prolif2', 
                  'alp', 'ars', 'vAF', 'vAni', 'vEcc', 'vEqD', 'tlength',
                  'tvolume', 'tnodes', 'scr', 'ulength', 'uarea', 'uvolume',
                  'vlength', 'varea', 'vvolume']

BIO_INDICATORS = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'prolif1', 'prolif2']
OSTEO_INDICATORS = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'alp', 'ars']
ANGIO_INDICATORS = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'vAF', 'vAni', 'vEcc', 
                    'vEqD', 'vlength', 'varea', 'vvolume']


class BSGDataset(Dataset):
    """PyTorch Dataset for BSG biomedical feature vectors.
    
    Args:
        features: Normalized input feature matrix (N, input_dim).
        targets: Target output matrix (N, output_dim).
        bsg_labels: Raw BSG configuration labels for stratification.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        bsg_labels: Optional[np.ndarray] = None
    ) -> None:
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Features ({features.shape[0]}) and targets ({targets.shape[0]}) "
                "must have same number of samples."
            )
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.bsg_labels = bsg_labels
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def load_data(
    train_path: str,
    val_path: str,
    test_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare train/validation/test data from tab-delimited files.
    
    BSG is now included as an input feature (22 features total).
    Targets are computed from three indicator groups:
        - BIO: Biocompatibility (BSG, AF, Ani, Ecc, EqD, prolif1, prolif2)
        - OSTEO: Osteogenic potential (BSG, AF, Ani, Ecc, EqD, alp, ars)
        - ANGIO: Angiogenic capacity (BSG, AF, Ani, Ecc, EqD, vAF, vAni, vEcc, vEqD, vlength, varea, vvolume)
    
    Softmax is applied to outputs ensuring BIO + OSTEO + ANGIO = 1.
    
    Args:
        train_path: Path to training data TSV.
        val_path: Path to validation data TSV.
        test_path: Path to test data TSV.
    
    Returns:
        Tuple of (train_features, train_targets, train_bsg,
                  val_features, val_targets, val_bsg,
                  test_features, test_targets, test_bsg).
    """
    logger.info("Loading datasets from %s, %s, %s", train_path, val_path, test_path)
    
    # Load raw data
    train_data = pd.read_csv(train_path, sep="\t", header=0)
    val_data = pd.read_csv(val_path, sep="\t", header=0)
    test_data = pd.read_csv(test_path, sep="\t", header=0)
    
    # Validate columns
    if list(train_data.columns) != INPUT_FEATURES:
        raise ValueError(
            f"Train columns {list(train_data.columns)} do not match expected {INPUT_FEATURES}"
        )
    if list(val_data.columns) != INPUT_FEATURES:
        raise ValueError(
            f"Val columns {list(val_data.columns)} do not match expected {INPUT_FEATURES}"
        )
    if list(test_data.columns) != INPUT_FEATURES:
        raise ValueError(
            f"Test columns {list(test_data.columns)} do not match expected {INPUT_FEATURES}"
        )
    
    # Extract BSG labels (for stratification)
    train_bsg = train_data['BSG'].values.astype(np.float32)
    val_bsg = val_data['BSG'].values.astype(np.float32)
    test_bsg = test_data['BSG'].values.astype(np.float32)
    
    # Extract ALL features including BSG (22 features total)
    train_features = train_data[INPUT_FEATURES].values.astype(np.float32)
    val_features = val_data[INPUT_FEATURES].values.astype(np.float32)
    test_features = test_data[INPUT_FEATURES].values.astype(np.float32)
    
    # Compute targets from indicator groups (average of each group)
    def compute_targets(df: pd.DataFrame) -> np.ndarray:
        """Compute target vectors from indicator groups.
        
        Targets are normalized so that BIO + OSTEO + ANGIO = 1 (softmax-ready).
        """
        bio_scores = df[BIO_INDICATORS].mean(axis=1).values
        osteo_scores = df[OSTEO_INDICATORS].mean(axis=1).values
        angio_scores = df[ANGIO_INDICATORS].mean(axis=1).values
        
        # Stack and apply softmax to ensure sum = 1
        raw_scores = np.stack([bio_scores, osteo_scores, angio_scores], axis=1)
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))  # Numerically stable softmax
        targets = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return targets.astype(np.float32)
    
    train_targets = compute_targets(train_data)
    val_targets = compute_targets(val_data)
    test_targets = compute_targets(test_data)
    
    logger.info(
        "Data loaded: train=%s, val=%s, test=%s",
        train_features.shape, val_features.shape, test_features.shape
    )
    logger.info(
        "Targets shape: train=%s, val=%s, test=%s",
        train_targets.shape, val_targets.shape, test_targets.shape
    )
    return (
        train_features, train_targets, train_bsg,
        val_features, val_targets, val_bsg,
        test_features, test_targets, test_bsg
    )


def normalize_features(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    method: str = "zscore",
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Normalize features using training set statistics.
    
    Args:
        train_features: Training feature matrix.
        val_features: Validation feature matrix.
        test_features: Test feature matrix.
        method: Normalization method ("zscore" or "minmax").
        save_path: Optional path to save normalization statistics.
    
    Returns:
        Tuple of (normalized_train, normalized_val, normalized_test, stats_dict).
    """
    if method == "zscore":
        train_mean = np.mean(train_features, axis=0)
        train_std = np.std(train_features, axis=0)
        train_std = np.where(train_std == 0, 1.0, train_std)  # Prevent division by zero
        
        norm_train = (train_features - train_mean) / train_std
        norm_val = (val_features - train_mean) / train_std
        norm_test = (test_features - train_mean) / train_std
        
        stats = {"method": "zscore", "mean": train_mean.tolist(), "std": train_std.tolist()}
        
    elif method == "minmax":
        train_min = np.min(train_features, axis=0)
        train_max = np.max(train_features, axis=0)
        train_range = np.where(train_max - train_min == 0, 1.0, train_max - train_min)
        
        norm_train = (train_features - train_min) / train_range
        norm_val = (val_features - train_min) / train_range
        norm_test = (test_features - train_min) / train_range
        
        stats = {"method": "minmax", "min": train_min.tolist(), "max": train_max.tolist()}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(stats, save_path)
        logger.info("Normalization statistics saved to %s", save_path)
    
    logger.info("Feature normalization completed (%s).", method)
    return norm_train, norm_val, norm_test, stats


def check_data_quality(
    features: np.ndarray,
    name: str,
    negative_threshold: float = 0.05
) -> Dict[str, Any]:
    """Check data quality and report potential issues.
    
    Args:
        features: Feature matrix.
        name: Dataset name for reporting.
        negative_threshold: Fraction of negatives to trigger warning.
    
    Returns:
        Quality report dictionary.
    """
    report = {
        "dataset": name,
        "n_samples": features.shape[0],
        "n_features": features.shape[1],
        "has_nan": bool(np.isnan(features).any()),
        "has_inf": bool(np.isinf(features).any()),
        "negative_ratios": {},
        "outlier_counts": {}
    }
    
    # Check for negative values (some features like uarea, uvolume should not be negative)
    for j in range(features.shape[1]):
        neg_ratio = (features[:, j] < 0).sum() / len(features)
        if neg_ratio > 0:
            report["negative_ratios"][j] = float(neg_ratio)
    
    # Check for extreme outliers (>5 std)
    for j in range(features.shape[1]):
        col = features[:, j]
        mean, std = col.mean(), col.std()
        if std > 0:
            outliers = ((col < mean - 5*std) | (col > mean + 5*std)).sum()
            if outliers > 0:
                report["outlier_counts"][j] = int(outliers)
    
    logger.info("Data quality check (%s): %d samples, %d features", 
                name, report["n_samples"], report["n_features"])
    if report["has_nan"]:
        logger.warning("NaN values detected in %s!", name)
    if report["negative_ratios"]:
        logger.warning("Negative values detected in %s: %s", name, report["negative_ratios"])
    
    return report


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ConstrainedRegressor(nn.Module):
    """Neural network with architecture matching the provided diagram.
    
    Architecture:
        Input (23) → Hidden1 (512) → Hidden2 (256) → Hidden3 (128) → 
        Hidden4 (64) → Hidden5 (32) → Output (3) → Softmax (100%)
    
    Args:
        input_dim: Number of input features (23, including BSG).
        hidden_dim1: First hidden layer dimension (512).
        hidden_dim2: Second hidden layer dimension (256).
        hidden_dim3: Third hidden layer dimension (128).
        hidden_dim4: Fourth hidden layer dimension (64).
        hidden_dim5: Fifth hidden layer dimension (32).
        output_dim: Number of output targets (3 for BIO/OSTEO/ANGIO).
        dropout_rate: Dropout probability (not shown in diagram).
    """
    
    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        hidden_dim3: int = 128,
        hidden_dim4: int = 64,
        hidden_dim5: int = 32,
        output_dim: int = 3,
        dropout_rate: float = 0.3
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.hidden_dim4 = hidden_dim4
        self.hidden_dim5 = hidden_dim5
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Architecture: Input → 5 Hidden Layers → Output → Softmax
        self.layers = nn.Sequential(
            # Hidden1: 23 → 512
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            # Hidden2: 512 → 256
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            # Hidden3: 256 → 128
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            # Hidden4: 128 → 64
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.ReLU(),
            # Hidden5: 64 → 32
            nn.Linear(hidden_dim4, hidden_dim5),
            nn.ReLU(),
            # Output: 32 → 3
            nn.Linear(hidden_dim5, output_dim)
            # Softmax applied in forward pass
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier/He initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax output ensuring BIO + OSTEO + ANGIO = 1.
        
        Args:
            x: Input tensor (batch_size, input_dim).
        
        Returns:
            Probability distribution (batch_size, output_dim) in [0, 1] with sum = 1.
        """
        raw_output = self.layers(x)
        return nn.functional.softmax(raw_output, dim=1)
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute approximate feature importance from first layer weights.
        
        Args:
            feature_names: Optional list of feature names.
        
        Returns:
            DataFrame with feature importance scores.
        """
        with torch.no_grad():
            # Use first layer weights for importance
            weights = self.layers[0].weight.abs().mean(dim=0).cpu().numpy()
        
        if feature_names is None:
            feature_names = [f"F{i+1}" for i in range(len(weights))]
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": weights
        }).sort_values("importance", ascending=False)
        
        return importance_df


# =============================================================================
# TRAINING ENGINE
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ModelConfig,
    device: torch.device,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """Train the constrained regressor with early stopping and LR scheduling.
    
    Args:
        model: The neural network model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        device: Computation device (CPU or CUDA).
        feature_names: Optional feature names for importance logging.
        verbose: Whether to print training progress.
    
    Returns:
        Training history dictionary containing loss and metric curves.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.lr_min
    )
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": [],
        "val_r2": [],
        "learning_rate": [],
        "epoch_times": []
    }
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    
    start_time = datetime.now()
    
    for epoch in range(config.num_epochs):
        epoch_start = datetime.now()
        
        # Training phase
        model.train()
        train_losses = []
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # MSE loss + L2 regularization
            loss = criterion(outputs, batch_targets)
            
            # Add explicit L2 penalty on weights
            l2_penalty = sum(p.pow(2.0).sum() for p in model.parameters())
            total_loss = loss + config.l2_lambda * l2_penalty
            
            total_loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_targets_list = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_losses.append(loss.item())
                
                val_preds.append(outputs.cpu().numpy())
                val_targets_list.append(batch_targets.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        
        # Compute metrics on full validation set
        val_preds_arr = np.concatenate(val_preds, axis=0)
        val_targets_arr = np.concatenate(val_targets_list, axis=0)
        
        val_rmse = np.sqrt(mean_squared_error(val_targets_arr, val_preds_arr))
        val_mae = mean_absolute_error(val_targets_arr, val_preds_arr)
        val_r2 = r2_score(val_targets_arr, val_preds_arr, multioutput="uniform_average")
        
        # Record history
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["val_rmse"].append(float(val_rmse))
        history["val_mae"].append(float(val_mae))
        history["val_r2"].append(float(val_r2))
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["epoch_times"].append((datetime.now() - epoch_start).total_seconds())
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping and model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "config": config.to_dict(),
            }, best_model_path)
        else:
            patience_counter += 1
        
        if verbose and (epoch % 50 == 0 or epoch < 5):
            logger.info(
                "Epoch %04d | Train: %.6f | Val: %.6f | RMSE: %.4f | R2: %.4f | LR: %.2e",
                epoch, avg_train_loss, avg_val_loss, val_rmse, val_r2, optimizer.param_groups[0]["lr"]
            )
        
        if patience_counter >= config.early_stopping_patience:
            logger.info("Early stopping triggered at epoch %d (best: %d, loss: %.6f)",
                        epoch, best_epoch, best_val_loss)
            break
    
    total_time = (datetime.now() - start_time).total_seconds()
    history["total_training_time"] = total_time
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = float(best_val_loss)
    
    logger.info("Training completed in %.1fs. Best model at epoch %d (val_loss=%.6f)",
                total_time, best_epoch, best_val_loss)
    
    # Save training history
    history_path = os.path.join(config.log_dir, "training_history.json")
    os.makedirs(config.log_dir, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", history_path)
    
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    bsg_labels: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Comprehensive model evaluation on test set.
    
    Args:
        model: Trained neural network.
        test_loader: Test data loader.
        device: Computation device.
        bsg_labels: Raw BSG labels for classification metrics.
        save_dir: Directory to save evaluation results.
    
    Returns:
        Evaluation metrics dictionary.
    """
    model.eval()
    criterion = nn.MSELoss()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Regression metrics
    metrics = {
        "mse": float(mean_squared_error(targets, preds)),
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2_macro": float(r2_score(targets, preds, multioutput="uniform_average")),
        "r2_per_target": {
            "biocompatibility": float(r2_score(targets[:, 0], preds[:, 0])),
            "osteogenic": float(r2_score(targets[:, 1], preds[:, 1])),
            "angiogenic": float(r2_score(targets[:, 2], preds[:, 2])),
        },
        "mae_per_target": {
            "biocompatibility": float(mean_absolute_error(targets[:, 0], preds[:, 0])),
            "osteogenic": float(mean_absolute_error(targets[:, 1], preds[:, 1])),
            "angiogenic": float(mean_absolute_error(targets[:, 2], preds[:, 2])),
        }
    }
    
    # Classification metrics (argmax prediction vs BSG label)
    if bsg_labels is not None:
        pred_labels = np.argmax(preds, axis=1)
        true_labels = np.array([
            0 if b == 5.0 else 1 if b == 10.0 else 2 for b in bsg_labels
        ])
        
        metrics["classification"] = {
            "accuracy": float(accuracy_score(true_labels, pred_labels)),
            "precision_macro": float(precision_score(true_labels, pred_labels, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(true_labels, pred_labels, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(true_labels, pred_labels).tolist(),
        }
    
    logger.info("Test Evaluation:")
    logger.info("  MSE: %.6f | RMSE: %.4f | MAE: %.4f | R2: %.4f",
                metrics["mse"], metrics["rmse"], metrics["mae"], metrics["r2_macro"])
    if "classification" in metrics:
        logger.info("  Accuracy: %.4f | F1 (macro): %.4f",
                    metrics["classification"]["accuracy"], metrics["classification"]["f1_macro"])
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions
        np.save(os.path.join(save_dir, "test_predictions.npy"), preds)
        np.save(os.path.join(save_dir, "test_targets.npy"), targets)
    
    return metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str
) -> None:
    """Plot and save training curves.
    
    Args:
        history: Training history dictionary.
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(len(history["train_loss"]))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], "r-", label="Validation", linewidth=1.5)
    ax.axvline(history["best_epoch"], color="g", linestyle="--", alpha=0.7, label=f"Best (epoch {history['best_epoch']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[0, 1]
    ax.plot(epochs, history["val_rmse"], "g-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Validation RMSE")
    ax.grid(True, alpha=0.3)
    
    # R2
    ax = axes[1, 0]
    ax.plot(epochs, history["val_r2"], "m-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R\u00b2 Score")
    ax.set_title("Validation R\u00b2 (Macro Averaged)")
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history["learning_rate"], "c-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Training curves saved to %s", save_path)


def plot_feature_importance(
    model: nn.Module,
    feature_names: List[str],
    save_path: str,
    top_k: int = 15
) -> None:
    """Plot feature importance from constraint projection weights.
    
    Args:
        model: Trained model.
        feature_names: List of feature names.
        save_path: Path to save the figure.
        top_k: Number of top features to display.
    """
    importance_df = model.get_feature_importance(feature_names)
    top_features = importance_df.head(top_k)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot grid behind bars
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    
    # Create horizontal bars with proper zorder
    bars = ax.barh(
        range(len(top_features)), 
        top_features["importance"].values, 
        color="steelblue",
        height=0.7,  # Set consistent bar height
        zorder=2  # Put bars on top of grid
    )
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute Weight", fontsize=12)
    ax.set_title(f"Top {top_k} Feature Importance", fontsize=14, pad=15)
    ax.tick_params(axis='both', labelsize=10)
    
    # Add value labels with proper positioning
    max_val = top_features["importance"].max()
    for bar, val in zip(bars, top_features["importance"].values):
        # Position labels to the right of each bar
        ax.text(
            val + max_val * 0.02,  # Offset by 2% of max value
            bar.get_y() + bar.get_height()/2, 
            f"{val:.3f}", 
            va="center", 
            ha="left",
            fontsize=9,
            zorder=3
        )
    
    # Adjust x-axis limit to accommodate labels
    ax.set_xlim(0, max_val * 1.15)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Feature importance plot saved to %s", save_path)


def plot_prediction_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
    save_path: str
) -> None:
    """Plot prediction vs target scatter plots for each output dimension.
    
    Args:
        predictions: Model predictions (N, 3).
        targets: Ground truth targets (N, 3).
        target_names: Names for each target dimension.
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        y_true = targets[:, i]
        y_pred = predictions[:, i]
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, c="steelblue", edgecolors="none")
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal")
        
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nR\u00b2 = {r2_score(y_true, y_pred):.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Prediction scatter plots saved to %s", save_path)


# =============================================================================
# K-FOLD CROSS VALIDATION
# =============================================================================

def cross_validate(
    features: np.ndarray,
    targets: np.ndarray,
    bsg_labels: np.ndarray,
    config: ModelConfig,
    n_splits: int = 5,
    device: torch.device = None
) -> Dict[str, Any]:
    """Perform stratified k-fold cross-validation.
    
    Args:
        features: Full feature matrix (train+val combined).
        targets: Full target matrix.
        bsg_labels: BSG labels for stratification.
        config: Model configuration.
        n_splits: Number of CV folds.
        device: Computation device.
    
    Returns:
        Cross-validation results dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Stratify by BSG category
    stratify_labels = np.array([0 if b == 5.0 else 1 if b == 10.0 else 2 for b in bsg_labels])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)
    
    cv_results = {
        "folds": [],
        "mean_rmse": 0.0,
        "std_rmse": 0.0,
        "mean_r2": 0.0,
        "std_r2": 0.0,
        "mean_accuracy": 0.0,
        "std_accuracy": 0.0,
    }
    
    fold_rmses = []
    fold_r2s = []
    fold_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, stratify_labels)):
        logger.info("CV Fold %d/%d", fold + 1, n_splits)
        
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]
        
        train_dataset = BSGDataset(X_train, y_train)
        val_dataset = BSGDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        model = ConstrainedRegressor(
            input_dim=config.input_dim,
            hidden_dim1=config.hidden_dim1,
            hidden_dim2=config.hidden_dim2,
            hidden_dim3=config.hidden_dim3,
            hidden_dim4=config.hidden_dim4,
            hidden_dim5=config.hidden_dim5,
            output_dim=config.output_dim,
            dropout_rate=config.dropout_rate
        ).to(device)
        
        # Temporarily adjust config for faster CV
        cv_config = ModelConfig(**config.to_dict())
        cv_config.num_epochs = min(config.num_epochs, 200)
        cv_config.early_stopping_patience = min(config.early_stopping_patience, 30)
        
        history = train_model(model, train_loader, val_loader, cv_config, device, verbose=False)
        
        # Load best model for evaluation
        best_path = os.path.join(cv_config.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
        
        # Evaluate on validation fold
        val_dataset_eval = BSGDataset(X_val, y_val)
        val_loader_eval = DataLoader(val_dataset_eval, batch_size=config.batch_size, shuffle=False)
        metrics = evaluate_model(model, val_loader_eval, device)
        
        fold_result = {
            "fold": fold + 1,
            "rmse": metrics["rmse"],
            "r2": metrics["r2_macro"],
            "mse": metrics["mse"],
        }
        if "classification" in metrics:
            fold_result["accuracy"] = metrics["classification"]["accuracy"]
            fold_accs.append(metrics["classification"]["accuracy"])
        
        cv_results["folds"].append(fold_result)
        fold_rmses.append(metrics["rmse"])
        fold_r2s.append(metrics["r2_macro"])
        
        logger.info("  Fold %d: RMSE=%.4f, R2=%.4f", fold + 1, metrics["rmse"], metrics["r2_macro"])
    
    cv_results["mean_rmse"] = float(np.mean(fold_rmses))
    cv_results["std_rmse"] = float(np.std(fold_rmses))
    cv_results["mean_r2"] = float(np.mean(fold_r2s))
    cv_results["std_r2"] = float(np.std(fold_r2s))
    if fold_accs:
        cv_results["mean_accuracy"] = float(np.mean(fold_accs))
        cv_results["std_accuracy"] = float(np.std(fold_accs))
    
    logger.info("CV Summary: RMSE=%.4f +/- %.4f | R2=%.4f +/- %.4f",
                cv_results["mean_rmse"], cv_results["std_rmse"],
                cv_results["mean_r2"], cv_results["std_r2"])
    
    return cv_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution pipeline for model training and evaluation."""
    
    # Initialize configuration
    config = ModelConfig()
    
    # Set seeds for reproducibility
    set_seed(config.random_seed)
    
    # Create output directories
    for dir_path in [config.checkpoint_dir, config.log_dir, config.result_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Save configuration
    config.save(os.path.join(config.log_dir, "config.json"))
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    train_features, train_targets, train_bsg, val_features, val_targets, val_bsg, test_features, test_targets, test_bsg = load_data(
        str(data_dir / "train.txt"),
        str(data_dir / "val.txt"),
        str(data_dir / "test.txt")
    )
    
    # Data quality checks
    for features, name in [(train_features, "train"), (val_features, "val"), (test_features, "test")]:
        quality = check_data_quality(features, name)
        with open(os.path.join(config.log_dir, f"data_quality_{name}.json"), "w") as f:
            json.dump(quality, f, indent=2)
    
    # Normalize features (z-score standardization)
    norm_train, norm_val, norm_test, norm_stats = normalize_features(
        train_features, val_features, test_features,
        method="zscore",
        save_path=os.path.join(config.log_dir, "normalization_stats.joblib")
    )
    
    # Create datasets
    train_dataset = BSGDataset(norm_train, train_targets, train_bsg)
    val_dataset = BSGDataset(norm_val, val_targets, val_bsg)
    test_dataset = BSGDataset(norm_test, test_targets, test_bsg)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = ConstrainedRegressor(
        input_dim=config.input_dim,
        hidden_dim1=config.hidden_dim1,
        hidden_dim2=config.hidden_dim2,
        hidden_dim3=config.hidden_dim3,
        hidden_dim4=config.hidden_dim4,
        hidden_dim5=config.hidden_dim5,
        output_dim=config.output_dim,
        dropout_rate=config.dropout_rate
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model initialized with %d parameters", n_params)
    
    # Optional: Run cross-validation before final training
    logger.info("=" * 60)
    logger.info("Starting 5-Fold Cross-Validation")
    logger.info("=" * 60)
    
    combined_features = np.concatenate([norm_train, norm_val], axis=0)
    combined_targets = np.concatenate([train_targets, val_targets], axis=0)
    combined_bsg = np.concatenate([train_bsg, val_bsg], axis=0)
    
    cv_results = cross_validate(combined_features, combined_targets, combined_bsg, config, device=device)
    
    cv_path = os.path.join(config.result_dir, "cross_validation_results.json")
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    logger.info("Cross-validation results saved to %s", cv_path)
    
    # Final training on full train+val
    logger.info("=" * 60)
    logger.info("Starting Final Model Training")
    logger.info("=" * 60)
    
    final_train_features = np.concatenate([norm_train, norm_val], axis=0)
    final_train_targets = np.concatenate([train_targets, val_targets], axis=0)
    final_train_dataset = BSGDataset(final_train_features, final_train_targets)
    final_train_loader = DataLoader(final_train_dataset, batch_size=config.batch_size, shuffle=True)
    
    final_model = ConstrainedRegressor(
        input_dim=config.input_dim,
        hidden_dim1=config.hidden_dim1,
        hidden_dim2=config.hidden_dim2,
        hidden_dim3=config.hidden_dim3,
        hidden_dim4=config.hidden_dim4,
        hidden_dim5=config.hidden_dim5,
        output_dim=config.output_dim,
        dropout_rate=config.dropout_rate
    ).to(device)
    
    history = train_model(final_model, final_train_loader, test_loader, config, device)
    
    # Load best model for evaluation
    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        final_model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Best model loaded from epoch %d", checkpoint["epoch"])
    
    # Evaluate on test set
    logger.info("=" * 60)
    logger.info("Final Test Set Evaluation")
    logger.info("=" * 60)
    
    test_metrics = evaluate_model(
        final_model, test_loader, device,
        bsg_labels=test_bsg,
        save_dir=config.result_dir
    )
    
    # Save test predictions
    test_preds = []
    with torch.no_grad():
        for batch_features, _ in test_loader:
            batch_features = batch_features.to(device)
            outputs = final_model(batch_features)
            test_preds.append(outputs.cpu().numpy())
    test_predictions = np.concatenate(test_preds, axis=0)
    np.save(os.path.join(config.result_dir, "test_predictions.npy"), test_predictions)
    
    # Visualizations
    feature_names = INPUT_FEATURES  # ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', ..., 'vvolume'] - 23 features
    
    plot_training_history(history, os.path.join(config.result_dir, "training_curves.png"))
    plot_feature_importance(final_model, feature_names, os.path.join(config.result_dir, "feature_importance.png"))
    plot_prediction_scatter(
        test_predictions, test_targets,
        ["Biocompatibility", "Osteogenic", "Angiogenic"],
        os.path.join(config.result_dir, "prediction_scatter.png")
    )
    
    # Save final model
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "config": config.to_dict(),
        "test_metrics": test_metrics,
        "cv_results": cv_results,
    }, os.path.join(config.checkpoint_dir, "final_model.pth"))
    logger.info("Final model saved to %s", os.path.join(config.checkpoint_dir, "final_model.pth"))
    
    # 计算统计指标的p值和置信区间
    logger.info("=" * 60)
    logger.info("Generating statistical significance report (p-values and 95% CI)")
    logger.info("=" * 60)
    
    # 获取测试集真实值和预测值
    test_targets_np = test_targets  # 已经是numpy数组
    
    # 计算指标的统计显著性
    p_value_results = {
        'module': '1revised_strict_constrained_model',
        'description': 'Statistical significance report for forward prediction model',
        'timestamp': datetime.now().isoformat(),
        'test_set_statistics': {},
        'cross_validation_statistics': {},
        'metrics_statistical_significance': {}
    }
    
    # 测试集指标的统计显著性
    p_value_results['test_set_statistics'] = {
        'sample_size': int(len(test_targets_np)),
        'output_dimensions': test_targets_np.shape[1]
    }
    
    # 计算预测指标的p值和置信区间
    metrics_stats = calculate_metrics_p_values(test_targets_np, test_predictions)
    p_value_results['metrics_statistical_significance'] = {
        'biocompatibility': metrics_stats.get('output_0', {}),
        'osteogenic': metrics_stats.get('output_1', {}),
        'angiogenic': metrics_stats.get('output_2', {})
    }
    
    # 交叉验证结果的统计显著性
    cv_p_values = {}
    if cv_results and 'folds' in cv_results:
        fold_rmse = [fold['rmse'] for fold in cv_results['folds']]
        fold_r2 = [fold['r2'] for fold in cv_results['folds']]
        fold_mse = [fold['mse'] for fold in cv_results['folds']]
        
        cv_p_values['rmse'] = calculate_cv_metrics_p_values(fold_rmse, 'RMSE')
        cv_p_values['r2'] = calculate_cv_metrics_p_values(fold_r2, 'R2')
        cv_p_values['mse'] = calculate_cv_metrics_p_values(fold_mse, 'MSE')
        
        # 添加均值和标准差信息
        cv_p_values['mean_metrics'] = {
            'rmse': float(np.mean(fold_rmse)),
            'r2': float(np.mean(fold_r2)),
            'mse': float(np.mean(fold_mse)),
            'std_rmse': float(np.std(fold_rmse)),
            'std_r2': float(np.std(fold_r2)),
            'std_mse': float(np.std(fold_mse))
        }
    
    p_value_results['cross_validation_statistics'] = cv_p_values
    
    # 保存p_value.json
    save_p_values_to_json(p_value_results, config.result_dir, 'p_value.json')
    logger.info("Statistical significance report saved to %s", os.path.join(config.result_dir, "p_value.json"))
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
