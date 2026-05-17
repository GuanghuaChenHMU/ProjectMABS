#!/usr/bin/env python3
"""
Global Configuration Parameters for ProjectLXJ-rev
Constrained Multi-Task Neural Network for Biomedical Material Property Prediction

This module provides centralized access to all shared parameters, paths,
and configurations used across the project scripts. Import this module
to ensure consistency across the entire pipeline.

Usage:
    from global_config import (
        PROJECT_DIR, DATA_DIR, RESULT_DIR,
        INPUT_FEATURES, OUTPUT_NAMES,
        ModelConfig, get_device, set_seed
    )
"""

import os
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import torch

PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
RESULT_DIR = PROJECT_DIR / "results"

MODEL_CHECKPOINT_DIR = RESULT_DIR / "1forward" / "model_checkpoints"
TRAINING_LOG_DIR = RESULT_DIR / "1forward" / "training_logs"
NORMALIZATION_PATH = TRAINING_LOG_DIR / "normalization_stats.joblib"

MODEL_PATH = MODEL_CHECKPOINT_DIR / "final_model.pth"
BEST_MODEL_PATH = MODEL_CHECKPOINT_DIR / "best_model.pth"

INPUT_FEATURES: List[str] = [
    'BSG', 'AF', 'Ani', 'Ecc', 'EqD',
    'prolif1', 'prolif2',
    'alp', 'ars',
    'vAF', 'vAni', 'vEcc', 'vEqD',
    'tlength', 'tvolume', 'tnodes',
    'scr', 'ulength', 'uarea', 'uvolume',
    'vlength', 'varea', 'vvolume'
]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    'BSG': 'Bioglass concentration',
    'AF': 'BSG Area fraction',
    'Ani': 'BSG Anisotropy',
    'Ecc': 'BSG Eccentricity',
    'EqD': 'BSG Equivalent Diameter',
    'prolif1': 'Proliferation index 1',
    'prolif2': 'Proliferation index 2',
    'alp': 'Alkaline phosphatase activity',
    'ars': 'Alizarin red stainning',
    'vAF': 'Vascular Area fraction',
    'vAni': 'Vascular Anisotropy',
    'vEcc': 'Vascular Eccentricity',
    'vEqD': 'Vascular Equivalent Diameter',
    'tlength': 'rat vessel length',
    'tvolume': 'rat vessel volume',
    'tnodes': 'rat vessel nodes',
    'scr': 'Scratch assay',
    'ulength': 'ultrasonic vessel length',
    'uarea': 'ultrasonic vessel area',
    'uvolume': 'ultrasonic vessel volume',
    'vlength': 'rabbit vessel length',
    'varea': 'rabbit vessel area',
    'vvolume': 'rabbit vessel volume'
}

BIO_INDICATORS: List[str] = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'prolif1', 'prolif2']
OSTEO_INDICATORS: List[str] = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'alp', 'ars']
ANGIO_INDICATORS: List[str] = [
    'BSG', 'AF', 'Ani', 'Ecc', 'EqD',
    'vAF', 'vAni', 'vEcc', 'vEqD',
    'vlength', 'varea', 'vvolume'
]

OUTPUT_NAMES: List[str] = ['Biocompatibility', 'Osteogenic', 'Angiogenic']

FEATURE_NAMES = INPUT_FEATURES

RANDOM_SEED: int = 42

FEATURE_INPUT_DIM: int = len(INPUT_FEATURES)
FEATURE_OUTPUT_DIM: int = 3


@dataclass
class ModelConfig:
    input_dim: int = 23
    hidden_dim1: int = 512
    hidden_dim2: int = 256
    hidden_dim3: int = 128
    hidden_dim4: int = 64
    hidden_dim5: int = 32
    output_dim: int = 3
    num_constraints: int = 10

    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 1000
    batch_size: int = 32
    early_stopping_patience: int = 100

    dropout_rate: float = 0.3
    l2_lambda: float = 1e-5

    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 30
    lr_min: float = 1e-6

    checkpoint_dir: str = str(MODEL_CHECKPOINT_DIR)
    log_dir: str = str(TRAINING_LOG_DIR)
    result_dir: str = str(RESULT_DIR / "1forward")

    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ImputationConfig:
    random_state: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    mice_max_iter: int = 10
    mice_initial_strategy: str = 'mean'
    mice_imputation_order: str = 'ascending'


@dataclass
class OptimizationConfig:
    ga_pop_size: int = 50
    ga_generations: int = 100
    ga_mutation_rate: float = 0.1
    ga_crossover_rate: float = 0.9
    gd_iterations: int = 500
    gd_lr: float = 0.1
    gd_tolerance: float = 1e-6


@dataclass
class SHAPConfig:
    background_samples: int = 100
    nsamples: int = 100
    random_seed: int = 42


@dataclass
class VisualizationConfig:
    dpi: int = 300
    figsize_single: tuple = (8, 6)
    figsize_double: tuple = (16, 6)
    figsize_triple: tuple = (24, 6)
    fontsize_title: int = 16
    fontsize_xlabel: int = 12
    fontsize_ylabel: int = 12
    fontsize_legend: int = 10


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif preferred == "cpu":
        return torch.device("cpu")
    else:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
    for subdir in ["2reverse", "3virtual_experiment", "5constraint_iterations",
                   "6individual_subcharts", "7english_charts"]:
        os.makedirs(RESULT_DIR / subdir, exist_ok=True)


def get_all_configs() -> Dict[str, Any]:
    return {
        "model": ModelConfig().to_dict(),
        "imputation": asdict(ImputationConfig()),
        "optimization": asdict(OptimizationConfig()),
        "shap": asdict(SHAPConfig()),
        "visualization": asdict(VisualizationConfig()),
        "paths": {
            "project_dir": str(PROJECT_DIR),
            "data_dir": str(DATA_DIR),
            "result_dir": str(RESULT_DIR),
            "model_path": str(MODEL_PATH),
            "normalization_path": str(NORMALIZATION_PATH)
        },
        "features": {
            "input_features": INPUT_FEATURES,
            "bio_indicators": BIO_INDICATORS,
            "osteo_indicators": OSTEO_INDICATORS,
            "angio_indicators": ANGIO_INDICATORS,
            "output_names": OUTPUT_NAMES,
            "input_dim": FEATURE_INPUT_DIM,
            "output_dim": FEATURE_OUTPUT_DIM
        }
    }


if __name__ == "__main__":
    print("=" * 70)
    print("ProjectLXJ-rev Global Configuration")
    print("=" * 70)
    print(f"\nProject Directory: {PROJECT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULT_DIR}")
    print(f"\nInput Features ({len(INPUT_FEATURES)}):")
    for i, feat in enumerate(INPUT_FEATURES, 1):
        desc = FEATURE_DESCRIPTIONS.get(feat, "")
        print(f"  {i:2d}. {feat:12s} - {desc}")
    print(f"\nTarget Indicators:")
    print(f"  Biocompatibility ({len(BIO_INDICATORS)}): {BIO_INDICATORS}")
    print(f"  Osteogenic ({len(OSTEO_INDICATORS)}): {OSTEO_INDICATORS}")
    print(f"  Angiogenic ({len(ANGIO_INDICATORS)}): {ANGIO_INDICATORS}")
    print(f"\nOutput Names: {OUTPUT_NAMES}")
    print(f"\nDefault Device: {get_device()}")
    print(f"Random Seed: {RANDOM_SEED}")
    print("\nConfiguration loaded successfully.")
