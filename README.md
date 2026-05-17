# Constrained Multi-Task Neural Network for Biomedical Material Property Prediction

## Abstract

This project presents a deep learning framework for predicting multi-task biomedical properties of 4D scaffold biomaterials. The proposed model employs a strictly constrained neural network architecture that guarantees physical validity through softmax output normalization, ensuring the sum of predicted probabilities equals unity (bio + osteo + angio = 1). The framework integrates comprehensive data preprocessing, multiple imputation strategies, reverse optimization via hybrid evolutionary algorithms, and SHAP-based interpretability analysis for scientific discovery.

---

## 1. Introduction

The prediction of biomedical material properties presents unique computational challenges due to:

- **Multi-task interdependence**: Biocompatibility, osteogenic, and angiogenic properties are correlated biological processes
- **Physical constraints**: Output predictions must satisfy sum-to-one probability constraints
- **Missing data**: Experimental biomedical datasets frequently contain incomplete observations
- **Interpretability requirements**: Scientific publication demands transparent model decision-making

This project implements a complete machine learning pipeline addressing these challenges through constrained deep learning, advanced imputation methods, and post-hoc interpretability techniques.

---

## 2. Project Structure

```
ProjectLXJ-rev/
├── data/                           # Experimental data repository
│   ├── impute/                     # Imputation analysis outputs
│   │   ├── diagnosis_report.txt
│   │   ├── recommendation.txt
│   │   └── sensitivity_report.txt
│   ├── train.txt                   # Training set (70%)
│   ├── val.txt                     # Validation set (15%)
│   └── test.txt                    # Test set (15%)
├── srcripts/                       # Core computational modules
│   ├── 0revised_data_impute_split.py
│   ├── 1revised_strict_constrained_model.py
│   ├── 2revised_reverse_prediction_analysis.py
│   ├── 3revised_virtual_experiment.py
│   ├── 4shap_total.py
│   ├── 5constrian.py
│   ├── 6revised_individual_subcharts.py
│   ├── 7merged_charts_generator.py
│   └── 8compare.py
├── results/                        # Computational outputs
│   ├── 1forward/                    # Forward prediction results
│   ├── 2reverse/                    # Reverse optimization outputs
│   ├── 3virtual_experiment/        # Virtual experiment reports
│   ├── 5constraint_iterations/     # Constraint analysis data
│   ├── 6individual_subcharts/      # Individual visualization panels
│   └── 7english_charts/            # Publication-ready figures
└── README.md                       # Project documentation
```

---

## 3. Methodology

### 3.1 Data Preprocessing Module

**Script**: `0revised_data_impute_split.py`

**Class**: `DataImputer`

**Description**: Handles missing data imputation and dataset partitioning for downstream model training.

**Key Methods**:
| Method | Function |
|--------|----------|
| `load_data()` | Loads raw tabular data and computes missing value statistics |
| `random_imputation()` | Random within-column sampling from observed values |
| `mice_imputation()` | Multivariate imputation via iterative chained equations |
| `diagnose_imputation()` | Generates comparative diagnostic report between methods |
| `save_best_imputed_split()` | Partitions data into train/val/test (70/15/15) |

**Input**: Raw data file with 23 biomedical features and 3 target indicators

**Output**: `train.txt`, `val.txt`, `test.txt` in tab-separated format

---

### 3.2 Constrained Neural Network Architecture

**Script**: `1revised_strict_constrained_model.py`

**Class**: `ConstrainedRegressor`

**Description**: Deep neural network with architecture-enforced sum-to-one constraint via softmax output layer.

**Architecture Specification**:
```
Input Layer (23 features)
    ↓
Hidden Layer 1: Linear(23 → 512) + ReLU
    ↓
Hidden Layer 2: Linear(512 → 256) + ReLU
    ↓
Hidden Layer 3: Linear(256 → 128) + ReLU
    ↓
Hidden Layer 4: Linear(128 → 64) + ReLU
    ↓
Hidden Layer 5: Linear(64 → 32) + ReLU
    ↓
Output Layer: Linear(32 → 3) + Softmax
    ↓
Constraint: Σ(output) = 1.0 ✓
```

**Key Features**:
- **Softmax normalization**: Ensures outputs represent valid probability distributions
- **Dropout regularization** (p=0.3): Prevents overfitting
- **L2 weight decay** (λ=10⁻⁵): Additional regularization
- **Learning rate scheduling**: ReduceLROnPlateau with patience=30
- **Early stopping**: Patience=100 epochs

**Configuration** (`ModelConfig`):
```python
@dataclass
class ModelConfig:
    input_dim: int = 23
    hidden_dim1: int = 512
    hidden_dim2: int = 256
    hidden_dim3: int = 128
    hidden_dim4: int = 64
    hidden_dim5: int = 32
    output_dim: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 1000
    batch_size: int = 32
    dropout_rate: float = 0.3
```

---

### 3.3 Reverse Prediction via Hybrid Optimization

**Script**: `2revised_reverse_prediction_analysis.py`

**Description**: Inverse problem solving to determine optimal input parameters achieving desired multi-task output distributions.

**Optimization Methods**:

| Method | Algorithm | Purpose |
|--------|-----------|---------|
| `reverse_optimize_gradient_descent()` | Stochastic gradient descent | Local refinement |
| `reverse_optimize_genetic_algorithm()` | Genetic algorithm | Global exploration |
| `reverse_optimize_hybrid()` | GA → GD pipeline | Global + local optimization |

**Hybrid Optimization Pipeline**:
```
Stage 1: Genetic Algorithm
├── Population size: 50
├── Generations: 100
├── Selection: Tournament
├── Crossover: Simulated binary (SBX)
└── Mutation: Polynomial mutation

Stage 2: Gradient Descent
├── Iterations: 500
├── Learning rate: 0.1
└── Adam optimizer
```

---

### 3.4 Virtual Experiment and Sensitivity Analysis

**Script**: `3revised_virtual_experiment.py`

**Description**: Systematic perturbation analysis to evaluate model behavior across input feature space.

**Method**: `simulate_material_parameters()`
- Generates perturbed samples based on data percentiles
- Evaluates prediction stability across feature ranges
- Computes comprehensive sensitivity metrics

---

### 3.5 Model Interpretability via SHAP

**Script**: `4shap_total.py`

**Description**: SHAP (SHapley Additive exPlanations)-based interpretability analysis.

**Visualizations Generated**:
| Plot Type | Description |
|-----------|-------------|
| Beeswarm plot | Global feature interactions and effects |
| Bar plot | Feature importance ranking |
| Waterfall plot | Local explanation for individual predictions |

**Output Names**: `['Biocompatibility', 'Osteogenic', 'Angiogenic']`

---

### 3.6 Constraint Satisfaction Analysis

**Script**: `5constrian.py`

**Description**: Comparative analysis across model variants with varying constraint enforcement levels.

**Model Variants**:
| Model | Constraint Level | Description |
|-------|-----------------|-------------|
| `InitialMLP` | None | Standard MLP with uncontrolled output |
| `ReferenceMLP` | Soft | MLP with softmax but no explicit constraints |
| `ConstrainedV1` | Moderate | Softmax + post-hoc correction |
| `StrictConstrainedMLP` | Strict | Architecture-enforced softmax |

**Metric**: `compute_constraint_satisfaction()`
$$\text{Satisfaction Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(|\sum_j \hat{y}_{ij} - 1.0| < \epsilon) \times 100\%$$

---

### 3.7 Visualization and Figure Generation

**Scripts**:
- `6revised_individual_subcharts.py`: Generates individual subplot panels
- `7merged_charts_generator.py`: Combines subplots into publication-ready figures

**Figure Types**:
- Forward prediction scatter plots
- Error distribution histograms
- Constraint validation plots
- Training convergence curves
- Confusion matrices

---

### 3.8 Comparative Model Analysis

**Script**: `8compare.py`

**Description**: Performance benchmarking against established machine learning baselines.

**Compared Models**:
| Model | Library | Configuration |
|-------|---------|----------------|
| Multi-Layer Perceptron | PyTorch | 5-layer constrained |
| XGBoost | xgboost | MultiOutputRegressor |
| Random Forest | sklearn | n_estimators=100 |

**Evaluation Metrics**:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Mean Absolute Error (MAE)
- Maximum Error

---

## 4. Feature Definitions

### Input Features (n=23)

| Feature Code | Description |
|--------------|-------------|
| BSG | Bioglass Scaffold Geometry parameter |
| AF | Angular Factor |
| Ani | Anisotropy coefficient |
| Ecc | Eccentricity |
| EqD | Equivalent Diameter |
| prolifer1 | Proliferation index 1 |
| prolifer2 | Proliferation index 2 |
| alp | Alkaline phosphatase activity |
| ars | Arginine side-chain contribution |
| vAF | Vascular Angular Factor |
| vAni | Vascular Anisotropy |
| vEcc | Vascular Eccentricity |
| vEqD | Vascular Equivalent Diameter |
| tlength | Total fiber length |
| tvolume | Total fiber volume |
| tnodes | Number of fiber nodes |
| scr | Scaffold compression ratio |
| ulength | Uracil length |
| uarea | Uracil area |
| uvolume | Uracil volume |
| vlength | Vascular length |
| varea | Vascular area |
| vvolume | Vascular volume |

### Target Indicators (n=3 groups)

| Indicator Group | Constituent Features |
|-----------------|---------------------|
| Biocompatibility | BSG, AF, Ani, Ecc, EqD, prolifer1, prolifer2 |
| Osteogenic | BSG, AF, Ani, Ecc, EqD, alp, ars |
| Angiogenic | BSG, AF, Ani, Ecc, EqD, vAF, vAni, vEcc, vEqD, vlength, varea, vvolume |

---

## 5. Workflow Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                            │
│  0revised_data_impute_split.py                                  │
│  ├── Random Imputation / MICE Imputation                        │
│  └── Train (70%) / Val (15%) / Test (15%) Split                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                               │
│  1revised_strict_constrained_model.py                           │
│  ├── ConstrainedRegressor (512→256→128→64→32→3 + Softmax)       │
│  ├── Early Stopping + Learning Rate Scheduling                  │
│  └── Best Model Checkpoint: results/1forward/model_checkpoints/ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
          ┌──────────────────┼──────────────────┐
          ↓                  ↓                  ↓
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ REVERSE ANALYSIS│ │ VIRTUAL EXPER.  │ │ SHAP ANALYSIS   │
│ 2revised_...py  │ │ 3revised_...py  │ │ 4shap_total.py  │
│                 │ │                 │ │                 │
│ Hybrid GA+GD    │ │ Sensitivity     │ │ Global/Local    │
│ optimization    │ │ analysis        │ │ explanations    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│               CONSTRAINT ANALYSIS                               │
│  5constrian.py                                                  │
│  ├── InitialMLP / ReferenceMLP / ConstrainedV1 / StrictMLP      │
│  └── Constraint satisfaction metrics                           │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION                                │
│  6revised_individual_subcharts.py → 7merged_charts_generator.py │
│  ├── Forward prediction plots                                  │
│  ├── Error distribution                                         │
│  └── Publication-ready figures                                 │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│               MODEL COMPARISON                                  │
│  8compare.py                                                    │
│  ├── MLP vs XGBoost vs Random Forest                            │
│  └── Comprehensive metrics benchmark                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Output Specifications

### Directory: `results/1forward/`

| File | Description |
|------|-------------|
| `model_checkpoints/best_model.pth` | Best validation loss model |
| `model_checkpoints/final_model.pth` | Final epoch model |
| `training_logs/training_history.json` | Epoch-level loss/R² history |
| `training_logs/config.json` | Model hyperparameters |
| `training_logs/normalization_stats.joblib` | Feature scaler |
| `test_predictions.npy` | Test set predictions (N×3) |
| `test_targets.npy` | Test set targets (N×3) |
| `test_metrics.json` | MSE, RMSE, R², MAE |

### Directory: `results/6individual_subcharts/`

Individual publication-quality subplot panels in PNG format.

### Directory: `results/7english_charts/`

Combined multi-panel figures for manuscript submission.

---

## 7. Technical Specifications

### Computational Environment

- **Framework**: PyTorch 2.x
- **Python**: 3.8+
- **Key Dependencies**: NumPy, Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn

### Hardware Acceleration

Automatic device selection with priority:
1. **Apple Silicon MPS**: `torch.device("mps")`
2. **NVIDIA CUDA**: `torch.device("cuda")`
3. **CPU fallback**: `torch.device("cpu")`

### Reproducibility

Random seed configuration (seed=42) for:
- Python `random` module
- NumPy `np.random`
- PyTorch `torch.manual_seed`
- CUDA deterministic mode

---

## 8. Usage Instructions

### Sequential Execution Order

```bash
# 1. Data preprocessing and imputation
python scripts/0revised_data_impute_split.py

# 2. Train constrained neural network
python scripts/1revised_strict_constrained_model.py

# 3. Reverse prediction optimization (requires trained model)
python scripts/2revised_reverse_prediction_analysis.py

# 4. Virtual experiment sensitivity analysis
python programs/3revised_virtual_experiment.py

# 5. SHAP interpretability analysis
python scripts/4shap_total.py

# 6. Constraint satisfaction comparison
python scripts/5constrian.py

# 7. Generate individual subplot panels
python scripts/6revised_individual_subcharts.py

# 8. Generate combined publication figures
python scripts/7merged_charts_generator.py

# 9. Compare with baseline ML models
python scripts/8compare.py
```

---

## 9. Key Innovations

1. **Architecture-Enforced Constraints**: Unlike post-hoc correction methods, our softmax output layer guarantees constraint satisfaction for all possible inputs

2. **Hybrid Inverse Optimization**: Combines evolutionary global search with gradient-based local refinement for robust inverse prediction

3. **Comprehensive Interpretability**: SHAP analysis provides both global feature importance and instance-level explanations

4. **Multi-Model Benchmarking**: Systematic comparison against XGBoost and Random Forest establishes baselines for the biomedical prediction task

---

## 10. Citation and License

This project was developed for biomedical research applications. When using this framework, please cite the corresponding manuscript and acknowledge the computational pipeline described herein.
