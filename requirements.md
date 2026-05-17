# Python Dependencies for ProjectLXJ-rev

This document lists all required Python packages and their recommended versions for the 4D Scaffold BSG Material AI Prediction System.

## Core Dependencies

### Deep Learning Framework
- **torch** >= 2.0.0
  - PyTorch for neural network implementation
  - Includes: torch.nn, torch.optim, torch.utils.data

### Scientific Computing
- **numpy** >= 1.21.0
  - Numerical operations and array manipulations
- **pandas** >= 1.3.0
  - Data manipulation and analysis

### Machine Learning
- **scikit-learn** >= 1.0.0
  - Data preprocessing, metrics, and model evaluation
  - Modules: StandardScaler, PCA, train_test_split, r2_score, mean_absolute_error, mean_squared_error, StratifiedKFold, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, RandomForestRegressor, MultiOutputRegressor

### Model Interpretability
- **shap** >= 0.41.0
  - SHAP values for model explanation and feature importance analysis

### Gradient Boosting
- **xgboost** >= 1.6.0
  - XGBoost regressor for model comparison

### Visualization
- **matplotlib** >= 3.5.0
  - Plotting and figure generation
  - Backend: Agg (non-interactive) for server environments
- **seaborn** >= 0.11.0
  - Statistical data visualization

### Model Persistence
- **joblib** >= 1.1.0
  - Saving and loading trained models and normalization statistics

### Progress Tracking
- **tqdm** >= 4.62.0
  - Progress bar visualization for pipeline execution

## Installation

### Using pip
```bash
pip install torch>=2.0.0 numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 shap>=0.41.0 xgboost>=1.6.0 matplotlib>=3.5.0 seaborn>=0.11.0 joblib>=1.1.0 tqdm>=4.62.0
```

### Using requirements.txt
Create a `requirements.txt` file with the following content:
```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
shap>=0.41.0
xgboost>=1.6.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
tqdm>=4.62.0
```

Then install:
```bash
pip install -r requirements.txt
```

### Using conda
```bash
conda create -n projectlxj python=3.9
conda activate projectlxj
conda install pytorch>=2.0.0 numpy pandas scikit-learn matplotlib seaborn joblib tqdm -c pytorch
conda install shap xgboost -c conda-forge
```

## Software Environment

### Operating Systems
- **Linux**: Ubuntu 20.04+, CentOS 7+, Debian 10+
- **macOS**: 10.15 (Catalina) or later
- **Windows**: Windows 10 or later (with WSL2 recommended)

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: Minimum 8GB, 16GB recommended for SHAP analysis
- **GPU**: Optional NVIDIA GPU with CUDA 11.8+ support for accelerated training
  - Recommended: NVIDIA RTX 3060 or higher with 8GB+ VRAM
- **Storage**: At least 2GB free space for model checkpoints, results, and intermediate files

### Python Version
- **Python**: >= 3.8, recommended 3.9 or 3.10
- **Package Manager**: pip >= 21.0 or conda >= 4.10

## Data Format Requirements

### Input Data Format
- **Training Data**: Tab-separated values (TSV) format
  - File: `data/train.txt`, `data/val.txt`, `data/test.txt`
  - Columns: 23 input features + target indicators
  - Encoding: UTF-8

### Output Data Format
- **Model Checkpoints**: PyTorch `.pth` format
- **Normalization Statistics**: Joblib `.joblib` format
- **Predictions**: NumPy `.npy` format
- **Training Logs**: JSON format
- **Figures**: PNG format (300 DPI, publication quality)

### Feature List (23 Input Features)
```
BSG, AF, Ani, Ecc, EqD, prolif1, prolif2, alp, ars, vAF, vAni, vEcc, vEqD,
tlength, tvolume, tnodes, scr, ulength, uarea, uvolume, vlength, varea, vvolume
```

### Target Indicators
- **Biocompatibility (Bio)**: BSG, AF, Ani, Ecc, EqD, prolif1, prolif2
- **Osteogenic (Osteo)**: BSG, AF, Ani, Ecc, EqD, alp, ars
- **Angiogenic (Angio)**: BSG, AF, Ani, Ecc, EqD, vAF, vAni, vEcc, vEqD, vlength, varea, vvolume

## Reproducibility

### Random Seed Configuration
All scripts use fixed random seeds for reproducibility:
- NumPy random seed: 42
- PyTorch random seed: 42
- Python random seed: 42
- CUDA deterministic mode: Enabled (if GPU available)

### Environment Isolation
For maximum reproducibility, we recommend using one of the following methods:

#### Option 1: Virtual Environment (Recommended)
```bash
python -m venv projectlxj_env
source projectlxj_env/bin/activate  # On Windows: projectlxj_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option 2: Conda Environment
```bash
conda env create -f environment.yml
conda activate projectlxj
```

#### Option 3: Docker Container
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg

CMD ["python", "start.py"]
```

Build and run:
```bash
docker build -t projectlxj .
docker run -v $(pwd)/results:/app/results projectlxj
```

### Version Control
- **Git Repository**: [https://github.com/GuanghuaChenHMU/ProjectMABS/tree/main]
- **Release Tag**: v1.0.0

## License Information

### Project License
- **License**: MIT License
- **Copyright**: 2024 ProjectLXJ Team
- **Permissions**: Free to use, modify, and distribute for academic and commercial purposes

### Dependency Licenses
- **PyTorch**: BSD 3-Clause License
- **NumPy**: BSD License
- **Pandas**: BSD 3-Clause License
- **scikit-learn**: BSD 3-Clause License
- **SHAP**: MIT License
- **XGBoost**: Apache License 2.0
- **Matplotlib**: PSF License
- **Seaborn**: BSD 3-Clause License
- **Joblib**: BSD 3-Clause License
- **tqdm**: MPL 2.0 / MIT License

All dependencies are open-source and compatible with academic use.

## Academic Citations

### Dependency Citations
Key dependencies that should be acknowledged:
- PyTorch: Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", NeurIPS 2019
- SHAP: Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
- XGBoost: Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016
- scikit-learn: Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 2011

## Contact Information

### Primary Contact
- **Name**: [Gunaghua Chen]
- **Email**: [602163@hrbmu.edu.cn]
- **Institution**: [The Second affiliated Hospital of Harbin Medical University]
- **Department**: [Orthopedic Department]

## Optional Dependencies

For GPU acceleration (CUDA):
```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure

The pipeline consists of 8 main scripts in the `scripts/` directory:

1. `1revised_strict_constrained_model.py` - Neural network training with constraints
2. `2revised_reverse_prediction_analysis.py` - Reverse optimization analysis
3. `3revised_virtual_experiment.py` - Virtual experiment and sensitivity analysis
4. `4shap_total.py` - SHAP model interpretability
5. `5constrian.py` - Constraint satisfaction analysis
6. `6revised_individual_subcharts.py` - Individual subplot generation
7. `7merged_charts_generator.py` - Publication-ready figure generation
8. `8compare.py` - Model comparison (MLP vs XGBoost vs RF)

## Execution

Run the complete pipeline:
```bash
python start.py
```

Or run individual scripts:
```bash
python scripts/1revised_strict_constrained_model.py
python scripts/2revised_reverse_prediction_analysis.py
# ... etc
```

## Troubleshooting

### PyTorch Installation
If you encounter issues with PyTorch installation, visit: https://pytorch.org/get-started/locally/

### SHAP Memory Issues
For large datasets, SHAP analysis may require significant memory. Consider:
- Using a subset of data for SHAP computation
- Increasing system RAM
- Using `shap.sample()` instead of `shap.kmeans()` for background data

### Matplotlib Backend
The scripts use `matplotlib.use("Agg")` for non-interactive plotting. If you need interactive plots, comment out this line or use a different backend.

## Version Compatibility

The project has been tested with:
- Python 3.9.7
- PyTorch 2.0.1
- NumPy 1.24.3
- Pandas 2.0.2
- scikit-learn 1.2.2
- SHAP 0.41.0
- XGBoost 1.7.5
- Matplotlib 3.7.1
- Seaborn 0.12.2
- Joblib 1.2.0
- tqdm 4.65.0