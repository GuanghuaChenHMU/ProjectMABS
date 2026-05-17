import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import json

plt.rcParams.update({'font.size': 14})

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

project_dir = Path(__file__).parent.parent
output_dir = project_dir / "results" / "8compare"
output_dir.mkdir(parents=True, exist_ok=True)

DATA_DIR = project_dir / "data"

INPUT_FEATURES = [
    'BSG', 'AF', 'Ani', 'Ecc', 'EqD',
    'prolif1', 'prolif2',
    'alp', 'ars',
    'vAF', 'vAni', 'vEcc', 'vEqD',
    'tlength', 'tvolume', 'tnodes',
    'scr', 'ulength', 'uarea', 'uvolume',
    'vlength', 'varea', 'vvolume'
]

BIO_INDICATORS = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'prolif1', 'prolif2']
OSTEO_INDICATORS = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'alp', 'ars']
ANGIO_INDICATORS = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'vAF', 'vAni', 'vEcc', 'vEqD', 'vlength', 'varea', 'vvolume']

def cross_validate_xgboost(X, y, n_splits=5):
    print(f"Performing {n_splits}-fold cross-validation for XGBoost...")
    y_strat = np.argmax(y, axis=1)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_folds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_strat)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = MultiOutputRegressor(XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=1.0,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            random_state=42,
            eval_metric='rmse',
            verbosity=0
        ))
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        metrics_folds.append(calculate_metrics(y_val_fold, y_pred))
        
        print(f"  Fold {fold+1}/{n_splits} - MSE: {metrics_folds[-1]['MSE']:.6f}, R2: {metrics_folds[-1]['R2']:.6f}")
    
    return aggregate_cv_results(metrics_folds, n_splits)

def cross_validate_random_forest(X, y, n_splits=5):
    print(f"Performing {n_splits}-fold cross-validation for Random Forest...")
    y_strat = np.argmax(y, axis=1)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_folds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_strat)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        metrics_folds.append(calculate_metrics(y_val_fold, y_pred))
        
        print(f"  Fold {fold+1}/{n_splits} - MSE: {metrics_folds[-1]['MSE']:.6f}, R2: {metrics_folds[-1]['R2']:.6f}")
    
    return aggregate_cv_results(metrics_folds, n_splits)

def cross_validate_mlp(X, y, n_splits=5):
    print(f"Performing {n_splits}-fold cross-validation for MLP...")
    y_strat = np.argmax(y, axis=1)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_folds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_strat)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        class MLPModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3)
                )
            
            def forward(self, x):
                raw_output = self.layers(x)
                return nn.functional.softmax(raw_output, dim=1)
        
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        model = MLPModel(X_train_fold.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6)
        
        X_train_tensor = torch.from_numpy(X_train_fold.astype(np.float32)).to(device)
        y_train_tensor = torch.from_numpy(y_train_fold.astype(np.float32)).to(device)
        X_val_tensor = torch.from_numpy(X_val_fold.astype(np.float32)).to(device)
        y_val_tensor = torch.from_numpy(y_val_fold.astype(np.float32)).to(device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50
        max_epochs = 500
        
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            train_loss = criterion(outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            lr_scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val_tensor).cpu().numpy()
        
        metrics_folds.append(calculate_metrics(y_val_fold, y_pred))
        print(f"  Fold {fold+1}/{n_splits} - MSE: {metrics_folds[-1]['MSE']:.6f}, R2: {metrics_folds[-1]['R2']:.6f}")
    
    return aggregate_cv_results(metrics_folds, n_splits)

def aggregate_cv_results(metrics_folds, n_splits):
    mse_values = [m['MSE'] for m in metrics_folds]
    rmse_values = [m['RMSE'] for m in metrics_folds]
    mae_values = [m['MAE'] for m in metrics_folds]
    r2_values = [m['R2'] for m in metrics_folds]
    
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    mean_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    mean_mae = np.mean(mae_values)
    std_mae = np.std(mae_values)
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    
    confidence_interval_mse = (mean_mse - 1.96 * std_mse / np.sqrt(n_splits), 
                               mean_mse + 1.96 * std_mse / np.sqrt(n_splits))
    confidence_interval_r2 = (mean_r2 - 1.96 * std_r2 / np.sqrt(n_splits), 
                              mean_r2 + 1.96 * std_r2 / np.sqrt(n_splits))
    
    return {
        'mean': {
            'MSE': mean_mse,
            'RMSE': mean_rmse,
            'MAE': mean_mae,
            'R2': mean_r2
        },
        'std': {
            'MSE': std_mse,
            'RMSE': std_rmse,
            'MAE': std_mae,
            'R2': std_r2
        },
        'confidence_interval': {
            'MSE': confidence_interval_mse,
            'R2': confidence_interval_r2
        },
        'folds': {
            'MSE': mse_values,
            'RMSE': rmse_values,
            'MAE': mae_values,
            'R2': r2_values
        }
    }

def load_data():
    print("Loading data...")
    train_data = pd.read_csv(DATA_DIR / 'train.txt', sep='\t')
    val_data = pd.read_csv(DATA_DIR / 'val.txt', sep='\t')
    test_data = pd.read_csv(DATA_DIR / 'test.txt', sep='\t')
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}")
    return train_data, val_data, test_data

def preprocess_data(train_data, val_data, test_data):
    print("Preprocessing data...")
    
    scaler = StandardScaler()
    scaler.fit(train_data[INPUT_FEATURES])
    
    train_scaled = pd.DataFrame(scaler.transform(train_data[INPUT_FEATURES]), columns=INPUT_FEATURES)
    val_scaled = pd.DataFrame(scaler.transform(val_data[INPUT_FEATURES]), columns=INPUT_FEATURES)
    test_scaled = pd.DataFrame(scaler.transform(test_data[INPUT_FEATURES]), columns=INPUT_FEATURES)
    
    def calculate_targets(data):
        bio_values = data[BIO_INDICATORS].mean(axis=1).values
        osteo_values = data[OSTEO_INDICATORS].mean(axis=1).values
        angio_values = data[ANGIO_INDICATORS].mean(axis=1).values
        
        raw_scores = np.stack([bio_values, osteo_values, angio_values], axis=1)
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        targets = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return targets
    
    y_train = calculate_targets(train_data)
    y_val = calculate_targets(val_data)
    y_test = calculate_targets(test_data)
    
    X_train = train_scaled.values
    X_val = val_scaled.values
    X_test = test_scaled.values
    
    processed_data = {
        'train': {'X': X_train, 'y': y_train},
        'val': {'X': X_val, 'y': y_val},
        'test': {'X': X_test, 'y': y_test}
    }
    
    return processed_data, scaler

def softmax_normalize(y):
    exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
    return exp_y / np.sum(exp_y, axis=1, keepdims=True)

def train_and_evaluate(processed_data, n_splits=5):
    print("Training and evaluating models...")
    
    X_train, y_train = processed_data['train']['X'], processed_data['train']['y']
    X_val, y_val = processed_data['val']['X'], processed_data['val']['y']
    X_test, y_test = processed_data['test']['X'], processed_data['test']['y']
    
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.vstack([y_train, y_val])
    
    cv_results = {}
    print("\n=== Cross-Validation ===")
    cv_results['XGBoost'] = cross_validate_xgboost(X_combined, y_combined, n_splits)
    cv_results['RandomForest'] = cross_validate_random_forest(X_combined, y_combined, n_splits)
    cv_results['MLP'] = cross_validate_mlp(X_combined, y_combined, n_splits)
    print("=== Cross-Validation Completed ===\n")
    
    results = {}
    output_names = ["Biocompatibility", "Osteogenic", "Angiogenic"]
    
    print("Training XGBoost...")
    xgb_model = MultiOutputRegressor(XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        eval_metric='rmse',
        verbosity=1
    ))
    print("  XGBoost parameters: n_estimators=200, max_depth=8, learning_rate=0.1")
    xgb_model.fit(X_train, y_train)
    print("  XGBoost training completed")
    y_pred_xgb = xgb_model.predict(X_test)
    
    results['XGBoost'] = {
        'model': xgb_model,
        'y_pred': y_pred_xgb,
        'y_true': y_test,
        'metrics': calculate_metrics(y_test, y_pred_xgb),
        'X_test': X_test
    }
    joblib.dump(xgb_model, output_dir / 'xgb_model.pkl')
    
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    print("  Random Forest parameters: n_estimators=200, max_depth=20, n_jobs=-1")
    rf_model.fit(X_train, y_train)
    print("  Random Forest training completed")
    y_pred_rf = rf_model.predict(X_test)
    
    results['RandomForest'] = {
        'model': rf_model,
        'y_pred': y_pred_rf,
        'y_true': y_test,
        'metrics': calculate_metrics(y_test, y_pred_rf),
        'X_test': X_test
    }
    joblib.dump(rf_model, output_dir / 'rf_model.pkl')
    
    print("Training MLP...")
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        
        class MLPModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3)
                )
            
            def forward(self, x):
                raw_output = self.layers(x)
                return nn.functional.softmax(raw_output, dim=1)
        
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        mlp_model = MLPModel(X_train.shape[1]).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mlp_model.parameters(), lr=5e-4, weight_decay=1e-4)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6)
        
        X_train_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
        y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device)
        X_val_tensor = torch.from_numpy(X_val.astype(np.float32)).to(device)
        y_val_tensor = torch.from_numpy(y_val.astype(np.float32)).to(device)
        X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)
        
        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 100
        max_epochs = 1000
        
        print(f"  MLP parameters: input_dim={X_train.shape[1]}, device={device.type}, lr=5e-4, patience={patience}")
        print(f"  Architecture: {X_train.shape[1]} -> 512 -> 256 -> 128 -> 64 -> 32 -> 3")
        print(f"  Training samples: {len(X_train)}")
        print("  Training started...")
        
        for epoch in range(max_epochs):
            mlp_model.train()
            optimizer.zero_grad()
            outputs = mlp_model(X_train_tensor)
            train_loss = criterion(outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            mlp_model.eval()
            with torch.no_grad():
                val_outputs = mlp_model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            history['loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            
            lr_scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(mlp_model.state_dict(), output_dir / 'mlp_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch:4d}/{max_epochs} | Train Loss: {train_loss.item():.6f} | Val Loss: {val_loss.item():.6f} | LR: {lr:.1e} | Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"    Early stopping triggered at epoch {epoch}")
                break
        
        mlp_model.load_state_dict(torch.load(output_dir / 'mlp_model.pth', map_location=device, weights_only=False))
        
        mlp_model.eval()
        with torch.no_grad():
            y_pred_mlp = mlp_model(X_test_tensor).cpu().numpy()
        
        mlp_metrics = calculate_metrics(y_test, y_pred_mlp)
        print(f"  MLP training completed | Final Best Val Loss: {best_val_loss:.6f}")
        print(f"  MLP test metrics: MSE={mlp_metrics['MSE']:.6f}, RMSE={mlp_metrics['RMSE']:.6f}, R2={mlp_metrics['R2']:.6f}")
        
        results['MLP'] = {
            'model': mlp_model,
            'y_pred': y_pred_mlp,
            'y_true': y_test,
            'metrics': mlp_metrics,
            'history': history,
            'X_test': X_test
        }
    except Exception as e:
        print(f"MLP training failed: {e}")
        import traceback
        traceback.print_exc()
        results['MLP'] = None
    
    print("\nTraining and evaluation completed successfully")
    return results, cv_results, output_names

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
    r2_macro = r2_score(y_true, y_pred, multioutput='uniform_average')
    max_errors = [max_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'SMAPE': smape,
        'R2': r2_macro,
        'R2_per_target': {
            'biocompatibility': r2_score(y_true[:, 0], y_pred[:, 0]),
            'osteogenic': r2_score(y_true[:, 1], y_pred[:, 1]),
            'angiogenic': r2_score(y_true[:, 2], y_pred[:, 2])
        },
        'MAE_per_target': {
            'biocompatibility': mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
            'osteogenic': mean_absolute_error(y_true[:, 1], y_pred[:, 1]),
            'angiogenic': mean_absolute_error(y_true[:, 2], y_pred[:, 2])
        },
        'MaxErrors': max_errors
    }

def save_results(results, cv_results=None):
    with open(output_dir / 'comparison_results.txt', 'w') as f:
        for model_name, result in results.items():
            if result is None:
                f.write(f"{model_name}: Training failed\n")
                continue
            
            metrics = result['metrics']
            f.write(f"{model_name} Results:\n")
            f.write(f"  Test Set Metrics:\n")
            f.write(f"    MSE: {metrics['MSE']:.6f}\n")
            f.write(f"    RMSE: {metrics['RMSE']:.6f}\n")
            f.write(f"    MAE: {metrics['MAE']:.6f}\n")
            f.write(f"    MAPE: {metrics['MAPE']:.6f}\n")
            f.write(f"    SMAPE: {metrics['SMAPE']:.6f}\n")
            f.write(f"    R2 (macro): {metrics['R2']:.6f}\n")
            if 'R2_per_target' in metrics:
                f.write(f"    R2 per target:\n")
                f.write(f"      - Biocompatibility: {metrics['R2_per_target']['biocompatibility']:.6f}\n")
                f.write(f"      - Osteogenic: {metrics['R2_per_target']['osteogenic']:.6f}\n")
                f.write(f"      - Angiogenic: {metrics['R2_per_target']['angiogenic']:.6f}\n")
            if 'MAE_per_target' in metrics:
                f.write(f"    MAE per target:\n")
                f.write(f"      - Biocompatibility: {metrics['MAE_per_target']['biocompatibility']:.6f}\n")
                f.write(f"      - Osteogenic: {metrics['MAE_per_target']['osteogenic']:.6f}\n")
                f.write(f"      - Angiogenic: {metrics['MAE_per_target']['angiogenic']:.6f}\n")
            f.write(f"    Max Errors: {metrics['MaxErrors']}\n")
            
            if cv_results and model_name in cv_results:
                cv = cv_results[model_name]
                f.write(f"  Cross-Validation Results (5-fold):\n")
                f.write(f"    Mean MSE: {cv['mean']['MSE']:.6f} ± {cv['std']['MSE']:.6f}\n")
                f.write(f"    Mean RMSE: {cv['mean']['RMSE']:.6f} ± {cv['std']['RMSE']:.6f}\n")
                f.write(f"    Mean MAE: {cv['mean']['MAE']:.6f} ± {cv['std']['MAE']:.6f}\n")
                f.write(f"    Mean R2: {cv['mean']['R2']:.6f} ± {cv['std']['R2']:.6f}\n")
                f.write(f"    95% CI for MSE: [{cv['confidence_interval']['MSE'][0]:.6f}, {cv['confidence_interval']['MSE'][1]:.6f}]\n")
                f.write(f"    95% CI for R2: [{cv['confidence_interval']['R2'][0]:.6f}, {cv['confidence_interval']['R2'][1]:.6f}]\n")
                f.write(f"    Fold MSE values: {[f'{v:.6f}' for v in cv['folds']['MSE']]}\n")
                f.write(f"    Fold R2 values: {[f'{v:.6f}' for v in cv['folds']['R2']]}\n")
            
            f.write("\n")
    
    metrics_df = pd.DataFrame()
    for model_name, result in results.items():
        if result is not None:
            metrics = result['metrics']
            metrics_df[model_name] = [metrics['MSE'], metrics['RMSE'], metrics['MAE'], 
                                      metrics['MAPE'], metrics['SMAPE'], metrics['R2']]
    
    metrics_df.index = ['MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'R2']
    metrics_df.to_csv(output_dir / 'comparison_metrics.csv')
    
    if cv_results:
        cv_metrics_df = pd.DataFrame()
        for model_name, cv in cv_results.items():
            cv_metrics_df[model_name + '_mean'] = [cv['mean']['MSE'], cv['mean']['RMSE'], cv['mean']['MAE'], cv['mean']['R2']]
            cv_metrics_df[model_name + '_std'] = [cv['std']['MSE'], cv['std']['RMSE'], cv['std']['MAE'], cv['std']['R2']]
        
        cv_metrics_df.index = ['MSE', 'RMSE', 'MAE', 'R2']
        cv_metrics_df.to_csv(output_dir / 'cross_validation_metrics.csv')
        
        fold_results = []
        for model_name, cv in cv_results.items():
            for fold_idx, (mse, r2) in enumerate(zip(cv['folds']['MSE'], cv['folds']['R2'])):
                fold_results.append({
                    'model': model_name,
                    'fold': fold_idx + 1,
                    'MSE': mse,
                    'R2': r2,
                    'RMSE': cv['folds']['RMSE'][fold_idx],
                    'MAE': cv['folds']['MAE'][fold_idx]
                })
        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(output_dir / 'cross_validation_fold_results.csv', index=False)
    
    print("Results saved.")

def set_plot_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

def plot_cv_boxplots(cv_results, metric='MSE'):
    set_plot_style()
    plt.figure(figsize=(12, 8))
    
    model_names = list(cv_results.keys())
    data = [cv_results[model]['folds'][metric] for model in model_names]
    
    sns.boxplot(data=data, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    plt.xticks(range(len(model_names)), model_names)
    plt.title(f'Cross-Validation {metric} Distribution (5-fold)', pad=20)
    plt.ylabel(metric)
    plt.xlabel('Models')
    plt.tight_layout()
    plt.savefig(output_dir / f'cv_{metric.lower()}_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cv_confidence_intervals(cv_results, metric='MSE'):
    set_plot_style()
    plt.figure(figsize=(12, 8))
    
    model_names = list(cv_results.keys())
    x = np.arange(len(model_names))
    width = 0.4
    
    means = [cv_results[model]['mean'][metric] for model in model_names]
    stds = [cv_results[model]['std'][metric] for model in model_names]
    cis_lower = [cv_results[model]['confidence_interval'][metric][0] for model in model_names]
    cis_upper = [cv_results[model]['confidence_interval'][metric][1] for model in model_names]
    
    plt.bar(x, means, width, yerr=[np.array(means) - np.array(cis_lower), np.array(cis_upper) - np.array(means)],
            capsize=10, alpha=0.8)
    
    plt.xticks(x, model_names)
    plt.title(f'Cross-Validation {metric} with 95% Confidence Intervals', pad=20)
    plt.ylabel(metric)
    plt.xlabel('Models')
    plt.tight_layout()
    plt.savefig(output_dir / f'cv_{metric.lower()}_confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cv_fold_comparison(cv_results, metric='MSE'):
    set_plot_style()
    plt.figure(figsize=(12, 8))
    
    model_names = list(cv_results.keys())
    colors = {'XGBoost': '#FF6B6B', 'RandomForest': '#4ECDC4', 'MLP': '#45B7D1'}
    folds = range(1, 6)
    
    for model_name in model_names:
        values = cv_results[model_name]['folds'][metric]
        plt.plot(folds, values, marker='o', label=model_name, color=colors.get(model_name, 'gray'), linewidth=2)
    
    plt.title(f'{metric} Across Folds', pad=20)
    plt.xlabel('Fold')
    plt.ylabel(metric)
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'cv_{metric.lower()}_fold_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cv_summary(cv_results):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    model_names = list(cv_results.keys())
    x = np.arange(len(model_names))
    
    mse_means = [cv_results[model]['mean']['MSE'] for model in model_names]
    mse_stds = [cv_results[model]['std']['MSE'] for model in model_names]
    
    r2_means = [cv_results[model]['mean']['R2'] for model in model_names]
    r2_stds = [cv_results[model]['std']['R2'] for model in model_names]
    
    axes[0].bar(x - 0.2, mse_means, width=0.4, yerr=mse_stds, capsize=5, label='MSE', color='#FF6B6B', alpha=0.7)
    axes[0].set_title('Cross-Validation MSE Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    
    axes[1].bar(x - 0.2, r2_means, width=0.4, yerr=r2_stds, capsize=5, label='R2', color='#4ECDC4', alpha=0.7)
    axes[1].set_title('Cross-Validation R2 Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names)
    axes[1].set_ylabel('R2 Score')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_loss(history, model_name):
    set_plot_style()
    plt.figure(figsize=(11, 7))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss ({model_name})')
    plt.savefig(output_dir / f'{model_name.lower()}_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(y_true, y_pred, title, output_path):
    set_plot_style()
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 8))
    sns.residplot(x=y_true, y=residuals, lowess=False, line_kws={'color': 'red'})
    plt.title(title, pad=20)
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(y_true, y_pred, title, output_path):
    set_plot_style()
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 8))
    sns.histplot(residuals, kde=True, color="blue")
    plt.title(title, pad=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_true_vs_pred(y_true, y_pred, title, output_path):
    set_plot_style()
    plt.figure(figsize=(12, 8))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.title(title, pad=20)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_true_vs_pred(y_true, y_pred, output_names, output_path, model_name):
    set_plot_style()
    colors = ['lightgray', 'lightgreen', 'lightpink']
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(output_names):
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, edgecolor='black', color=colors[i], label=name)
        z = np.polyfit(y_true[:, i], y_pred[:, i], 1)
        p = np.poly1d(z)
        plt.plot(y_true[:, i], p(y_true[:, i]), linestyle='--', color=colors[i])
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.title(f'True vs Predicted (Combined) - {model_name}', pad=20)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_residuals(y_true, y_pred, output_names, output_path, model_name):
    set_plot_style()
    colors = ['lightgray', 'lightgreen', 'lightpink']
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(output_names):
        residuals = y_true[:, i] - y_pred[:, i]
        sns.residplot(x=y_true[:, i], y=residuals, lowess=False,
                     line_kws={'color': colors[i]},
                     scatter_kws={'color': colors[i], 'edgecolors': 'black', 'alpha': 0.6},
                     label=name)
    
    plt.title(f'Combined Residual Plot - {model_name}', pad=20)
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance_heatmap(feature_importances, input_columns, model_name):
    set_plot_style()
    plt.figure(figsize=(12, 7))
    sns.heatmap([feature_importances], cmap='viridis', xticklabels=input_columns, yticklabels=['Feature Importance'])
    plt.title(f'Feature Importance Heatmap ({model_name})', pad=20)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name.lower()}_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_radar(metrics, model_name):
    set_plot_style()
    metrics_names = ['MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'R²']
    mse, rmse, mae, mape, smape, r2 = metrics['MSE'], metrics['RMSE'], metrics['MAE'], metrics['MAPE'], metrics['SMAPE'], metrics['R2']
    
    raw_mape = mape / 100
    raw_smape = smape / 100
    
    error_values = np.array([mse, rmse, mae, raw_mape, raw_smape])
    min_error = np.min(error_values)
    max_error = np.max(error_values)
    
    normalized_mse = 1 - ((mse - min_error) / (max_error - min_error)) if (max_error - min_error) != 0 else 0.5
    normalized_rmse = 1 - ((rmse - min_error) / (max_error - min_error)) if (max_error - min_error) != 0 else 0.5
    normalized_mae = 1 - ((mae - min_error) / (max_error - min_error)) if (max_error - min_error) != 0 else 0.5
    normalized_mape = 1 - ((raw_mape - min_error) / (max_error - min_error)) if (max_error - min_error) != 0 else 0.5
    normalized_smape = 1 - ((raw_smape - min_error) / (max_error - min_error)) if (max_error - min_error) != 0 else 0.5
    normalized_r2 = r2
    
    metrics_values = np.array([normalized_mse, normalized_rmse, normalized_mae, normalized_mape, normalized_smape, normalized_r2])
    
    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    metrics_values = np.concatenate((metrics_values, [metrics_values[0]]))
    metrics_names = np.concatenate((metrics_names, [metrics_names[0]]))
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, metrics_values, 'o-', linewidth=2)
    ax.fill(angles, metrics_values, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, metrics_names)
    ax.set_ylim(0, 1)
    plt.title(f'Model Performance Metrics ({model_name})', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name.lower()}_metrics_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison(results, output_names, cv_results=None):
    plt.figure(figsize=(12, 8))
    colors = {'XGBoost': '#FF6B6B', 'RandomForest': '#4ECDC4', 'MLP': '#45B7D1'}
    
    for model_name, result in results.items():
        if result is None:
            continue
        
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        for i, name in enumerate(output_names):
            plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.4, edgecolor='black', 
                       color=colors[model_name], label=f'{model_name}-{name}')
    
    min_val = min([result['y_true'].min() for result in results.values() if result is not None])
    max_val = max([result['y_true'].max() for result in results.values() if result is not None])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    plt.title('True vs Predicted (All Models)')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'true_vs_pred_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    x = np.arange(len(metrics))
    width = 0.25
    
    model_names = [name for name, result in results.items() if result is not None]
    
    for i, model_name in enumerate(model_names):
        result = results[model_name]
        values = [result['metrics'][m] for m in metrics]
        plt.bar(x + i*width, values, width, label=model_name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    
    if 'XGBoost' in results and results['XGBoost'] is not None:
        xgb_importance = results['XGBoost']['model'].estimators_[0].feature_importances_
        plt.bar([i - 0.2 for i in range(len(INPUT_FEATURES))], xgb_importance, width=0.2, label='XGBoost')
    
    if 'RandomForest' in results and results['RandomForest'] is not None:
        rf_importance = results['RandomForest']['model'].feature_importances_
        plt.bar([i + 0.2 for i in range(len(INPUT_FEATURES))], rf_importance, width=0.2, label='RandomForest')
    
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance Comparison')
    plt.xticks(range(len(INPUT_FEATURES)), INPUT_FEATURES, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'XGBoost' in results and results['XGBoost'] is not None:
        xgb_result = results['XGBoost']
        y_true = xgb_result['y_true']
        y_pred = xgb_result['y_pred']
        
        plot_combined_true_vs_pred(y_true, y_pred, output_names, output_dir / 'xgb_combined_true_vs_pred.png', 'XGBoost')
        plot_combined_residuals(y_true, y_pred, output_names, output_dir / 'xgb_combined_residual_plot.png', 'XGBoost')
        
        xgb_importance = xgb_result['model'].estimators_[0].feature_importances_
        plot_feature_importance_heatmap(xgb_importance, INPUT_FEATURES, 'XGBoost')
        
        for i, name in enumerate(output_names):
            plot_residuals(y_true[:, i], y_pred[:, i], f"Residual Plot for {name} (XGBoost)", 
                          output_dir / f'xgb_residual_plot_{name}.png')
            plot_error_distribution(y_true[:, i], y_pred[:, i], f"Error Distribution for {name} (XGBoost)", 
                                   output_dir / f'xgb_error_distribution_{name}.png')
    
    if 'RandomForest' in results and results['RandomForest'] is not None:
        rf_result = results['RandomForest']
        y_true = rf_result['y_true']
        y_pred = rf_result['y_pred']
        
        plot_combined_true_vs_pred(y_true, y_pred, output_names, output_dir / 'rf_combined_true_vs_pred.png', 'RandomForest')
        plot_combined_residuals(y_true, y_pred, output_names, output_dir / 'rf_combined_residual_plot.png', 'RandomForest')
        
        rf_importance = rf_result['model'].feature_importances_
        plot_feature_importance_heatmap(rf_importance, INPUT_FEATURES, 'RandomForest')
        
        for i, name in enumerate(output_names):
            plot_residuals(y_true[:, i], y_pred[:, i], f"Residual Plot for {name} (RandomForest)", 
                          output_dir / f'rf_residual_plot_{name}.png')
            plot_error_distribution(y_true[:, i], y_pred[:, i], f"Error Distribution for {name} (RandomForest)", 
                                   output_dir / f'rf_error_distribution_{name}.png')
    
    if 'MLP' in results and results['MLP'] is not None:
        mlp_result = results['MLP']
        y_true = mlp_result['y_true']
        y_pred = mlp_result['y_pred']
        history = mlp_result['history']
        
        plot_training_loss(history, 'MLP')
        plot_metrics_radar(mlp_result['metrics'], 'MLP')
        plot_combined_true_vs_pred(y_true, y_pred, output_names, output_dir / 'mlp_combined_true_vs_pred.png', 'MLP')
        plot_combined_residuals(y_true, y_pred, output_names, output_dir / 'mlp_combined_residual_plot.png', 'MLP')
        
        for i, name in enumerate(output_names):
            plot_residuals(y_true[:, i], y_pred[:, i], f"Residual Plot for {name} (MLP)", 
                          output_dir / f'mlp_residual_plot_{name}.png')
            plot_error_distribution(y_true[:, i], y_pred[:, i], f"Error Distribution for {name} (MLP)", 
                                   output_dir / f'mlp_error_distribution_{name}.png')
    
    if cv_results:
        plot_cv_boxplots(cv_results, 'MSE')
        plot_cv_boxplots(cv_results, 'R2')
        plot_cv_confidence_intervals(cv_results, 'MSE')
        plot_cv_confidence_intervals(cv_results, 'R2')
        plot_cv_fold_comparison(cv_results, 'MSE')
        plot_cv_fold_comparison(cv_results, 'R2')
        plot_cv_summary(cv_results)
        print("Cross-validation plots saved.")
    
    print("Plots saved.")

def main():
    print("=== Model Comparison Script ===")
    print(f"Output directory: {output_dir}")
    print(f"Input features: {len(INPUT_FEATURES)}")
    print()
    
    try:
        train_data, val_data, test_data = load_data()
        processed_data, scaler = preprocess_data(train_data, val_data, test_data)
        results, cv_results, output_names = train_and_evaluate(processed_data)
        save_results(results, cv_results)
        plot_comparison(results, output_names, cv_results)
        
        # 计算统计指标的p值和置信区间
        print("\n" + "="*60)
        print("Generating statistical significance report (p-values and 95% CI)")
        print("="*60)
        
        p_value_results = {
            'module': '8compare',
            'description': 'Statistical significance report for model comparison',
            'timestamp': pd.Timestamp.now().isoformat(),
            'models': [],
            'cross_validation_statistics': {}
        }
        
        # 获取测试集真实值和预测值
        X_train_val = np.concatenate([processed_data['X_train'], processed_data['X_val']], axis=0)
        y_train_val = np.concatenate([processed_data['y_train'], processed_data['y_val']], axis=0)
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # 各模型的统计显著性分析
        for model_name in cv_results.keys():
            if model_name in results:
                # 获取预测结果
                y_pred = results[model_name]['y_pred']
                
                # 计算指标的p值和置信区间
                metrics_stats = calculate_metrics_p_values(y_test, y_pred)
                
                # 交叉验证结果的统计分析
                cv_folds = cv_results[model_name]['folds']
                fold_rmse = [fold['RMSE'] for fold in cv_folds]
                fold_r2 = [fold['R2'] for fold in cv_folds]
                
                model_stats = {
                    'model_name': model_name,
                    'test_set_statistics': {
                        'biocompatibility': metrics_stats.get('output_0', {}),
                        'osteogenic': metrics_stats.get('output_1', {}),
                        'angiogenic': metrics_stats.get('output_2', {})
                    },
                    'cross_validation_statistics': {
                        'rmse': calculate_cv_metrics_p_values(fold_rmse, 'RMSE'),
                        'r2': calculate_cv_metrics_p_values(fold_r2, 'R2'),
                        'mean_rmse': float(np.mean(fold_rmse)),
                        'mean_r2': float(np.mean(fold_r2)),
                        'std_rmse': float(np.std(fold_rmse)),
                        'std_r2': float(np.std(fold_r2))
                    },
                    'sample_size': {
                        'train_val': int(len(y_train_val)),
                        'test': int(len(y_test))
                    }
                }
                
                p_value_results['models'].append(model_stats)
        
        # 保存p_value.json
        save_p_values_to_json(p_value_results, output_dir, 'p_value.json')
        print(f"[OK] p_value.json saved to: {output_dir / 'p_value.json'}")
        
        print("\n=== All operations completed! ===")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()