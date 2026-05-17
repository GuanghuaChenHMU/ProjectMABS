#!/usr/bin/env python3
"""
虚拟实验分析脚本 - 独立运行模块
包含敏感性分析、材料参数模拟和综合指标评估
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
import joblib

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

# 路径配置
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULT_DIR = PROJECT_DIR / "results" / "3virtual_experiment"
MODEL_PATH = PROJECT_DIR / "results" / "1forward" / "model_checkpoints" / "final_model.pth"
NORMALIZATION_PATH = PROJECT_DIR / "results" / "1forward" / "training_logs" / "normalization_stats.joblib"

# Feature definitions
INPUT_FEATURES = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'prolif1', 'prolif2', 
                  'alp', 'ars', 'vAF', 'vAni', 'vEcc', 'vEqD', 'tlength',
                  'tvolume', 'tnodes', 'scr', 'ulength', 'uarea', 'uvolume',
                  'vlength', 'varea', 'vvolume']

BIO_INDICATORS = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'prolif1', 'prolif2']
OSTEO_INDICATORS = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'alp', 'ars']
ANGIO_INDICATORS = ['BSG', 'AF', 'Ani', 'Ecc', 'EqD', 'vAF', 'vAni', 'vEcc', 
                    'vEqD', 'vlength', 'varea', 'vvolume']

FEATURE_NAMES = INPUT_FEATURES  # Alias for compatibility

os.makedirs(RESULT_DIR, exist_ok=True)


class ConstrainedRegressor(nn.Module):
    """Neural network with architecture: Input(23) -> Hidden1(512) -> Hidden2(256) -> 
       Hidden3(128) -> Hidden4(64) -> Hidden5(32) -> Output(3) -> Softmax"""
    
    def __init__(self, input_dim=23, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128,
                 hidden_dim4=64, hidden_dim5=32, output_dim=3, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            # Hidden1: 23 -> 512
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            # Hidden2: 512 -> 256
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            # Hidden3: 256 -> 128
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            # Hidden4: 128 -> 64
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.ReLU(),
            # Hidden5: 64 -> 32
            nn.Linear(hidden_dim4, hidden_dim5),
            nn.ReLU(),
            # Output: 32 -> 3
            nn.Linear(hidden_dim5, output_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        raw_output = self.layers(x)
        return nn.functional.softmax(raw_output, dim=1)


def load_real_data():
    """加载真实数据（确保使用真实训练/验证/测试数据文件）"""
    print("\n" + "="*60)
    print("数据加载阶段（使用真实数据文件）")
    print("="*60)
    
    # 数据文件路径
    train_path = DATA_DIR / 'train.txt'
    val_path = DATA_DIR / 'val.txt'
    test_path = DATA_DIR / 'test.txt'
    
    # 验证数据文件存在
    required_files = [train_path, val_path, test_path]
    missing_files = [str(f) for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"缺少必需的数据文件！请确保以下文件存在:\n"
            f"  - {train_path}\n"
            f"  - {val_path}\n"
            f"  - {test_path}\n"
            f"缺失文件: {', '.join(missing_files)}"
        )
    
    print(f"Loading data from:\n  - {train_path}\n  - {val_path}\n  - {test_path}")
    
    train_data = pd.read_csv(train_path, sep='\t')
    val_data = pd.read_csv(val_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')

    # 验证数据集中的列是否完整
    required_columns = set(INPUT_FEATURES + BIO_INDICATORS + OSTEO_INDICATORS + ANGIO_INDICATORS)
    missing_cols = required_columns - set(train_data.columns)
    if missing_cols:
        raise ValueError(
            f"训练数据缺少必需的列！\n"
            f"缺失列: {', '.join(sorted(missing_cols))}\n"
            f"当前列: {', '.join(sorted(train_data.columns))}"
        )
    
    print(f"样本数: 训练={len(train_data)}, 验证={len(val_data)}, 测试={len(test_data)}")
    print(f"特征维度: {len(INPUT_FEATURES)} (包含BSG)")

    # 包含BSG作为输入特征（23个特征）
    train_features = train_data[INPUT_FEATURES].values.astype(np.float32)
    val_features = val_data[INPUT_FEATURES].values.astype(np.float32)
    test_features = test_data[INPUT_FEATURES].values.astype(np.float32)

    # 计算目标值（使用指标组的平均值并应用softmax）
    def compute_targets(df):
        bio_scores = df[BIO_INDICATORS].mean(axis=1).values
        osteo_scores = df[OSTEO_INDICATORS].mean(axis=1).values
        angio_scores = df[ANGIO_INDICATORS].mean(axis=1).values
        
        raw_scores = np.stack([bio_scores, osteo_scores, angio_scores], axis=1)
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        targets = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return targets.astype(np.float32)

    train_targets = compute_targets(train_data)
    val_targets = compute_targets(val_data)
    test_targets = compute_targets(test_data)

    X = np.vstack([train_features, val_features, test_features])
    y = np.vstack([train_targets, val_targets, test_targets])
    
    X_test = test_features
    y_test = test_targets

    print(f"Loaded {len(X)} samples total, X shape: {X.shape}, y shape: {y.shape}")
    print(f"Test set: {len(X_test)} samples")
    return X, y, X_test, y_test


def load_model_and_normalization():
    """加载训练好的模型和归一化参数（确保使用预训练模型）"""
    print("\n" + "="*60)
    print("模型加载阶段（使用预训练模型）")
    print("="*60)
    
    # 验证模型文件存在
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型文件不存在！请先运行 1revised_strict_constrained_model.py 训练模型。\n"
            f"期望路径: {MODEL_PATH}"
        )
    
    # 验证归一化参数文件存在
    if not NORMALIZATION_PATH.exists():
        raise FileNotFoundError(
            f"归一化参数文件不存在！\n"
            f"期望路径: {NORMALIZATION_PATH}"
        )
    
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Loading normalization stats from: {NORMALIZATION_PATH}")
    
    norm_stats = joblib.load(NORMALIZATION_PATH)

    model = ConstrainedRegressor(input_dim=23)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[OK] Model and normalization loaded successfully")
    return model, norm_stats


def get_predictions(model, norm_stats, X):
    """使用模型获取预测结果"""
    mean = np.array(norm_stats["mean"])
    std = np.array(norm_stats["std"])
    X_scaled = (X - mean) / std
    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    return predictions


def simulate_material_parameters(model, norm_stats, X):
    """模拟不同材料配比/几何参数下的性能响应（使用系统采样而非随机）"""
    print("[SIMULATE] Simulating material parameter combinations...")
    
    mean = np.array(norm_stats["mean"])
    std = np.array(norm_stats["std"])
    
    n_samples = 50
    param_names = FEATURE_NAMES[:8]
    param_indices = list(range(8))
    
    param_ranges = {}
    for idx, name in zip(param_indices, param_names):
        p5, p95 = np.percentile(X[:, idx], [5, 95])
        param_ranges[name] = (p5, p95)
        print(f"  {name}: [{p5:.6f}, {p95:.6f}] (5th-95th percentile)")
    
    simulations = []
    
    # 生成确定性的扰动因子（基于数据范围的系统采样）
    perturbation_factors = np.linspace(-0.1, 0.1, n_samples)
    
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            base_idx = i % len(X)
            sample = X[base_idx].copy()
            
            # 使用确定性扰动因子，基于数据百分位数范围进行系统调整
            for j, name in enumerate(param_names):
                min_val, max_val = param_ranges[name]
                range_val = max_val - min_val
                # 使用线性插值的扰动，基于数据范围而非随机
                perturbation = perturbation_factors[i] * range_val * 0.5
                sample[j] = sample[j] + perturbation
                sample[j] = np.clip(sample[j], min_val, max_val)
            
            sample_scaled = (sample - mean) / std
            X_tensor = torch.FloatTensor(sample_scaled.reshape(1, -1))
            pred = model(X_tensor).numpy()[0]
            
            simulations.append({
                **{name: float(sample[j]) for j, name in enumerate(FEATURE_NAMES)},
                'pred_bio': float(pred[0]),
                'pred_osteo': float(pred[1]),
                'pred_angio': float(pred[2]),
                'pred_sum': float(np.sum(pred))
            })
    
    df_simulations = pd.DataFrame(simulations)
    df_simulations.to_csv(RESULT_DIR / 'virtual_material_parameter_simulations.csv', index=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].scatter(df_simulations['AF'], df_simulations['pred_bio'], 
                    c=df_simulations['pred_osteo'], cmap='viridis', s=50, alpha=0.7)
    axes[0].set_xlabel('AF', fontsize=12)
    axes[0].set_ylabel('Biocompatibility', fontsize=12)
    axes[0].set_title('Bio vs AF', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(df_simulations['Ani'], df_simulations['pred_osteo'], 
                    c=df_simulations['pred_angio'], cmap='viridis', s=50, alpha=0.7)
    axes[1].set_xlabel('Anisotropy', fontsize=12)
    axes[1].set_ylabel('Osteogenic', fontsize=12)
    axes[1].set_title('Osteo vs Anisotropy', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(df_simulations['Ecc'], df_simulations['pred_angio'], 
                    c=df_simulations['pred_bio'], cmap='viridis', s=50, alpha=0.7)
    axes[2].set_xlabel('Eccentricity', fontsize=12)
    axes[2].set_ylabel('Angiogenic', fontsize=12)
    axes[2].set_title('Angio vs Eccentricity', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULT_DIR / 'virtual_material_parameter_simulation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[DONE] Material parameter simulation completed")


def generate_comprehensive_metrics_report(y_true, y_pred):
    """生成综合指标报告"""
    print("[REPORT] Generating comprehensive metrics report...")
    
    true_bio, true_osteo, true_angio = y_true[:, 0], y_true[:, 1], y_true[:, 2]
    pred_bio, pred_osteo, pred_angio = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    from scipy.stats import pearsonr
    
    metrics = {
        'Biocompatibility': {
            'R²': float(r2_score(true_bio, pred_bio)),
            'RMSE': float(np.sqrt(mean_squared_error(true_bio, pred_bio))),
            'MAE': float(mean_absolute_error(true_bio, pred_bio)),
            'Pearson r': float(pearsonr(true_bio, pred_bio)[0])
        },
        'Osteogenic': {
            'R²': float(r2_score(true_osteo, pred_osteo)),
            'RMSE': float(np.sqrt(mean_squared_error(true_osteo, pred_osteo))),
            'MAE': float(mean_absolute_error(true_osteo, pred_osteo)),
            'Pearson r': float(pearsonr(true_osteo, pred_osteo)[0])
        },
        'Angiogenic': {
            'R²': float(r2_score(true_angio, pred_angio)),
            'RMSE': float(np.sqrt(mean_squared_error(true_angio, pred_angio))),
            'MAE': float(mean_absolute_error(true_angio, pred_angio)),
            'Pearson r': float(pearsonr(true_angio, pred_angio)[0])
        },
        'Overall': {
            'Mean R²': float(np.mean([r2_score(true_bio, pred_bio), 
                                      r2_score(true_osteo, pred_osteo), 
                                      r2_score(true_angio, pred_angio)])),
            'Mean RMSE': float(np.mean([np.sqrt(mean_squared_error(true_bio, pred_bio)),
                                        np.sqrt(mean_squared_error(true_osteo, pred_osteo)),
                                        np.sqrt(mean_squared_error(true_angio, pred_angio))])),
            'Mean MAE': float(np.mean([mean_absolute_error(true_bio, pred_bio),
                                       mean_absolute_error(true_osteo, pred_osteo),
                                       mean_absolute_error(true_angio, pred_angio)])),
            'Pearson r': float(np.mean([pearsonr(true_bio, pred_bio)[0],
                                       pearsonr(true_osteo, pred_osteo)[0],
                                       pearsonr(true_angio, pred_angio)[0]]))
        }
    }
    
    constraint_sum = np.sum(y_pred, axis=1)
    constraint_errors = np.abs(constraint_sum - 1.0)
    metrics['Overall']['Constraint Satisfaction'] = float(np.mean(constraint_errors < 0.001) * 100)
    
    with open(RESULT_DIR / 'virtual_comprehensive_metrics_report.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n" + "="*60)
    print("综合指标报告")
    print("="*60)
    print(f"{'指标':<15} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'Pearson r':<10}")
    print("-"*60)
    
    for key in ['Biocompatibility', 'Osteogenic', 'Angiogenic']:
        m = metrics[key]
        print(f"{key:<15} {m['R²']:<8.4f} {m['RMSE']:<8.4f} {m['MAE']:<8.4f} {m['Pearson r']:<10.4f}")
    
    print("-"*60)
    o = metrics['Overall']
    print(f"{'Mean':<15} {o['Mean R²']:<8.4f} {o['Mean RMSE']:<8.4f} {o['Mean MAE']:<8.4f} {o['Pearson r']:<10.4f}")
    print(f"{'Constraint Satisfaction':<15} {o['Constraint Satisfaction']:.2f}%")
    print("="*60)
    
    metrics_df = pd.DataFrame({
        'Biocompatibility': [metrics['Biocompatibility']['R²'], metrics['Biocompatibility']['RMSE'], 
                            metrics['Biocompatibility']['MAE'], metrics['Biocompatibility']['Pearson r']],
        'Osteogenic': [metrics['Osteogenic']['R²'], metrics['Osteogenic']['RMSE'], 
                      metrics['Osteogenic']['MAE'], metrics['Osteogenic']['Pearson r']],
        'Angiogenic': [metrics['Angiogenic']['R²'], metrics['Angiogenic']['RMSE'], 
                      metrics['Angiogenic']['MAE'], metrics['Angiogenic']['Pearson r']]
    }, index=['R²', 'RMSE', 'MAE', 'Pearson r'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_title('Performance Metrics by Output Dimension', fontsize=14)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(RESULT_DIR / 'virtual_comprehensive_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[DONE] Comprehensive metrics report generated")


def create_virtual_experiment_plots(model, norm_stats, X, y):
    """生成虚拟实验验证图"""
    print("[PLOT] Generating virtual experiment validation plots...")
    
    predictions = get_predictions(model, norm_stats, X)
    true_bio, true_osteo, true_angio = y[:, 0], y[:, 1], y[:, 2]
    pred_bio, pred_osteo, pred_angio = predictions[:, 0], predictions[:, 1], predictions[:, 2]

    # 预测准确性散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(true_bio, pred_bio, alpha=0.7, color='#FF6B6B', s=40, label='Biocompatibility')
    ax.scatter(true_osteo, pred_osteo, alpha=0.7, color='#4ECDC4', s=40, label='Osteogenic')
    ax.scatter(true_angio, pred_angio, alpha=0.7, color='#45B7D1', s=40, label='Angiogenic')
    all_true = np.concatenate([true_bio, true_osteo, true_angio])
    all_pred = np.concatenate([pred_bio, pred_osteo, pred_angio])
    min_val, max_val = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('True Performance (%)', fontsize=12)
    ax.set_ylabel('Predicted Performance (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / 'virtual_experiment_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 预测误差分布直方图
    errors_bio = pred_bio - true_bio
    errors_osteo = pred_osteo - true_osteo
    errors_angio = pred_angio - true_angio
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(errors_bio, bins=15, alpha=0.7, label='Bio', color='#FF6B6B', density=True)
    ax.hist(errors_osteo, bins=15, alpha=0.7, label='Osteo', color='#4ECDC4', density=True)
    ax.hist(errors_angio, bins=15, alpha=0.7, label='Angio', color='#45B7D1', density=True)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / 'virtual_experiment_errors.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 约束满足率图
    total_pred = pred_bio + pred_osteo + pred_angio
    total_true = true_bio + true_osteo + true_angio
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(total_true, total_pred, alpha=0.7, color='green', s=50, label='Predictions')
    ax.plot([0.999, 1.001], [0.999, 1.001], 'r--', linewidth=2, label='Perfect Constraint')
    
    constraint_errors = np.abs(total_pred - 1.0)
    constraint_satisfaction = np.mean(constraint_errors < 0.001) * 100
    
    ax.text(0.05, 0.95, f'Constraint Satisfaction: {constraint_satisfaction:.1f}%',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            fontsize=12, fontweight='bold')
    ax.set_xlabel('True Total (Probability)', fontsize=12)
    ax.set_ylabel('Predicted Total (Probability)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0.998, 1.002)
    ax.set_ylim(0.998, 1.002)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / 'virtual_experiment_constraint.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[DONE] Virtual experiment validation plots generated")


def main():
    """主函数"""
    print("="*60)
    print("虚拟实验分析脚本 - 独立运行模块")
    print("="*60)
    print(f"数据路径: {DATA_DIR}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"输出目录: {RESULT_DIR}")
    print()
    
    # 加载数据和模型
    X, y, X_test, y_test = load_real_data()
    model, norm_stats = load_model_and_normalization()
    
    # 执行虚拟实验分析（使用测试集）
    create_virtual_experiment_plots(model, norm_stats, X_test, y_test)
    simulate_material_parameters(model, norm_stats, X_test)
    
    # 获取预测结果并生成指标报告（使用测试集）
    predictions = get_predictions(model, norm_stats, X_test)
    generate_comprehensive_metrics_report(y_test, predictions)
    
    # 计算统计指标的p值和置信区间
    print("\n" + "="*60)
    print("Generating statistical significance report (p-values and 95% CI)")
    print("="*60)
    
    p_value_results = {
        'module': '3revised_virtual_experiment',
        'description': 'Statistical significance report for virtual experiment analysis',
        'timestamp': pd.Timestamp.now().isoformat(),
        'test_set_statistics': {},
        'metrics_statistical_significance': {},
        'sensitivity_analysis_statistics': {}
    }
    
    # 测试集统计
    p_value_results['test_set_statistics'] = {
        'sample_size': int(len(y_test)),
        'feature_dimensions': int(X_test.shape[1]),
        'output_dimensions': int(y_test.shape[1])
    }
    
    # 计算预测指标的p值和置信区间
    metrics_stats = calculate_metrics_p_values(y_test, predictions)
    p_value_results['metrics_statistical_significance'] = {
        'biocompatibility': metrics_stats.get('output_0', {}),
        'osteogenic': metrics_stats.get('output_1', {}),
        'angiogenic': metrics_stats.get('output_2', {})
    }
    
    # 敏感性分析的统计显著性（如果有数据）
    sensitivity_stats = {}
    
    # 检查是否有敏感性分析结果
    sensitivity_path = RESULT_DIR / 'sensitivity_analysis_results.json'
    if sensitivity_path.exists():
        with open(sensitivity_path, 'r') as f:
            sensitivity_data = json.load(f)
        
        for feature, results in sensitivity_data.items():
            if isinstance(results, dict) and 'mean_effect' in results:
                effects = np.array([results.get('min_effect', 0), results.get('mean_effect', 0), results.get('max_effect', 0)])
                ci = calculate_confidence_interval(effects, confidence=0.95)
                sensitivity_stats[feature] = {
                    'mean_effect': float(results.get('mean_effect', 0)),
                    'confidence_interval_95': {
                        'lower_bound': ci['lower_bound'],
                        'upper_bound': ci['upper_bound'],
                        'confidence_level': ci['confidence_level']
                    }
                }
    
    p_value_results['sensitivity_analysis_statistics'] = sensitivity_stats
    
    # 保存p_value.json
    save_p_values_to_json(p_value_results, RESULT_DIR, 'p_value.json')
    print(f"[OK] p_value.json saved to: {RESULT_DIR / 'p_value.json'}")
    
    print("\n" + "="*60)
    print("[COMPLETE] Virtual experiment analysis completed!")
    print(f"[OUTPUT] Directory: {RESULT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()