#!/usr/bin/env python3
"""
SHAP分析脚本 - 对三个模块进行综合解释性分析
包含全局解释、局部解释、交互依赖可视化等
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import joblib
import shap
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

# 路径配置
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULT_DIR = PROJECT_DIR / "results"

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
OUTPUT_NAMES = ['Biocompatibility', 'Osteogenic', 'Angiogenic']


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


def load_data():
    """加载数据"""
    train_data = pd.read_csv(DATA_DIR / 'train.txt', sep='\t')
    val_data = pd.read_csv(DATA_DIR / 'val.txt', sep='\t')
    test_data = pd.read_csv(DATA_DIR / 'test.txt', sep='\t')
    
    # 包含BSG作为输入特征（23个特征）
    X_train = train_data[INPUT_FEATURES].values.astype(np.float32)
    X_val = val_data[INPUT_FEATURES].values.astype(np.float32)
    X_test = test_data[INPUT_FEATURES].values.astype(np.float32)
    
    # 计算目标值（使用指标组的平均值并应用softmax）
    def compute_targets(df):
        bio_scores = df[BIO_INDICATORS].mean(axis=1).values
        osteo_scores = df[OSTEO_INDICATORS].mean(axis=1).values
        angio_scores = df[ANGIO_INDICATORS].mean(axis=1).values
        
        raw_scores = np.stack([bio_scores, osteo_scores, angio_scores], axis=1)
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        targets = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return targets.astype(np.float32)
    
    y_train = compute_targets(train_data)
    y_val = compute_targets(val_data)
    y_test = compute_targets(test_data)
    
    X = np.vstack([X_train, X_val, X_test])
    y = np.vstack([y_train, y_val, y_test])
    
    return X, y, X_train, X_test, y_train, y_test


def load_model_and_normalization(model_path, norm_path):
    """加载模型和归一化参数"""
    norm_stats = joblib.load(norm_path)
    model = ConstrainedRegressor(input_dim=23)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, norm_stats


def normalize_data(X, norm_stats):
    """归一化数据"""
    mean = np.array(norm_stats["mean"])
    std = np.array(norm_stats["std"])
    return (X - mean) / std


def create_shap_explainer(model, X_scaled):
    """创建SHAP解释器"""
    def model_wrapper(x):
        model.eval()
        with torch.no_grad():
            return model(torch.FloatTensor(x)).numpy()
    
    # 使用Kernel SHAP进行解释（适用于复杂模型）
    background = X_scaled[np.random.choice(X_scaled.shape[0], 100, replace=False)]
    explainer = shap.KernelExplainer(model_wrapper, background)
    
    return explainer


def plot_global_explanations(shap_values, X_scaled, feature_names, output_names, save_dir):
    """生成全局解释可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Beeswarm Plot - 每个输出维度
    for i, output_name in enumerate(output_names):
        plt.figure(figsize=(12, 8))
        expl = shap.Explanation(values=shap_values[:, :, i], data=X_scaled, feature_names=feature_names)
        shap.plots.beeswarm(expl, show=False)
        plt.title(f'Beeswarm Plot - {output_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f'beeswarm_{output_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Bar Plot - 特征重要性
    for i, output_name in enumerate(output_names):
        plt.figure(figsize=(10, 6))
        expl = shap.Explanation(values=shap_values[:, :, i], data=X_scaled, feature_names=feature_names)
        shap.plots.bar(expl, show=False)
        plt.title(f'Feature Importance - {output_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f'barplot_{output_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Heatmap - 样本x特征的SHAP值
    for i, output_name in enumerate(output_names):
        plt.figure(figsize=(14, 10))
        shap_values_np = shap_values[:, :, i]
        sns.heatmap(shap_values_np, 
                    xticklabels=feature_names, 
                    yticklabels=False,
                    cmap='RdBu_r', 
                    center=0,
                    vmin=-np.max(np.abs(shap_values_np)),
                    vmax=np.max(np.abs(shap_values_np)))
        plt.title(f'SHAP Value Heatmap - {output_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f'heatmap_{output_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] 全局解释可视化已保存到 {save_dir}")


def plot_local_explanations(explainer, shap_values, X_scaled, feature_names, output_names, save_dir, sample_indices=[0, 1, 2]):
    """生成局部解释可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in sample_indices:
        # 1. Waterfall Plot
        for i, output_name in enumerate(output_names):
            plt.figure(figsize=(10, 6))
            expl = shap.Explanation(values=shap_values[idx, :, i], 
                                   base_values=explainer.expected_value[i],
                                   data=X_scaled[idx], 
                                   feature_names=feature_names)
            shap.plots.waterfall(expl, show=False)
            plt.title(f'Waterfall Plot - Sample {idx} - {output_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(save_dir / f'waterfall_sample{idx}_{output_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"[OK] 局部解释可视化已保存到 {save_dir}")


def plot_interaction_dependence(shap_values, X_scaled, X_raw, feature_names, output_names, save_dir):
    """生成交互依赖可视化（使用原始特征值）"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择最重要的几个特征进行交互分析
    top_features = ['AF', 'Ani', 'Ecc', 'EqD', 'prolif1', 'prolif2']
    
    for i, output_name in enumerate(output_names):
        # 使用原始特征值进行可视化
        expl = shap.Explanation(values=shap_values[:, :, i], data=X_raw, feature_names=feature_names)
        for feature_name in top_features[:4]:
            plt.figure(figsize=(10, 6))
            shap.plots.scatter(expl[:, feature_name], color=expl[:, 'vvolume'], show=False)
            plt.title(f'Dependence Plot - {feature_name} - {output_name}', fontsize=14)
            plt.xlabel(f'{feature_name} (Original Scale)', fontsize=12)
            plt.ylabel(f'SHAP Value for {output_name}', fontsize=12)
            plt.tight_layout()
            plt.savefig(save_dir / f'dependence_{feature_name.lower()}_{output_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"[OK] 交互依赖可视化已保存到 {save_dir}")


def compute_shap_metrics(shap_values, explainer, X_scaled, y_true, y_pred, feature_names, output_names, save_dir):
    """计算并保存SHAP相关指标"""
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = {
        'baseline_values': explainer.expected_value.tolist(),
        'feature_importance': {},
        'shap_summary': {}
    }
    
    # 计算每个输出维度的特征重要性（平均绝对SHAP值）
    for i, output_name in enumerate(output_names):
        mean_abs_shap = np.mean(np.abs(shap_values[:, :, i]), axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        metrics['feature_importance'][output_name] = importance_df.to_dict('records')
        
        # 保存SHAP值矩阵
        np.save(save_dir / f'shap_values_{output_name.lower().replace(" ", "_")}.npy', shap_values[:, :, i])
    
    # 计算SHAP值的统计信息
    metrics['shap_summary']['mean'] = np.mean(shap_values, axis=(0, 1)).tolist()
    metrics['shap_summary']['std'] = np.std(shap_values, axis=(0, 1)).tolist()
    metrics['shap_summary']['min'] = np.min(shap_values, axis=(0, 1)).tolist()
    metrics['shap_summary']['max'] = np.max(shap_values, axis=(0, 1)).tolist()
    
    # 保存指标到JSON
    with open(save_dir / 'shap_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 保存特征重要性CSV
    for output_name in output_names:
        df = pd.DataFrame(metrics['feature_importance'][output_name])
        df.to_csv(save_dir / f'feature_importance_{output_name.lower().replace(" ", "_")}.csv', index=False)
    
    print(f"[OK] SHAP指标已保存到 {save_dir}")
    return metrics


def run_shap_analysis(model_path, norm_path, result_subdir, model_name):
    """运行SHAP分析"""
    print(f"\n{'='*60}")
    print(f"SHAP分析 - {model_name}")
    print(f"{'='*60}")
    
    # 创建结果目录
    save_dir = RESULT_DIR / result_subdir
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    X, y, X_train, X_test, y_train, y_test = load_data()
    
    # 加载模型和归一化参数
    model, norm_stats = load_model_and_normalization(model_path, norm_path)
    
    # 保留原始数据（用于交互依赖可视化）
    X_raw = X_test[:100]
    
    # 归一化数据
    X_scaled = normalize_data(X_test[:100], norm_stats)  # 使用测试集的前100个样本进行分析
    
    # 创建SHAP解释器
    print("[INFO] Creating SHAP explainer...")
    explainer = create_shap_explainer(model, X_scaled)
    
    # 计算SHAP值
    print("[INFO] Computing SHAP values...")
    shap_values = explainer.shap_values(X_scaled)
    
    # 获取预测值
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_scaled)).numpy()
    
    # 生成全局解释可视化
    print("[INFO] Generating global explanation visualizations...")
    plot_global_explanations(shap_values, X_scaled, FEATURE_NAMES, OUTPUT_NAMES, save_dir)
    
    # 生成局部解释可视化
    print("[INFO] Generating local explanation visualizations...")
    plot_local_explanations(explainer, shap_values, X_scaled, FEATURE_NAMES, OUTPUT_NAMES, save_dir)
    
    # 生成交互依赖可视化（使用原始特征值）
    print("[INFO] Generating interaction dependence visualizations...")
    plot_interaction_dependence(shap_values, X_scaled, X_raw, FEATURE_NAMES, OUTPUT_NAMES, save_dir)
    
    # 计算和保存SHAP指标
    print("[INFO] Computing SHAP metrics...")
    metrics = compute_shap_metrics(shap_values, explainer, X_scaled, y_test[:100], y_pred, FEATURE_NAMES, OUTPUT_NAMES, save_dir)
    
    # 打印特征重要性摘要
    print("\n[SUMMARY] Feature importance:")
    for output_name in OUTPUT_NAMES:
        print(f"\n{output_name}:")
        top_features = metrics['feature_importance'][output_name][:5]
        for feat in top_features:
            print(f"  {feat['feature']}: {feat['mean_abs_shap']:.4f}")
    
    print(f"\n[OK] {model_name} SHAP analysis complete!")
    print(f"[DIR] Results saved to: {save_dir}")


def main():
    """主函数"""
    print("="*60)
    print("SHAP分析脚本 - 对三个模块进行综合解释性分析")
    print("="*60)
    
    # 定义三个模块的路径
    modules = [
        {
            'name': 'Forward Model',
            'model_path': PROJECT_DIR / 'results' / '1forward' / 'model_checkpoints' / 'final_model.pth',
            'norm_path': PROJECT_DIR / 'results' / '1forward' / 'training_logs' / 'normalization_stats.joblib',
            'result_dir': '4shap_forward'
        },
        {
            'name': 'Reverse Model',
            'model_path': PROJECT_DIR / 'results' / '1forward' / 'model_checkpoints' / 'final_model.pth',
            'norm_path': PROJECT_DIR / 'results' / '1forward' / 'training_logs' / 'normalization_stats.joblib',
            'result_dir': '4shap_reverse'
        },
        {
            'name': 'Virtual Experiment',
            'model_path': PROJECT_DIR / 'results' / '1forward' / 'model_checkpoints' / 'final_model.pth',
            'norm_path': PROJECT_DIR / 'results' / '1forward' / 'training_logs' / 'normalization_stats.joblib',
            'result_dir': '4shap_virtual'
        }
    ]
    
    # 对每个模块运行SHAP分析
    for module in modules:
        run_shap_analysis(module['model_path'], 
                         module['norm_path'], 
                         module['result_dir'], 
                         module['name'])
    
    print("\n" + "="*60)
    print("[OK] SHAP analysis completed for all modules!")
    print("="*60)


if __name__ == "__main__":
    main()
