import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 导入统计工具模块
try:
    from stats_utils import (
        calculate_confidence_interval,
        calculate_p_value_correlation,
        calculate_p_value_ttest,
        save_p_values_to_json
    )
except ImportError:
    print("Warning: stats_utils module not found, using inline implementations")

class DataImputer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.random_imputed = None
        self.mice_imputed = None
        self.missing_mask = None
        self.original_missing = None
    
    def load_data(self):
        """Load raw data"""
        self.data = pd.read_csv(self.data_path, sep='\t')
        print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        self.original_missing = self.data.isnull().sum().sum()
        print(f"Total missing values in raw data: {self.original_missing}")
        self.missing_mask = self.data.isnull()
        return self
    
    def random_imputation(self):
        """Random within-column imputation: sample from non-missing values"""
        df = self.data.copy()
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                non_missing = df[col].dropna().values
                if len(non_missing) > 0:
                    missing_idx = df[col].isnull()
                    n_missing = missing_idx.sum()
                    if n_missing > 0:
                        random_values = np.random.choice(non_missing, size=n_missing, replace=True)
                        df.loc[missing_idx, col] = random_values
        self.random_imputed = df
        print(f"Random imputation completed")
        return self
    
    def mice_imputation(self):
        """MICE imputation"""
        df = self.data.copy()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        imputer = IterativeImputer(
            estimator=None,
            max_iter=10,
            initial_strategy='mean',
            imputation_order='ascending',
            random_state=42
        )
        
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        self.mice_imputed = df
        print(f"MICE imputation completed")
        return self
    
    def diagnose_imputation(self):
        """Generate imputation diagnosis report"""
        report = "=" * 70 + "\n"
        report += "          Data Imputation Diagnosis Report\n"
        report += "=" * 70 + "\n\n"
        
        report += "1. Raw Data Statistics\n"
        report += "-" * 50 + "\n"
        report += f"   Samples: {self.data.shape[0]}\n"
        report += f"   Features: {self.data.shape[1]}\n"
        report += f"   Total missing values: {self.original_missing}\n"
        report += f"   Missing rate: {(self.original_missing / (self.data.shape[0] * self.data.shape[1])) * 100:.2f}%\n\n"
        
        report += "2. Missing Values by Column\n"
        report += "-" * 50 + "\n"
        missing_info = self.data.isnull().sum()
        for col, count in missing_info.items():
            if count > 0:
                rate = (count / self.data.shape[0]) * 100
                report += f"   {col:15s}: {count} missing values ({rate:.2f}%)\n"
        report += "\n"
        
        report += "3. Statistical Comparison After Imputation\n"
        report += "-" * 50 + "\n"
        
        for col in self.data.select_dtypes(include=['float64', 'int64']).columns:
            original_stats = self.data[col].describe()
            random_stats = self.random_imputed[col].describe()
            mice_stats = self.mice_imputed[col].describe()
            
            report += f"\n   Feature: {col}\n"
            report += f"           Original | Random    | MICE\n"
            report += f"   Mean:   {original_stats['mean']:10.4f} | {random_stats['mean']:10.4f} | {mice_stats['mean']:10.4f}\n"
            report += f"   Std:    {original_stats['std']:10.4f} | {random_stats['std']:10.4f} | {mice_stats['std']:10.4f}\n"
            report += f"   Min:    {original_stats['min']:10.4f} | {random_stats['min']:10.4f} | {mice_stats['min']:10.4f}\n"
            report += f"   Max:    {original_stats['max']:10.4f} | {random_stats['max']:10.4f} | {mice_stats['max']:10.4f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        return report
    
    def sensitivity_analysis(self, n_iterations=10):
        """Sensitivity analysis: evaluate imputation stability"""
        report = "=" * 70 + "\n"
        report += "          Imputation Sensitivity Analysis Report\n"
        report += "=" * 70 + "\n\n"
        
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        report += f"Analysis parameters: {n_iterations} random imputation iterations\n\n"
        report += "1. Random Imputation Sensitivity\n"
        report += "-" * 50 + "\n"
        
        for col in numeric_cols[:5]:  # Show first 5 columns only
            if self.data[col].isnull().sum() > 0:
                means = []
                stds = []
                for _ in range(n_iterations):
                    temp_df = self.data.copy()
                    non_missing = temp_df[col].dropna().values
                    missing_idx = temp_df[col].isnull()
                    random_values = np.random.choice(non_missing, size=missing_idx.sum(), replace=True)
                    temp_df.loc[missing_idx, col] = random_values
                    means.append(temp_df[col].mean())
                    stds.append(temp_df[col].std())
                
                report += f"   {col}:\n"
                report += f"     Mean range: {min(means):.4f} ~ {max(means):.4f} (variation: {max(means)-min(means):.4f})\n"
                report += f"     Std range:  {min(stds):.4f} ~ {max(stds):.4f} (variation: {max(stds)-min(stds):.4f})\n\n"
        
        report += "2. Imputation Method Comparison\n"
        report += "-" * 50 + "\n"
        
        report += "   Statistics   | Original(complete) | Random   | MICE\n"
        report += "   ------------|---------------------|----------|--------\n"
        
        for col in numeric_cols[:5]:
            complete_data = self.data[col].dropna()
            report += f"   {col:12s} | "
            report += f"μ={complete_data.mean():.4f} | "
            report += f"μ={self.random_imputed[col].mean():.4f} | "
            report += f"μ={self.mice_imputed[col].mean():.4f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        return report
    
    def recommend_method(self):
        """Recommend imputation method based on diagnosis"""
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        missing_rate = (self.original_missing / (self.data.shape[0] * len(numeric_cols))) * 100
        
        import time
        
        report = "=" * 70 + "\n"
        report += "          Imputation Method Recommendation\n"
        report += "=" * 70 + "\n\n"
        
        report += f"Data missing rate: {missing_rate:.2f}%\n\n"
        
        if missing_rate < 5:
            report += "Recommended: Random within-column imputation\n"
            report += "Reason: Low missing rate, random imputation is simple and efficient without introducing systematic bias\n"
            best_method = "random"
        elif missing_rate < 20:
            report += "Recommended: MICE imputation\n"
            report += "Reason: Moderate missing rate, MICE leverages feature correlations for more accurate imputation\n"
            best_method = "mice"
        else:
            report += "Recommended: MICE imputation (with domain knowledge)\n"
            report += "Reason: High missing rate, recommended to combine with domain knowledge for data quality inspection\n"
            best_method = "mice"
        
        report += "\nDetailed Comparison based on computed metrics:\n"
        report += "-" * 50 + "\n"
        
        stability_random = self._calculate_stability(self.data, self.random_imputed, numeric_cols)
        stability_mice = self._calculate_stability(self.data, self.mice_imputed, numeric_cols)
        
        report += f"1. Stability (mean deviation from original):\n"
        report += f"   Random Imputation: {stability_random:.4f}\n"
        report += f"   MICE Imputation:   {stability_mice:.4f}\n\n"
        
        corr_utilization_random = "No (independent sampling)"
        corr_utilization_mice = "Yes (uses feature correlations)"
        report += f"2. Correlation Utilization:\n"
        report += f"   Random Imputation: {corr_utilization_random}\n"
        report += f"   MICE Imputation:   {corr_utilization_mice}\n\n"
        
        start = time.time()
        self._run_random_imputation_test(self.data, numeric_cols)
        random_time = time.time() - start
        
        start = time.time()
        self._run_mice_imputation_test(self.data, numeric_cols)
        mice_time = time.time() - start
        
        report += f"3. Computational Efficiency (execution time):\n"
        report += f"   Random Imputation: {random_time*1000:.2f} ms\n"
        report += f"   MICE Imputation:   {mice_time*1000:.2f} ms\n\n"
        
        random_best_for = f"Missing rate < 5% (current: {missing_rate:.2f}%)"
        mice_best_for = f"Missing rate >= 5% (current: {missing_rate:.2f}%)"
        report += f"4. Best Suitable For:\n"
        report += f"   Random Imputation: {random_best_for}\n"
        report += f"   MICE Imputation:   {mice_best_for}\n"
        
        report += "\n" + "=" * 70 + "\n"
        return report, best_method
    
    def _calculate_stability(self, original, imputed, cols):
        """Calculate stability as mean relative deviation from original data"""
        total_dev = 0.0
        count = 0
        for col in cols:
            if original[col].isnull().sum() > 0:
                original_mean = original[col].dropna().mean()
                imputed_mean = imputed[col].mean()
                if original_mean != 0:
                    total_dev += abs(imputed_mean - original_mean) / abs(original_mean)
                    count += 1
        return total_dev / count if count > 0 else 0.0
    
    def _run_random_imputation_test(self, df, cols):
        """Test random imputation for efficiency measurement"""
        temp_df = df.copy()
        for col in cols:
            if temp_df[col].dtype in ['float64', 'int64']:
                non_missing = temp_df[col].dropna().values
                if len(non_missing) > 0:
                    missing_idx = temp_df[col].isnull()
                    if missing_idx.sum() > 0:
                        random_values = np.random.choice(non_missing, size=missing_idx.sum(), replace=True)
                        temp_df.loc[missing_idx, col] = random_values
    
    def _run_mice_imputation_test(self, df, cols):
        """Test MICE imputation for efficiency measurement"""
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        temp_df = df.copy()
        imputer = IterativeImputer(estimator=None, max_iter=10, initial_strategy='mean', 
                                   imputation_order='ascending', random_state=42)
        temp_df[cols] = imputer.fit_transform(temp_df[cols])
    
    def save_best_imputed_split(self, output_dir, best_method, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Select best imputed data and directly generate train/val/test split files."""
        os.makedirs(output_dir, exist_ok=True)
        
        if best_method == "random":
            data = self.random_imputed
        else:
            data = self.mice_imputed
        
        train, temp = train_test_split(data, train_size=train_ratio, random_state=42)
        val, test = train_test_split(temp, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)
        
        train.to_csv(os.path.join(output_dir, 'train.txt'), sep='\t', index=False)
        val.to_csv(os.path.join(output_dir, 'val.txt'), sep='\t', index=False)
        test.to_csv(os.path.join(output_dir, 'test.txt'), sep='\t', index=False)
        
        print(f"Best method ({best_method.upper()}) data split completed:")
        print(f"  - train.txt: {train.shape[0]} samples ({(train.shape[0]/data.shape[0])*100:.1f}%)")
        print(f"  - val.txt: {val.shape[0]} samples ({(val.shape[0]/data.shape[0])*100:.1f}%)")
        print(f"  - test.txt: {test.shape[0]} samples ({(test.shape[0]/data.shape[0])*100:.1f}%)")
        return train, val, test

def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'data.txt'
    impute_dir = project_root / 'data' / 'impute'
    split_dir = project_root / 'data'
    
    imputer = DataImputer(data_path)
    
    imputer.load_data()
    imputer.random_imputation()
    imputer.mice_imputation()
    
    diag_report = imputer.diagnose_imputation()
    sens_report = imputer.sensitivity_analysis()
    rec_report, best_method = imputer.recommend_method()
    
    os.makedirs(impute_dir, exist_ok=True)
    with open(os.path.join(impute_dir, 'diagnosis_report.txt'), 'w') as f:
        f.write(diag_report)
    with open(os.path.join(impute_dir, 'sensitivity_report.txt'), 'w') as f:
        f.write(sens_report)
    with open(os.path.join(impute_dir, 'recommendation.txt'), 'w') as f:
        f.write(rec_report)
    
    print("\n" + "="*70)
    print("Reports generated:")
    print(f"  - {os.path.join(impute_dir, 'diagnosis_report.txt')}")
    print(f"  - {os.path.join(impute_dir, 'sensitivity_report.txt')}")
    print(f"  - {os.path.join(impute_dir, 'recommendation.txt')}")
    print("="*70 + "\n")
    
    print("\n" + "="*70)
    print(f"Using {best_method.upper()} imputed data for splitting (no intermediate files saved)")
    print("="*70)
    
    imputer.save_best_imputed_split(split_dir, best_method)
    
    print("\n" + "="*70)
    print("Imputation Method Recommendation")
    print("="*70)
    print(rec_report)
    
    # 计算统计指标的p值和置信区间
    print("\n" + "="*70)
    print("Generating statistical significance report (p-values and 95% CI)")
    print("="*70)
    
    # 获取数据用于统计分析
    numeric_cols = imputer.data.select_dtypes(include=['float64', 'int64']).columns
    
    # 计算插补前后统计量的差异
    p_value_results = {
        'module': '0revised_data_impute_split',
        'description': 'Statistical significance report for data imputation',
        'timestamp': pd.Timestamp.now().isoformat(),
        'original_data_stats': {},
        'imputation_comparison': {},
        'split_statistics': {}
    }
    
    # 原始数据统计
    for col in numeric_cols[:5]:  # 取前5个特征进行统计分析
        original_data = imputer.data[col].dropna().values
        if len(original_data) >= 3:
            ci = calculate_confidence_interval(original_data, confidence=0.95)
            p_value_results['original_data_stats'][col] = {
                'mean': float(np.mean(original_data)),
                'std': float(np.std(original_data)),
                'sample_size': int(len(original_data)),
                'confidence_interval_95': {
                    'lower_bound': ci['lower_bound'],
                    'upper_bound': ci['upper_bound']
                }
            }
    
    # 插补方法比较（t检验）
    for col in numeric_cols[:5]:
        if imputer.data[col].isnull().sum() > 0:
            random_imputed = imputer.random_imputed[col].values
            mice_imputed = imputer.mice_imputed[col].values
            
            # 检验两种插补方法结果是否有显著差异
            ttest_result = calculate_p_value_ttest(random_imputed, mice_imputed)
            
            # 计算置信区间
            random_ci = calculate_confidence_interval(random_imputed)
            mice_ci = calculate_confidence_interval(mice_imputed)
            
            p_value_results['imputation_comparison'][col] = {
                'random_imputation': {
                    'mean': float(np.mean(random_imputed)),
                    'confidence_interval_95': {
                        'lower_bound': random_ci['lower_bound'],
                        'upper_bound': random_ci['upper_bound']
                    }
                },
                'mice_imputation': {
                    'mean': float(np.mean(mice_imputed)),
                    'confidence_interval_95': {
                        'lower_bound': mice_ci['lower_bound'],
                        'upper_bound': mice_ci['upper_bound']
                    }
                },
                'ttest_between_methods': {
                    't_statistic': ttest_result['t_statistic'],
                    'p_value': ttest_result['p_value'],
                    'df': ttest_result['df'],
                    'mean_difference': ttest_result['mean_diff']
                }
            }
    
    # 分割统计
    data = imputer.random_imputed if best_method == "random" else imputer.mice_imputed
    train, temp = train_test_split(data, train_size=train_ratio, random_state=42)
    val, test = train_test_split(temp, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)
    
    p_value_results['split_statistics'] = {
        'total_samples': int(len(data)),
        'train_samples': int(len(train)),
        'val_samples': int(len(val)),
        'test_samples': int(len(test)),
        'train_ratio': float(train_ratio),
        'val_ratio': float(val_ratio),
        'test_ratio': float(test_ratio),
        'best_imputation_method': best_method
    }
    
    # 保存p_value.json
    save_p_values_to_json(p_value_results, impute_dir, 'p_value.json')
    print(f"[OK] p_value.json saved to: {os.path.join(impute_dir, 'p_value.json')}")

if __name__ == "__main__":
    main()