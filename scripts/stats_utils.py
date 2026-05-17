"""
统计计算工具函数模块
提供p值计算和置信区间估计功能
"""

import numpy as np
from scipy import stats


def calculate_confidence_interval(data, confidence=0.95):
    """
    计算均值的置信区间
    
    Args:
        data: 数值数组或列表
        confidence: 置信水平，默认0.95
        
    Returns:
        dict: 包含均值、标准差、置信区间上下限的字典
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # 使用样本标准差
    
    if n == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'sample_size': 0,
            'confidence_level': confidence,
            'lower_bound': 0.0,
            'upper_bound': 0.0
        }
    
    # 计算标准误差
    se = std / np.sqrt(n)
    
    # 计算置信区间
    margin_of_error = stats.t.ppf((1 + confidence) / 2, n - 1) * se
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return {
        'mean': float(mean),
        'std': float(std),
        'sample_size': int(n),
        'confidence_level': float(confidence),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'margin_of_error': float(margin_of_error)
    }


def calculate_p_value_correlation(x, y):
    """
    计算皮尔逊相关系数的p值
    
    Args:
        x: 第一个变量
        y: 第二个变量
        
    Returns:
        dict: 包含相关系数r和p值的字典
    """
    x = np.array(x)
    y = np.array(y)
    
    if len(x) != len(y):
        raise ValueError("x和y必须具有相同的长度")
    
    if len(x) < 3:
        return {
            'r': 0.0,
            'p_value': 1.0,
            'sample_size': len(x)
        }
    
    r, p = stats.pearsonr(x, y)
    
    return {
        'r': float(r),
        'p_value': float(p),
        'sample_size': int(len(x))
    }


def calculate_p_value_ttest(sample1, sample2):
    """
    计算两个独立样本的t检验p值
    
    Args:
        sample1: 第一个样本
        sample2: 第二个样本
        
    Returns:
        dict: 包含t统计量和p值的字典
    """
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    if len(sample1) < 2 or len(sample2) < 2:
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'df': 0,
            'mean_diff': 0.0
        }
    
    t_stat, p_val = stats.ttest_ind(sample1, sample2)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'df': int(len(sample1) + len(sample2) - 2),
        'mean_diff': float(np.mean(sample1) - np.mean(sample2))
    }


def calculate_p_value_one_sample_ttest(sample, popmean=0):
    """
    计算单样本t检验p值
    
    Args:
        sample: 样本数据
        popmean: 总体均值假设，默认0
        
    Returns:
        dict: 包含t统计量和p值的字典
    """
    sample = np.array(sample)
    
    if len(sample) < 2:
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'df': 0
        }
    
    t_stat, p_val = stats.ttest_1samp(sample, popmean)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'df': int(len(sample) - 1),
        'sample_mean': float(np.mean(sample)),
        'hypothesized_mean': float(popmean)
    }


def calculate_p_value_anova(*groups):
    """
    计算方差分析(ANOVA)的p值
    
    Args:
        *groups: 多个样本组
        
    Returns:
        dict: 包含F统计量和p值的字典
    """
    groups = [np.array(g) for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {
            'f_statistic': 0.0,
            'p_value': 1.0,
            'df_between': 0,
            'df_within': 0
        }
    
    f_stat, p_val = stats.f_oneway(*groups)
    
    n_total = sum(len(g) for g in groups)
    df_between = len(groups) - 1
    df_within = n_total - len(groups)
    
    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_val),
        'df_between': int(df_between),
        'df_within': int(df_within)
    }


def calculate_metrics_p_values(true_values, predicted_values):
    """
    计算预测指标的统计显著性（p值和置信区间）
    
    Args:
        true_values: 真实值数组
        predicted_values: 预测值数组
        
    Returns:
        dict: 包含各种指标及其统计显著性的字典
    """
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    
    if true_values.ndim > 1:
        n_outputs = true_values.shape[1]
    else:
        n_outputs = 1
        true_values = true_values.reshape(-1, 1)
        predicted_values = predicted_values.reshape(-1, 1)
    
    results = {}
    
    for i in range(n_outputs):
        y_true = true_values[:, i]
        y_pred = predicted_values[:, i]
        
        # 计算残差
        residuals = y_true - y_pred
        
        # 计算R²的p值（通过F检验）
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum(residuals ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        n = len(y_true)
        k = 1  # 预测变量数量
        df_reg = k
        df_res = n - k - 1
        
        if df_res > 0 and ss_tot > 0:
            f_stat = (r2 / df_reg) / ((1 - r2) / df_res)
            p_value_r2 = stats.f.sf(f_stat, df_reg, df_res)
        else:
            f_stat = 0.0
            p_value_r2 = 1.0
        
        # 计算MAE的置信区间
        mae = np.mean(np.abs(residuals))
        mae_ci = calculate_confidence_interval(np.abs(residuals))
        
        # 计算RMSE的置信区间
        rmse = np.sqrt(np.mean(residuals ** 2))
        rmse_ci = calculate_confidence_interval(residuals ** 2)
        rmse_ci['mean'] = np.sqrt(rmse_ci['mean'])
        rmse_ci['lower_bound'] = np.sqrt(max(0, rmse_ci['lower_bound']))
        rmse_ci['upper_bound'] = np.sqrt(rmse_ci['upper_bound'])
        
        # 计算皮尔逊相关系数及其p值
        corr_result = calculate_p_value_correlation(y_true, y_pred)
        
        results[f'output_{i}'] = {
            'r2': {
                'value': float(r2),
                'p_value': float(p_value_r2),
                'f_statistic': float(f_stat),
                'df_regression': int(df_reg),
                'df_residual': int(df_res)
            },
            'mae': {
                'value': float(mae),
                'confidence_interval': {
                    'mean': mae_ci['mean'],
                    'lower_bound': mae_ci['lower_bound'],
                    'upper_bound': mae_ci['upper_bound'],
                    'confidence_level': mae_ci['confidence_level']
                }
            },
            'rmse': {
                'value': float(rmse),
                'confidence_interval': {
                    'mean': rmse_ci['mean'],
                    'lower_bound': rmse_ci['lower_bound'],
                    'upper_bound': rmse_ci['upper_bound'],
                    'confidence_level': rmse_ci['confidence_level']
                }
            },
            'pearson_correlation': corr_result,
            'sample_size': int(n)
        }
    
    return results


def calculate_cv_metrics_p_values(cv_metrics, metric_name='MSE'):
    """
    计算交叉验证指标的统计显著性
    
    Args:
        cv_metrics: 各折的指标值列表
        metric_name: 指标名称
        
    Returns:
        dict: 包含均值、标准差、置信区间和p值的字典
    """
    cv_metrics = np.array(cv_metrics)
    
    n_folds = len(cv_metrics)
    mean = np.mean(cv_metrics)
    std = np.std(cv_metrics, ddof=1)
    
    # 计算95%置信区间
    ci = calculate_confidence_interval(cv_metrics, confidence=0.95)
    
    # 单样本t检验：检验均值是否显著大于0
    t_test_result = calculate_p_value_one_sample_ttest(cv_metrics, popmean=0)
    
    return {
        'metric': metric_name,
        'mean': float(mean),
        'std': float(std),
        'n_folds': int(n_folds),
        'confidence_interval': {
            'lower_bound': ci['lower_bound'],
            'upper_bound': ci['upper_bound'],
            'confidence_level': ci['confidence_level']
        },
        't_test': t_test_result
    }


def save_p_values_to_json(results, output_dir, filename='p_value.json'):
    """
    将统计结果保存为JSON文件
    
    Args:
        results: 统计结果字典
        output_dir: 输出目录
        filename: 文件名，默认为p_value.json
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"统计结果已保存到: {output_path}")


if __name__ == "__main__":
    # 测试示例
    np.random.seed(42)
    
    # 测试置信区间计算
    sample_data = np.random.normal(100, 15, 50)
    ci_result = calculate_confidence_interval(sample_data)
    print("置信区间测试:")
    print(ci_result)
    
    # 测试相关性p值计算
    x = np.random.normal(0, 1, 100)
    y = x + np.random.normal(0, 0.5, 100)
    corr_result = calculate_p_value_correlation(x, y)
    print("\n相关性p值测试:")
    print(corr_result)
    
    # 测试t检验
    sample1 = np.random.normal(50, 10, 30)
    sample2 = np.random.normal(55, 10, 30)
    ttest_result = calculate_p_value_ttest(sample1, sample2)
    print("\nt检验测试:")
    print(ttest_result)
    
    # 测试指标p值计算
    y_true = np.random.rand(100, 3)
    y_pred = y_true + np.random.normal(0, 0.1, y_true.shape)
    metrics_result = calculate_metrics_p_values(y_true, y_pred)
    print("\n指标p值测试:")
    print(json.dumps(metrics_result, indent=2))
