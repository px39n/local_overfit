"""
Figure 5 Supporting Functions

This module contains model fitting, evaluation, and visualization helper functions
for the accuracy surface analysis.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
from tqdm.auto import tqdm

from sklearn.preprocessing import PolynomialFeatures, StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from pygam import LinearGAM, s, l, f, te

# Optional: LightGBM (use fallback if not installed)
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# ============================================================
# Helper Functions (ported from OG_transformer.plot.aggregate_bin)
# ============================================================

def extract_model_names(df, lower=True):
    """Extract model names from predicted columns"""
    predicted_cols = [col for col in df.columns if 'predicted' in col.lower()]
    if lower:
        model_names = [col.lower().replace('predicted_', '') if 'predicted_' in col.lower() else 'default' for col in predicted_cols]
    else:
        model_names = [col.replace('predicted_', '') if 'predicted_' in col else 'default' for col in predicted_cols]
    return dict(zip(predicted_cols, model_names))


def calculate_score(observed, predicted, metric):
    """Calculate performance score"""
    if metric == 'r2':
        score = r2_score(observed, predicted)
        return max(0, score)  # clip negative R2 to 0
    elif metric == 'rmse':
        from sklearn.metrics import mean_squared_error
        return np.sqrt(mean_squared_error(observed, predicted))
    else:  # nrmse
        from sklearn.metrics import mean_squared_error
        return np.sqrt(mean_squared_error(observed, predicted)) / observed.std()


def calculate_r2_with_bins(df, sparsity_bins, use_seeds=True, verbose=False, target_model=None, sufficiency_bins=None, remove_outliers=False):
    """
    Calculate R² data with binning.
    
    Args:
        df: DataFrame with predicted columns
        sparsity_bins: Number of sparsity bins
        use_seeds: Whether to aggregate by seeds
        verbose: Print debug information
        target_model: If specified, only calculate for this model
        sufficiency_bins: If specified, bin sufficiency values
        remove_outliers: Whether to remove outliers using IQR method
    
    Returns:
        DataFrame with binned R² results
    """
    # Get model columns
    model_columns = [col for col in df.columns if col.startswith('predicted_')]
    
    # If target model specified, only process that model
    if target_model is not None:
        target_col = f'predicted_{target_model}'
        if target_col in model_columns:
            model_columns = [target_col]
        else:
            raise ValueError(f"Model {target_model} not found in data")
    
    # Create grouping variables
    df_copy = df.copy()
    sparsity_cut_result = pd.cut(df_copy['sparsity'], bins=sparsity_bins, labels=False, retbins=True)
    df_copy['sparsity_bin'] = sparsity_cut_result[0]
    actual_sparsity_bins = sparsity_cut_result[1]
    
    if verbose:
        sparsity_centers = [(actual_sparsity_bins[i] + actual_sparsity_bins[i+1])/2 for i in range(len(actual_sparsity_bins)-1)]
        print(f"Sparsity bin edges: {actual_sparsity_bins}")
        print(f"Sparsity bin centers: {sparsity_centers}")
    
    # Handle sufficiency binning
    unique_suff_values = np.sort(df_copy['sufficiency'].unique())
    n_unique = len(unique_suff_values)
    
    if sufficiency_bins is None:
        sufficiency_bins = n_unique
    
    values_per_bin = n_unique // sufficiency_bins
    remainder = n_unique % sufficiency_bins
    
    # Create bin mapping
    suff_to_bin = {}
    current_idx = 0
    
    for bin_idx in range(sufficiency_bins):
        current_bin_size = values_per_bin + (1 if bin_idx < remainder else 0)
        for i in range(current_bin_size):
            if current_idx < n_unique:
                suff_to_bin[unique_suff_values[current_idx]] = bin_idx
                current_idx += 1
    
    df_copy['sufficiency_bin'] = df_copy['sufficiency'].map(suff_to_bin)
    
    # Determine grouping
    base_group_cols = ['sufficiency_bin', 'sparsity_bin']
    
    if use_seeds and 'seed' in df_copy.columns:
        group_cols = base_group_cols + ['seed']
    else:
        group_cols = base_group_cols
        use_seeds = False
    
    # Calculate R²
    results = []
    for name, group in df_copy.groupby(group_cols):
        if len(group) < 10:
            continue
            
        for model_col in model_columns:
            observed = group['observed'].values
            predicted = group[model_col].values
            valid_mask = ~(np.isnan(observed) | np.isnan(predicted) | 
                         np.isinf(observed) | np.isinf(predicted))
            if valid_mask.sum() < 5:
                continue
                
            r2 = r2_score(observed[valid_mask], predicted[valid_mask])
            
            result = {
                'sufficiency_bin': name[0],
                'sufficiency': group['sufficiency'].mean(),
                'sparsity_bin': name[1],
                'sparsity': group['sparsity'].mean(),
                'model': model_col.replace('predicted_', ''),
                'r2': r2,
                'n_samples': valid_mask.sum()
            }
            if use_seeds:
                result['seed'] = name[2]
            
            results.append(result)
    
    df_results = pd.DataFrame(results)
    
    # Aggregate if using seeds
    if use_seeds and len(df_results) > 0:
        agg_cols = ['sufficiency_bin', 'sparsity_bin', 'model']
        
        df_final = df_results.groupby(agg_cols).agg({
            'r2': ['mean', 'std', 'count'],
            'sufficiency': 'mean',
            'sparsity': 'mean',
            'n_samples': 'sum'
        }).reset_index()
        
        df_final.columns = ['sufficiency_bin', 'sparsity_bin', 'model', 'r2', 'r2_std', 'n_seeds', 'sufficiency', 'sparsity', 'total_samples']
        df_final['r2_std'] = df_final['r2_std'].fillna(0)
        df_final = df_final[['sufficiency_bin', 'sufficiency', 'sparsity_bin', 'sparsity', 'model', 'r2', 'r2_std', 'n_seeds', 'total_samples']]
    else:
        df_final = df_results.copy()
        df_final['r2_std'] = 0.0
        df_final['n_seeds'] = 1
        df_final['total_samples'] = df_final['n_samples']
        df_final = df_final[['sufficiency_bin', 'sufficiency', 'sparsity_bin', 'sparsity', 'model', 'r2', 'r2_std', 'n_seeds', 'total_samples']]
    
    if verbose:
        print(f"Generated {len(df_final)} data points for GAM fitting")
        if use_seeds:
            print(f"Average std across all points: {df_final['r2_std'].mean():.4f}")
    
    if remove_outliers:
        df_final_list = []
        total_removed = 0
        
        for model_name in df_final['model'].unique():
            model_data = df_final[df_final['model'] == model_name].copy()
            original_count = len(model_data)
            
            q1 = model_data['r2'].quantile(0.25)
            q3 = model_data['r2'].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            model_data_clean = model_data[(model_data['r2'] >= lower_bound) & (model_data['r2'] <= upper_bound)]
            removed_count = original_count - len(model_data_clean)
            total_removed += removed_count
            
            if verbose and removed_count > 0:
                print(f"  {model_name}: Removed {removed_count}/{original_count} outliers (R² < {lower_bound:.3f} or > {upper_bound:.3f})")
            
            df_final_list.append(model_data_clean)
        
        df_final = pd.concat(df_final_list, ignore_index=True)
        
        if verbose:
            print(f"Outlier removal: {total_removed} total points removed across all models")
    
    return df_final


def find_optimal_gam_params(df, sparsity_bins, target_model='Ours', use_seeds=True, 
                           lam_range=(0.1, 100), spline_range=(4, 20)):
    """
    Find optimal GAM hyperparameters using ternary search.
    
    Args:
        df: DataFrame
        sparsity_bins: Number of sparsity bins
        target_model: Target model name
        use_seeds: Whether to use seeds aggregation
        lam_range: Lambda parameter search range (min, max)
        spline_range: Spline parameter search range (min, max)
    
    Returns:
        tuple: (best_lam, best_spline, best_score)
    """
    import math
    
    target_col = f'predicted_{target_model}'
    if target_col not in df.columns:
        print(f"Error: Model {target_model} not found in data!")
        return 1.0, 8, -999
    
    other_predicted_cols = [col for col in df.columns if col.startswith('predicted_') and col != target_col]
    df_filtered = df.drop(columns=other_predicted_cols)
    
    df_bins = calculate_r2_with_bins(
        df_filtered, sparsity_bins, use_seeds, verbose=False, target_model=target_model
    )
    
    if len(df_bins) < 15:
        print("Error: Insufficient data for GAM fitting")
        return 1.0, 8, -999
    
    model_data = df_bins.copy()
    model_data['sufficiency_log'] = np.log10(model_data['sufficiency'])
    X = model_data[['sufficiency_log', 'sparsity']].values
    y = model_data['r2'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    def evaluate_gam(lam, spline):
        try:
            gam_mono = LinearGAM(
                s(0, constraints='monotonic_inc', n_splines=spline, lam=lam) + 
                s(1, constraints='monotonic_inc', n_splines=spline, lam=lam)
            )
            gam_mono.fit(X_scaled, y)
            return gam_mono.statistics_['pseudo_r2']['explained_deviance']
        except:
            return -999
    
    # Stage 1: Search for optimal lam with fixed spline=8
    best_lam = 1.0
    best_score = -999
    
    log_lam_min, log_lam_max = math.log10(lam_range[0]), math.log10(lam_range[1])
    tested_lams = {}
    
    def get_lam_score(log_lam):
        if log_lam not in tested_lams:
            lam = 10 ** log_lam
            score = evaluate_gam(lam, 8)
            tested_lams[log_lam] = score
        return tested_lams[log_lam]
    
    left, right = log_lam_min, log_lam_max
    while right - left > 0.2:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        score1 = get_lam_score(mid1)
        score2 = get_lam_score(mid2)
        
        if score1 > best_score:
            best_score = score1
            best_lam = 10 ** mid1
        if score2 > best_score:
            best_score = score2
            best_lam = 10 ** mid2
        
        if score1 < score2:
            left = mid1
        else:
            right = mid2
    
    # Stage 2: Search for optimal spline with best lam
    best_spline = 8
    tested_splines = {}
    
    def get_spline_score(spline):
        if spline not in tested_splines:
            score = evaluate_gam(best_lam, spline)
            tested_splines[spline] = score
        return tested_splines[spline]
    
    left, right = spline_range[0], spline_range[1]
    while right - left > 2:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        score1 = get_spline_score(mid1)
        score2 = get_spline_score(mid2)
        
        if score1 > best_score:
            best_score = score1
            best_spline = mid1
        if score2 > best_score:
            best_score = score2
            best_spline = mid2
        
        if score1 < score2:
            left = mid1
        else:
            right = mid2
    
    for spline in range(left, right + 1):
        score = get_spline_score(spline)
        if score > best_score:
            best_score = score
            best_spline = spline
    
    # Stage 3: Fine-tune lam with best spline
    current_log_lam = math.log10(best_lam)
    search_range = 0.5
    
    left = max(log_lam_min, current_log_lam - search_range)
    right = min(log_lam_max, current_log_lam + search_range)
    
    while right - left > 0.1:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        lam1, lam2 = 10 ** mid1, 10 ** mid2
        score1 = evaluate_gam(lam1, best_spline)
        score2 = evaluate_gam(lam2, best_spline)
        
        if score1 > best_score:
            best_score = score1
            best_lam = lam1
        if score2 > best_score:
            best_score = score2
            best_lam = lam2
        
        if score1 < score2:
            left = mid1
        else:
            right = mid2
    
    return best_lam, best_spline, best_score


def find_optimal_sparsity_bins(df, min_bins=5, max_bins=25, use_seeds=True, target_model='Ours',
                              lam=1.0, spline=8):
    """
    Find optimal sparsity bin count using ternary search.
    
    Args:
        df: DataFrame
        min_bins: Minimum bin count
        max_bins: Maximum bin count
        use_seeds: Whether to use seeds aggregation
        target_model: Target model name
        lam: GAM lambda parameter
        spline: GAM spline parameter
    
    Returns:
        int: Optimal bin count
    """
    target_col = f'predicted_{target_model}'
    if target_col not in df.columns:
        print(f"Error: Model {target_model} not found in data!")
        available_models = [col.replace('predicted_', '') for col in df.columns if col.startswith('predicted_')]
        print(f"Available models: {available_models}")
        return min_bins
    
    other_predicted_cols = [col for col in df.columns if col.startswith('predicted_') and col != target_col]
    df_filtered = df.drop(columns=other_predicted_cols)
    
    def evaluate_bins(n_bins):
        try:
            # Use local fit_gam_models instead of importing from OG_transformer
            fitted_models = fit_gam_models(
                df_filtered, n_bins, use_seeds, spline=spline, lam=lam, verbose=False
            )
            
            if target_model not in fitted_models:
                return -999
            
            model_info = fitted_models[target_model]
            r2_val = model_info['r2_score']
            
            return r2_val
                
        except Exception as e:
            return -999
    
    left, right = min_bins, max_bins
    best_bins, best_score = min_bins, -999
    
    tested_points = {}
    
    def get_score(bins):
        if bins not in tested_points:
            score = evaluate_bins(bins)
            tested_points[bins] = score
        return tested_points[bins]
    
    # Ternary search
    while right - left > 2:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        score1 = get_score(mid1)
        score2 = get_score(mid2)
        
        if score1 > best_score:
            best_score = score1
            best_bins = mid1
        if score2 > best_score:
            best_score = score2
            best_bins = mid2
        
        if score1 < score2:
            left = mid1
        else:
            right = mid2
    
    # Check remaining points
    for bins in range(left, right + 1):
        score = get_score(bins)
        if score > best_score:
            best_score = score
            best_bins = bins
    
    print(f"GAM parameters: lam={lam:.3f}, spline={spline}")
    print(f"Optimal bins: {best_bins} (R²={best_score:.4f})")
    
    return best_bins

def fit_accuracy_surface(results_dict, model_type='svm', poly_degree=2):
    """
    对准确率结果进行表面拟合
    
    Args:
        results_dict: 包含sufficiency, sparsity, accuracy数据的字典
        model_type: 使用的模型类型 ('svm', 'lightgbm', 'linear', 'gam_free')
        poly_degree: 多项式特征的次数(仅用于部分模型)
        
    Returns:
        dict: 包含拟合模型和相关信息的字典
    """
    # 提取数据
    sufficiency = np.log10(results_dict['sufficiency'])  # 取对数
    sparsity = results_dict['sparsity']
    accuracy = results_dict['accuracy']
    
    # 准备特征矩阵
    X = np.column_stack([sufficiency, sparsity])
    y = accuracy
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 根据模型类型进行拟合
    r2_test = _fit_single_model(model_type, X_train, X_test, y_train, y_test)
    
    # 准备预测网格
    suff_range = np.linspace(sufficiency.min(), sufficiency.max(), 50)
    spar_range = np.linspace(sparsity.min(), sparsity.max(), 50)
    suff_grid, spar_grid = np.meshgrid(suff_range, spar_range)
    
    # 创建最终模型用于预测
    if model_type == 'svm':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_scaled, y)
        
        # 预测
        grid_points = np.column_stack([suff_grid.ravel(), spar_grid.ravel()])
        grid_points_scaled = scaler.transform(grid_points)
        predictions = model.predict(grid_points_scaled)
        
    elif model_type == 'gam_free':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearGAM(s(0, n_splines=8, lam=1) + s(1, n_splines=8, lam=1))
        model.fit(X_scaled, y)
        
        # 预测
        grid_points = np.column_stack([suff_grid.ravel(), spar_grid.ravel()])
        grid_points_scaled = scaler.transform(grid_points)
        predictions = model.predict(grid_points_scaled)
        
    else:  # linear, lightgbm等
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'lightgbm' and HAS_LIGHTGBM:
            model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, 
                                random_state=42, verbose=-1)
        else:
            model = LinearRegression()  # 默认使用线性回归
            
        model.fit(X, y)
        
        # 预测
        grid_points = np.column_stack([suff_grid.ravel(), spar_grid.ravel()])
        predictions = model.predict(grid_points)
    
    # 重塑预测结果
    predictions = predictions.reshape(suff_grid.shape)
    
    return {
        'model': model,
        'scaler': scaler if model_type in ['svm', 'gam_free'] else None,
        'predictions': predictions,
        'suff_grid': suff_grid,
        'spar_grid': spar_grid,
        'r2_test': r2_test,
        'model_type': model_type
    }

def fit_gam_models(df, sparsity_bins, use_seeds=True, sufficiency_bins=None, spline=14, lam=5, verbose=True, remove_outliers=True, gam_type='spline'):
    """
    重构的GAM模型拟合函数
    
    Args:
        df: 包含观测值和预测值的DataFrame
        sparsity_bins: sparsity分bin的边界
        use_seeds: 是否按seeds分组计算（True）还是直接计算全部（False）
        spline: GAM样条函数的节点数
        lam: GAM正则化参数
        verbose: 是否打印详细信息
        remove_outliers: 是否移除R²异常值以提高拟合质量
        gam_type: GAM函数类型 ('spline', 'linear', 'mixed', 'polynomial', 'monotonic_polynomial')
        
    Returns:
        dict: 拟合好的模型字典
    """
    # 1. 使用独立函数计算R²数据
    df_bins = calculate_r2_with_bins(
        df=df, 
        sparsity_bins=sparsity_bins,
        use_seeds=use_seeds,
        verbose=verbose,
        remove_outliers=remove_outliers,
        sufficiency_bins=sufficiency_bins
    )
    
    # 2. 为每个模型拟合GAM
    fitted_models = {}
    
    for model_name in df_bins['model'].unique():
        model_data = df_bins[df_bins['model'] == model_name].copy()
        
        # 特征工程
        model_data['sufficiency_log'] = np.log10(model_data['sufficiency'])
        X = model_data[['sufficiency_log', 'sparsity']].values
        y = model_data['r2'].values
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 特殊处理 monotonic_polynomial（移出外层try-except）
        if gam_type == 'monotonic_polynomial':
            try:
                # 单调 + 交互效应：单调样条 + 无约束交互项
                gam_model = LinearGAM(
                    s(0, constraints='monotonic_inc', n_splines=spline, lam=lam) + 
                    s(1, constraints='monotonic_inc', n_splines=spline, lam=lam) +
                    te(0, 1, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2)
                )
                
                # 拟合模型
                gam_model.fit(X_scaled, y)
                
                # 保存模型信息（标准格式）
                fitted_models[model_name] = {
                    'model': gam_model,
                    'scaler': scaler,
                    'r2_score': gam_model.statistics_['pseudo_r2']['explained_deviance'],
                    'gam_type': gam_type
                }
                
                if verbose:
                    print(f"Model {model_name} ({gam_type}): R² = {gam_model.statistics_['pseudo_r2']['explained_deviance']:.4f}, n_points = {len(model_data)}")
                    
            except Exception as poly_error:
                if verbose:
                    print(f"Model {model_name} ({gam_type}): Failed to fit - {str(poly_error)}")
            continue
        
        # 根据gam_type选择不同的GAM模型
        try:
            if gam_type == 'spline':
                # 单调样条函数 (默认)
                gam_model = LinearGAM(
                    s(0, constraints='monotonic_inc', n_splines=spline, lam=lam) + 
                    s(1, constraints='monotonic_inc', n_splines=spline, lam=lam)
                )
                
            elif gam_type == 'linear':
                # 纯线性项 (天然单调)
                gam_model = LinearGAM(
                    l(0, lam=lam) + l(1, lam=lam)
                )
                
            elif gam_type == 'mixed':
                # 混合：sufficiency用样条，sparsity用线性
                gam_model = LinearGAM(
                    s(0, constraints='monotonic_inc', n_splines=spline, lam=lam) + 
                    l(1, lam=lam)
                )
                
            elif gam_type == 'polynomial':
                # 多项式：使用tensor product来近似多项式交互效果，无约束
                gam_model = LinearGAM(
                    s(0, n_splines=spline, lam=lam) + 
                    s(1, n_splines=spline, lam=lam) + 
                    te(0, 1, n_splines=[spline//2, spline//2], lam=lam)
                )
                

            else:
                raise ValueError(f"Unknown gam_type: {gam_type}")
            
            # 所有模型都使用统一的拟合方式
            gam_model.fit(X_scaled, y)
            r2_val = gam_model.statistics_['pseudo_r2']['explained_deviance']
            
            # 保存模型信息
            fitted_models[model_name] = {
                'model': gam_model,
                'scaler': scaler,
                'r2_score': r2_val,
                'gam_type': gam_type
            }
            
            # 根据verbose参数决定是否打印信息
            if verbose:
                print(f"Model {model_name} ({gam_type}): R² = {r2_val:.4f}, n_points = {len(model_data)}")
                
        except Exception as e:
            if verbose:
                print(f"Model {model_name} ({gam_type}): Failed to fit - {str(e)}")
            continue
    
    return fitted_models

def fit_universal_models(df, sparsity_bins, model_type='gam_monotonic', use_seeds=True, sufficiency_bins=None, 
                        spline=7, lam=0.5, verbose=True, remove_outliers=True, full_features='Density', resolution=None, splitby='grid', outlier_threshold=-9999, diagnose=False, spline_order=2, diagnose_path=None):
    """
    通用模型拟合函数 - 支持所有_fit_single_model中的模型类型
    类似fit_gam_models，一次只拟合一个模型类型
    
    Args:
        df: 包含观测值和预测值的DataFrame
        sparsity_bins: sparsity分bin的边界
        model_type: 模型类型，支持:
                    'linear', 'ridge', 'elasticnet', 'rf', 'gbr', 'lightgbm', 'svm',
                    'svm_with_constraint' (SVM with light monotonic constraints on density/sufficiency),
                    'gam_free', 'gam_monotonic', 'gam_without_interaction', 'gam_mono_noint',
                    'interpolation' (IDW spatial interpolation, requires spatial features),
                    'two_stage' (GAM Density + SVM Spatial Residual, requires Full features),
                    'two_stage_1' (Only Stage 1: GAM Density prediction, requires Density features),
                    'two_stage_2' (Only Stage 2: SVM fitted residual prediction)
                    Note: 'two_stage_residual' not supported here (requires y_true)
        use_seeds: 是否按seeds分组计算（True）还是直接计算全部（False）
        sufficiency_bins: sufficiency分bin的数量
        spline: GAM样条函数的节点数
        lam: GAM正则化参数
        verbose: 是否打印详细信息
        remove_outliers: 是否移除R²异常值以提高拟合质量
        full_features: str or list
                      String options:
                        'Density': 使用[sufficiency_log, sparsity]
                        'Spatial': 使用[longitude, latitude] (需要df中有longitude, latitude列)
                        'Full': 使用全部4个特征[longitude, latitude, sufficiency_log, sparsity]
                      List: 直接指定特征列表，例如 ['longitude', 'latitude', 'sparsity']
                            会从Full模式的聚合数据中选择这些特征
        resolution: [lon_bins, lat_bins], 默认[10, 10] - 用于spatial aggregation
        splitby: 'grid' or 'station' - 聚合方式
        outlier_threshold: R² 异常值处理（默认 -9999，几乎不过滤）
                          - 数字（如 -0.5）：将 R² <= threshold 的值替换为 threshold
                          - 字符串（如 "0_remove", "-0.5_remove"）：移除 R² <= threshold 的数据点
                          - None：不进行任何处理
        diagnose: 是否在训练 two_stage 模型后绘制诊断图（默认 False）
                 仅对 two_stage/two_stage_1/two_stage_2 模型有效
        spline_order: GAM 样条阶数（默认 3 = 三次样条）
                     0=常数, 1=线性, 2=二次, 3=三次
                     降低可用更少样本，但曲线更"硬"
                     仅对 GAM 和 two_stage 系列模型有效
        diagnose_path: dict, 诊断图保存路径（仅对 two_stage 系列有效）
                      格式：{"model_name": ["stage1_path.svg", "stage2_path.png"]}
                      例如：{"mlp+biglightgbm": ["figures/stage1.svg", "figures/stage2.png"]}
                      只有在字典中指定的模型才会保存诊断图
                      根据文件扩展名自动识别格式（svg/png/pdf等）
                      None 则不保存
        
    Returns:
        dict: 拟合好的模型字典，格式为 {model_name: {...}}，与fit_gam_models相同
              每个模型包含: 'model', 'scaler', 'r2_score', 'model_type', 'full_features'
    
    Examples:
        >>> # 使用预设特征集
        >>> models = fit_universal_models(df, sparsity_bins=7, full_features='Full')
        
        >>> # 使用自定义特征列表
        >>> models = fit_universal_models(
        ...     df, sparsity_bins=7, 
        ...     full_features=['longitude', 'latitude', 'sparsity']
        ... )
        
        >>> # 只使用空间特征
        >>> models = fit_universal_models(
        ...     df, sparsity_bins=7,
        ...     full_features=['longitude', 'latitude']
        ... )
    """
    if resolution is None:
        resolution = [10, 10]
    
    # Check if full_features is a list (custom features)
    is_custom_features = isinstance(full_features, (list, tuple))
    
    # 1. 根据full_features准备数据
    if is_custom_features:
        # Custom feature list - use Full mode aggregation then select specified features
        if verbose:
            print(f"Using custom features: {full_features}")
            print(f"Aggregating by spatial grid with sampling features (resolution={resolution})...")
        
        # Validate that required features are available
        required_cols = set(full_features)
        spatial_cols = {'longitude', 'latitude'}
        needs_spatial = bool(spatial_cols & required_cols)
        
        if needs_spatial and ('longitude' not in df.columns or 'latitude' not in df.columns):
            raise ValueError("Custom features include longitude/latitude but these columns are not in df")
        
        model_columns = [col for col in df.columns if col.startswith('predicted_')]
        model_names = [col.replace('predicted_', '') for col in model_columns]
        
        bins_intervals = find_bins_intervals(df, sparsity_bins)
        aggregated_data = prepare_spatial_features(
            df, model_names, split_by=splitby,
            include_sampling_features=True,  # Need all features to select from
            bins_intervals=bins_intervals,
            resolution=resolution,
            outlier_threshold=outlier_threshold
        )
        
    elif full_features == 'Spatial':
        # 使用spatial features进行aggregation
        if 'longitude' not in df.columns or 'latitude' not in df.columns:
            raise ValueError("Spatial mode requires 'longitude' and 'latitude' columns in df")
        
        if verbose:
            print(f"Aggregating by spatial grid (resolution={resolution})...")
        
        # 获取模型列表
        model_columns = [col for col in df.columns if col.startswith('predicted_')]
        model_names = [col.replace('predicted_', '') for col in model_columns]
        
        # 使用prepare_spatial_features进行聚合
        bins_intervals = find_bins_intervals(df, sparsity_bins)
        aggregated_data = prepare_spatial_features(
            df, model_names, split_by='grid',
            include_sampling_features=False,  # Spatial模式不包含sampling features
            bins_intervals=None,
            resolution=resolution,
            outlier_threshold=outlier_threshold
        )
        
    elif full_features == 'Full':
        # 使用spatial + sampling features
        if 'longitude' not in df.columns or 'latitude' not in df.columns:
            raise ValueError("Full mode requires 'longitude' and 'latitude' columns in df")
        
        if verbose:
            print(f"Aggregating by spatial grid with sampling features (resolution={resolution})...")
        
        model_columns = [col for col in df.columns if col.startswith('predicted_')]
        model_names = [col.replace('predicted_', '') for col in model_columns]
        
        bins_intervals = find_bins_intervals(df, sparsity_bins)
        aggregated_data = prepare_spatial_features(
            df, model_names, split_by=splitby,
            include_sampling_features=True,  # Full模式包含所有features
            bins_intervals=bins_intervals,
            resolution=resolution,
            outlier_threshold=outlier_threshold
        )
        
    else:  # Density (default)
        # 使用原有的calculate_r2_with_bins方法
        if verbose:
            print(f"Aggregating by sufficiency×sparsity bins...")
        
        df_bins = calculate_r2_with_bins(
            df=df, 
            sparsity_bins=sparsity_bins,
            use_seeds=use_seeds,
            verbose=verbose,
            remove_outliers=remove_outliers,
            sufficiency_bins=sufficiency_bins
        )
        
        # 转换为aggregated_data格式
        aggregated_data = {}
        for model_name in df_bins['model'].unique():
            model_data = df_bins[df_bins['model'] == model_name].copy()
            model_data['sufficiency_log'] = np.log10(model_data['sufficiency'])
            aggregated_data[model_name] = model_data[['sufficiency_log', 'sparsity', 'r2']]
    
    # 2. 为每个模型拟合指定的model_type
    fitted_models = {}
    
    # 特殊处理：two_stage 系列需要原始数据
    # two_stage_residual 需要特殊处理（不能用于网格预测，只能用于评估）
    if model_type == 'two_stage_residual':
        raise ValueError("two_stage_residual cannot be used in fit_universal_models (requires y_true). "
                        "Use two_stage_2 to predict fitted SVM residuals, or use eval_baseline_comparison for residual analysis.")
    
    if model_type in ['two_stage', 'two_stage_1', 'two_stage_2']:
        if model_type in ['two_stage', 'two_stage_2']:
            # two_stage 和 two_stage_2 需要全部 4 个特征
            if full_features != 'Full' and not (is_custom_features and 
                                               set(full_features) >= {'longitude', 'latitude', 'sufficiency_log', 'sparsity'}):
                raise ValueError(f"{model_type} model requires full_features='Full' or custom features with all 4 columns")
        # two_stage_1 可以使用 Density 特征或更多
        
        # 获取模型列表
        model_columns = [col for col in df.columns if col.startswith('predicted_')]
        model_names = [col.replace('predicted_', '') for col in model_columns]
        
        # 准备 bins_intervals
        bins_intervals = find_bins_intervals(df, sparsity_bins)
        
        # 为每个模型训练 two_stage
        for model_name in model_names:
            # 从 diagnose_path 字典中获取当前模型的路径（如果指定）
            model_diagnose_path = diagnose_path.get(model_name) if diagnose_path else None
            
            two_stage_model = TwoStageModel(
                spline=spline, lam=lam, resolution=resolution, 
                diagnose=diagnose, spline_order=spline_order,
                diagnose_path=model_diagnose_path
            )
            stage1_score, stage2_score = two_stage_model.fit(
                df, model_name, bins_intervals,
                use_seeds=use_seeds, split_by=splitby, metric='r2'
            )
            
            # 根据 model_type 选择显示的得分
            if model_type == 'two_stage_1':
                display_score = stage1_score
            elif model_type == 'two_stage_2':
                display_score = stage2_score  # Stage2 本身不好评估，这里保持 stage2
            else:  # two_stage
                display_score = stage2_score
            
            fitted_models[model_name] = {
                'model': two_stage_model,
                'scaler': None,
                'r2_score': display_score,
                'model_type': model_type,
                'full_features': full_features,
                'stage1_score': stage1_score,
                'stage2_score': stage2_score
            }
            
            if verbose:
                if model_type == 'two_stage_1':
                    print(f"  {model_name} [two_stage_1]: Stage1={stage1_score:.4f} (only), n_points = {len(df)}")
                elif model_type == 'two_stage_2':
                    print(f"  {model_name} [two_stage_2]: Stage2 residual (Stage1={stage1_score:.4f}, Stage2={stage2_score:.4f}), n_points = {len(df)}")
                else:
                    print(f"  {model_name} [two_stage]: Stage1={stage1_score:.4f}, Stage2={stage2_score:.4f}, n_points = {len(df)}")
        
        return fitted_models
    
    # 普通模型训练流程
    for model_name in aggregated_data.keys():
        model_data = aggregated_data[model_name].copy()
        
        if len(model_data) < 10:
            if verbose:
                print(f"Skipping {model_name}: insufficient data (n={len(model_data)})")
            continue
        
        # 准备特征
        if is_custom_features:
            # Custom feature list
            try:
                X = model_data[list(full_features)].values
            except KeyError as e:
                if verbose:
                    print(f"  {model_name}: Missing feature {e}, skipping")
                continue
        elif full_features == 'Full':
            X = model_data[['longitude', 'latitude', 'sufficiency_log', 'sparsity']].values
        elif full_features == 'Spatial':
            X = model_data[['longitude', 'latitude']].values
        else:  # Density
            X = model_data[['sufficiency_log', 'sparsity']].values
        
        y = model_data['r2'].values
        
        # 特殊处理：interpolation 需要 spatial features
        if model_type == 'interpolation':
            if full_features == 'Spatial' or (is_custom_features and set(full_features) >= {'longitude', 'latitude'}):
                # 确保使用 lon/lat
                X = model_data[['longitude', 'latitude']].values
            else:
                if verbose:
                    print(f"  {model_name} [interpolation]: Requires spatial features (longitude, latitude), skipping")
                continue
        
        try:
            # 使用全部数据训练（没有train/test split，类似fit_gam_models）
            score, model_obj, scaler = _fit_single_model(
                model_type, X, X, y, y, 
                metric='r2', spline=spline, lam=lam, return_model=True
            )
            
            if score == -999:
                if verbose:
                    print(f"  {model_name} [{model_type}]: Failed to fit")
                continue
            
            # 保存模型信息（与fit_gam_models格式相同）
            fitted_models[model_name] = {
                'model': model_obj,
                'scaler': scaler,
                'r2_score': score,
                'model_type': model_type,
                'full_features': full_features
            }
            
            if verbose:
                print(f"  {model_name} [{model_type}]: R² = {score:.4f}, n_points = {len(model_data)}")
                
        except Exception as e:
            if verbose:
                print(f"  {model_name} [{model_type}]: Failed - {str(e)}")
            continue
    
    return fitted_models


class IDWInterpolator:
    """IDW (Inverse Distance Weighting) 插值器，提供统一的 .predict() 接口"""
    
    def __init__(self, X_train, y_train):
        """
        Args:
            X_train: 训练数据坐标 (n_samples, 2) - [longitude, latitude]
            y_train: 训练数据值 (n_samples,)
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        """
        使用 IDW 方法预测
        
        Args:
            X_test: 测试数据坐标 (n_samples, 2)
            
        Returns:
            y_pred: 预测值 (n_samples,)
        """
        y_pred = []
        for x_test_point in X_test:
            distances = np.sqrt(np.sum((self.X_train - x_test_point)**2, axis=1))
            
            # 避免除零：如果距离为0，直接使用该点的值
            if np.any(distances == 0):
                y_pred.append(self.y_train[distances == 0][0])
            else:
                # IDW插值：权重 = 1 / distance^2
                weights = 1.0 / (distances ** 2)
                weights = weights / np.sum(weights)  # 归一化
                y_pred.append(np.sum(weights * self.y_train))
        
        return np.array(y_pred)


class TwoStageModel:
    """
    双阶段模型：GAM (Density) + SVM (Spatial Residual)
    
    Stage 1: 使用 sampling aggregation + GAM_Monotonic 预测基础 R² (基于 density 特征)
            🔍 自动检测 sufficiency 数量：
            - 单一 sufficiency → 使用 sparsity 单变量单调 GAM
            - 多个 sufficiency → 使用 (sufficiency_log, sparsity) 双变量 GAM + 交互项
    
    Stage 2: 使用 spatial aggregation + SVM 预测残差 (基于 spatial 特征)
    
    Final: 预测值 = GAM 预测 + SVM 残差预测
    """
    
    def __init__(self, spline=7, lam=0.5, resolution=None, stage2_features=None, diagnose=False, spline_order=2, svm_only=0, clip=None, diagnose_path=None):
        """
        Args:
            spline: GAM 样条函数节点数
            lam: GAM 正则化参数
            resolution: spatial aggregation 分辨率 [lon_bins, lat_bins]
            stage2_features: 第二阶段 SVM 使用的特征列表
                           默认 ['longitude', 'latitude']
                           可选添加 'sufficiency_log' 和/或 'sparsity'
                           例如：['longitude', 'latitude', 'sufficiency_log', 'sparsity']
            diagnose: 是否在训练后绘制诊断图（默认 False）
            spline_order: 样条阶数（0=常数, 1=线性, 2=二次, 3=三次[默认]）
                         降低 spline_order 可以用更少的样本，但曲线会更"硬"
            svm_only: 是否只使用 SVM（跳过 GAM Stage1），默认 False
            clip: R² 截断范围 [min, max]，默认 [-0.5, 1.0]
                 在数据准备阶段截断异常 R² 值，防止极端值影响训练
                 设为 None 则不截断
            diagnose_path: 诊断图保存路径 [stage1_path, stage2_path]
                          例如：["figures/stage1.svg", "figures/stage2.png"]
                          根据文件扩展名自动识别格式，None 则不保存
        """
        self.spline = spline
        self.lam = lam
        self.resolution = resolution if resolution else [10, 10]
        self.stage2_features = stage2_features if stage2_features else ['longitude', 'latitude', 'sufficiency_log']# ['longitude', 'latitude',]#['longitude', 'latitude']   'sparsity'
        self.diagnose = diagnose
        self.spline_order = spline_order
        self.svm_only = svm_only
        self.clip = clip if clip is not None else [-0.5, 1.0]  # 默认截断范围
        self.diagnose_path = diagnose_path  # 诊断图保存路径
        
        # 第一阶段：GAM 模型 (Density features)
        self.gam_model = None
        self.gam_scaler = None
        
        # 第二阶段：SVM 模型 (Spatial features for residual)
        self.svm_model = None
        self.svm_scaler = None
        
        # 自动检测标志：是否只有单一 sufficiency 值
        self.single_sufficiency = False
        
        # 诊断数据缓存 - Stage 1 (GAM)
        self._stage1_X_raw = None
        self._stage1_X_scaled = None
        self._stage1_y_true = None
        self._stage1_y_pred = None
        self._actual_spline = None  # 实际使用的 spline 数量（可能自动调整）
        
        # 诊断数据缓存 - Stage 2 (SVM)
        self._stage2_data = None  # 完整的 stage2 训练数据 DataFrame
    
    def fit(self, df_train_raw, model_name, bins_intervals, use_seeds=True, split_by='grid', metric='r2'):
        """
        训练双阶段模型
        
        Args:
            df_train_raw: 原始训练数据
            model_name: 要预测的模型名称（例如 'Ours'）
            bins_intervals: sparsity bins 区间
            use_seeds: 是否使用 seeds 聚合
            split_by: spatial 聚合方式 ('grid' or 'station')
            metric: 评估指标 ('r2' or 'correlation')
        
        Returns:
            stage1_score: 第一阶段 GAM 的得分
            stage2_score: 第二阶段（GAM + SVM）的得分
        """
        from sklearn.metrics import r2_score
        from scipy.stats import pearsonr
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from pygam import LinearGAM, s, te
        
        # ====== Stage 1: 使用 sampling aggregation 训练 GAM ======
        # stage1_data_dict = prepare_sampling_features(
        #     bins_intervals, df_train_raw, [model_name], use_seeds,
        #     include_spatial_coords=True
        # )

        stage1_data_dict = prepare_sampling_features_space(
                bins_intervals, df_train_raw, [model_name], 
                split_by=split_by,
                resolution=self.resolution,
                use_seeds=use_seeds,
                clip=self.clip  # 传递截断参数
            )

        stage1_data = stage1_data_dict.get(model_name)
        
        if stage1_data is None or len(stage1_data) < 5:
            raise ValueError(f"Insufficient data for model {model_name} in stage 1")
        
        # 🔍 自动检测：sufficiency 是否只有唯一值
        # 使用容差判断，避免浮点数精度问题导致的假阳性
        suff_log_values = stage1_data['sufficiency_log'].values
        suff_log_std = np.std(suff_log_values)
        self.single_sufficiency = (suff_log_std < 1e-6)  # 标准差 < 1e-6 视为单一值
        self.stage2_features=['longitude', 'latitude']
        
        # 提取 density 特征（根据 sufficiency 数量选择）
        if self.single_sufficiency:
            # 只有单一 sufficiency，只用 sparsity
            X_stage1 = stage1_data[['sparsity']].values
            # print(f"  ⚠️  Detected single sufficiency value, using sparsity-only monotonic GAM")
        else:
            # 多个 sufficiency，使用双变量 GAM
            X_stage1 = stage1_data[['sufficiency_log', 'sparsity']].values
        
        y_stage1 = stage1_data['r2'].values
        
        # 🔍 自动调整 spline 数量：满足 PyGAM 要求 n_splines > spline_order
        n_samples = len(y_stage1)
        min_splines = self.spline_order + 1
        adjusted_spline = self.spline
        self._actual_spline = adjusted_spline
        # 训练 GAM with monotonic constraints
        self.gam_scaler = StandardScaler()
        X_stage1_scaled = self.gam_scaler.fit_transform(X_stage1)
        
        if self.single_sufficiency:
            # 单变量 GAM：只用 sparsity (monotonic increasing)
            self.gam_model = LinearGAM(
                s(0, constraints='monotonic_inc', n_splines=adjusted_spline, 
                  spline_order=self.spline_order, lam=self.lam)
            )
        else:
            # 双变量 GAM：sufficiency_log + sparsity + interaction
            interaction_spline = max(min_splines, adjusted_spline // 4)  # 恢复原始逻辑
            self.gam_model = LinearGAM(
                s(0, constraints='monotonic_inc', n_splines=adjusted_spline, 
                  spline_order=self.spline_order, lam=self.lam) + 
                s(1, constraints='monotonic_inc', n_splines=adjusted_spline, 
                  spline_order=self.spline_order, lam=self.lam) +
                te(0, 1, n_splines=[interaction_spline, interaction_spline],spline_order=self.spline_order,  lam=self.lam*2)
            )
        
        self.gam_model.fit(X_stage1_scaled, y_stage1)
        
        # 评估 Stage 1
        y_gam_pred_train = self.gam_model.predict(X_stage1_scaled)
        if metric == 'correlation':
            stage1_score, _ = pearsonr(y_stage1, y_gam_pred_train)
        else:
            stage1_score = r2_score(y_stage1, y_gam_pred_train)
        
        # 🔍 保存诊断数据并可选绘图
        self._stage1_X_raw = X_stage1  # 原始特征（未标准化）
        self._stage1_X_scaled = X_stage1_scaled  # 标准化后的特征
        self._stage1_y_true = y_stage1
        self._stage1_y_pred = y_gam_pred_train
        
        if self.diagnose:
            stage1_save_path = self.diagnose_path[0] if self.diagnose_path and len(self.diagnose_path) > 0 else None
            self.plot_stage1_diagnosis(save_path=stage1_save_path)
        
        # ====== Stage 2: 使用 spatial aggregation 训练 SVM（预测残差）======
        stage2_data_dict = prepare_spatial_features(
            df_train_raw, [model_name],
            split_by=split_by,
            include_sampling_features=True,  # 需要 density 特征用于 GAM 预测
            bins_intervals=bins_intervals,
            resolution=self.resolution,
            clip=self.clip  # 传递截断参数
        )
        stage2_data = stage2_data_dict.get(model_name)
        
        if stage2_data is None or len(stage2_data) < 5:
            raise ValueError(f"Insufficient spatial data for model {model_name} in stage 2")
        
        # 计算残差
        y_true = stage2_data['r2'].values
        
        if self.svm_only:
            # SVM-only 模式：直接拟合 R²，不用 GAM
            residuals = y_true
            y_gam_pred = 0  # 占位，不使用
        else:
            # 正常模式：使用 GAM 预测 stage2 数据（根据 sufficiency 数量选择特征）
            if self.single_sufficiency:
                X_stage2_density = stage2_data[['sparsity']].values
            else:
                X_stage2_density = stage2_data[['sufficiency_log', 'sparsity']].values
            
            X_stage2_density_scaled = self.gam_scaler.transform(X_stage2_density)
            y_gam_pred = self.gam_model.predict(X_stage2_density_scaled)
            residuals = y_true - y_gam_pred
        
        # 训练 SVM 预测残差（使用可配置的特征）
        X_stage2_spatial = stage2_data[self.stage2_features].values
        
        self.svm_scaler = StandardScaler()
        X_stage2_spatial_scaled = self.svm_scaler.fit_transform(X_stage2_spatial)
        
        self.svm_model = SVR(kernel='rbf', C=1, gamma='scale')
        self.svm_model.fit(X_stage2_spatial_scaled, residuals)
        
        # 评估 Stage 2 (GAM + SVM)
        residual_pred = self.svm_model.predict(X_stage2_spatial_scaled)
        y_final_pred = y_gam_pred + residual_pred
        
        if metric == 'correlation':
            stage2_score, _ = pearsonr(y_true, y_final_pred)
        else:
            stage2_score = r2_score(y_true, y_final_pred)
        
        # 🔍 保存 Stage 2 诊断数据
        stage2_data_diag = stage2_data.copy()
        stage2_data_diag['r2_gam_pred'] = y_gam_pred if not self.svm_only else 0
        stage2_data_diag['residual_true'] = residuals
        stage2_data_diag['residual_pred'] = residual_pred
        self._stage2_data = stage2_data_diag
        
        # 🔍 自动诊断 Stage 2（如果启用）
        if self.diagnose:
            stage2_save_path = self.diagnose_path[1] if self.diagnose_path and len(self.diagnose_path) > 1 else None
            self.plot_stage2_diagnosis(save_path=stage2_save_path)
        
        return stage1_score, stage2_score
    
    def plot_stage1_diagnosis(self, figsize=None, save_path=None):
        """
        诊断 Stage 1 GAM 拟合效果
        
        单 sufficiency: 1×3 图（sparsity 的 3 个视角）
        多 sufficiency: 2×3 图
          - 第一行：sufficiency_log 的 3 个视角
          - 第二行：sparsity 的 3 个视角
        
        Args:
            figsize: 图片大小，默认自动（单变量 14×5，双变量 14×10）
            save_path: 保存路径，根据扩展名自动识别格式（svg/png），None 则不保存
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if self._stage1_X_raw is None or self._stage1_y_true is None:
            print("⚠️  No diagnosis data available. Run fit() first.")
            return
        
        X_raw = self._stage1_X_raw
        y_true = self._stage1_y_true
        y_pred = self._stage1_y_pred
        
        # 根据模式确定布局
        if self.single_sufficiency:
            # 单变量：1×3 图
            nrows, ncols = 1, 3
            figsize = figsize or (14, 5)
        else:
            # 双变量：2×3 图
            nrows, ncols = 2, 3
            figsize = figsize or (14, 10)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # 确保 axes 是二维数组
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        # 绘制函数：为每个变量画 3 个子图
        def plot_variable_row(ax_row, X_var, var_name, row_idx):
            """画一行（3个子图）：X vs y_true, X vs y_pred+curve, y_true vs y_pred"""
            
            # 子图 1: X vs y_true
            ax_row[0].scatter(X_var, y_true, alpha=0.6, s=30, c='steelblue', edgecolors='k', linewidths=0.5)
            ax_row[0].set_xlabel(var_name, fontsize=12)
            ax_row[0].set_ylabel('R² (True)', fontsize=12)
            ax_row[0].set_title(f'{var_name}: Training Data', fontsize=12, fontweight='bold')
            ax_row[0].grid(True, alpha=0.3)
            
            # 子图 2: X vs y_pred + GAM 拟合曲线
            ax_row[1].scatter(X_var, y_pred, alpha=0.6, s=30, c='coral', edgecolors='k', linewidths=0.5, label='GAM predictions')
            
            # 生成拟合曲线
            X_range = np.linspace(X_var.min(), X_var.max(), 200)
            
            if self.single_sufficiency:
                # 单变量：直接预测
                X_range_input = X_range.reshape(-1, 1)
                X_range_scaled = self.gam_scaler.transform(X_range_input)
                y_curve = self.gam_model.predict(X_range_scaled)
            else:
                # 双变量：计算边际平均效应（对另一个变量的所有值取平均）
                if row_idx == 0:
                    # 第一行（sufficiency_log）：对所有 sparsity 值取平均
                    other_var_values = X_raw[:, 1]  # 所有 sparsity 值
                    y_curve_list = []
                    for x_val in X_range:
                        # 对每个 sufficiency_log 值，遍历所有 sparsity
                        X_grid = np.column_stack([
                            np.full(len(other_var_values), x_val),  # 固定 sufficiency_log
                            other_var_values  # 所有 sparsity
                        ])
                        X_grid_scaled = self.gam_scaler.transform(X_grid)
                        y_grid = self.gam_model.predict(X_grid_scaled)
                        y_curve_list.append(y_grid.mean())  # 平均
                    y_curve = np.array(y_curve_list)
                else:
                    # 第二行（sparsity）：对所有 sufficiency_log 值取平均
                    other_var_values = X_raw[:, 0]  # 所有 sufficiency_log 值
                    y_curve_list = []
                    for x_val in X_range:
                        # 对每个 sparsity 值，遍历所有 sufficiency_log
                        X_grid = np.column_stack([
                            other_var_values,  # 所有 sufficiency_log
                            np.full(len(other_var_values), x_val)  # 固定 sparsity
                        ])
                        X_grid_scaled = self.gam_scaler.transform(X_grid)
                        y_grid = self.gam_model.predict(X_grid_scaled)
                        y_curve_list.append(y_grid.mean())  # 平均
                    y_curve = np.array(y_curve_list)
            ax_row[1].plot(X_range, y_curve, 'b-', linewidth=2.5, label='GAM curve', alpha=0.8)
            
            ax_row[1].set_xlabel(var_name, fontsize=12)
            ax_row[1].set_ylabel('R² (GAM Predicted)', fontsize=12)
            ax_row[1].set_title(f'{var_name}: GAM Predictions + Curve', fontsize=12, fontweight='bold')
            ax_row[1].legend(loc='best', fontsize=9)
            ax_row[1].grid(True, alpha=0.3)
            
            # 子图 3: y_true vs y_pred（所有行共享）
            ax_row[2].scatter(y_true, y_pred, alpha=0.6, s=30, c='green', edgecolors='k', linewidths=0.5)
            ax_row[2].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                         'r--', linewidth=2, label='Perfect fit')
            ax_row[2].set_xlabel('R² (True)', fontsize=12)
            ax_row[2].set_ylabel('R² (GAM Predicted)', fontsize=12)
            ax_row[2].set_title('GAM Fit Quality', fontsize=12, fontweight='bold')
            ax_row[2].legend(fontsize=9)
            ax_row[2].grid(True, alpha=0.3)
        
        # 根据模式画图
        if self.single_sufficiency:
            # 单变量：只画 sparsity
            plot_variable_row(axes[0], X_raw[:, 0], 'Sparsity', 0)
        else:
            # 双变量：第一行 sufficiency_log，第二行 sparsity
            plot_variable_row(axes[0], X_raw[:, 0], 'Sufficiency_log', 0)
            plot_variable_row(axes[1], X_raw[:, 1], 'Sparsity', 1)
        
        # 计算统计信息
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # 添加标注
        mode_text = "Single Sufficiency (Sparsity-only GAM)" if self.single_sufficiency else "Multiple Sufficiency (Full GAM)"
        fig.suptitle(f'Stage 1 GAM Diagnosis | Mode: {mode_text} | R²={r2:.4f}, MAE={mae:.4f}', 
                     fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 保存图片（如果指定了路径）
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            file_format = os.path.splitext(save_path)[1][1:]  # 获取扩展名（不含点）
            plt.savefig(save_path, format=file_format, dpi=300, bbox_inches='tight')
            print(f"✓ Stage 1 diagram saved to: {save_path}")
        
        plt.show()
        
        # 打印详细统计
        print(f"\n{'='*70}")
        print(f"Stage 1 GAM Training Diagnosis Report")
        print(f"{'='*70}")
        print(f"Mode: {mode_text}")
        print(f"Training samples: {len(y_true)}")
        if self._actual_spline is not None:
            spline_text = f"{self._actual_spline}"
            if self._actual_spline < self.spline:
                spline_text += f" (auto-adjusted from {self.spline})"
            print(f"Splines used: {spline_text}")
        print(f"")
        print(f"Training Feature Range (Raw):")
        for i in range(X_raw.shape[1]):
            feat_name = 'sparsity' if self.single_sufficiency else ('sufficiency_log' if i == 0 else 'sparsity')
            print(f"  {feat_name:20s}: [{X_raw[:, i].min():.6e}, {X_raw[:, i].max():.6e}]")
        print(f"")
        print(f"Model Performance:")
        print(f"  R² score: {r2:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Prediction range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
        print(f"  True R² range: [{y_true.min():.6f}, {y_true.max():.6f}]")
        
        if abs(y_pred.max() - y_pred.min()) < 1e-6:
            print(f"\n⚠️  WARNING: GAM predictions are nearly constant!")
            print(f"   This suggests the model failed to learn from the data.")
            print(f"   Possible causes:")
            print(f"   - Too few training samples ({len(y_true)} points)")
            print(f"   - Feature range too small")
            print(f"   - StandardScaler issue")
            print(f"   - GAM hyperparameters (lam={self.lam})")
        
        print(f"{'='*70}")
        print(f"💡 Use model.diagnose_prediction(X_test) to check prediction issues")
        print(f"{'='*70}\n")
    
    def diagnose_prediction(self, X_test):
        """
        诊断预测时的特征处理
        
        检查：
        1. 输入特征的范围
        2. 标准化后的特征范围
        3. 与训练数据的对比
        
        Args:
            X_test: 测试数据 [longitude, latitude, sufficiency_log, sparsity]
        """
        import numpy as np
        
        if self.gam_model is None:
            print("⚠️  Model not fitted yet.")
            return
        
        print(f"\n{'='*70}")
        print(f"Stage 1 GAM Prediction Diagnosis")
        print(f"{'='*70}")
        
        # 提取特征
        feature_map = {
            'sufficiency_log': 2,
            'sparsity': 3
        }
        
        if self.single_sufficiency:
            X_density = X_test[:, feature_map['sparsity']].reshape(-1, 1)
            feature_names = ['sparsity']
        else:
            X_density = np.column_stack([
                X_test[:, feature_map['sufficiency_log']],
                X_test[:, feature_map['sparsity']]
            ])
            feature_names = ['sufficiency_log', 'sparsity']
        
        print(f"\n1️⃣  Mode: {'Single Sufficiency (sparsity-only)' if self.single_sufficiency else 'Multiple Sufficiency'}")
        print(f"   Features used: {feature_names}")
        print(f"   Test samples: {len(X_test)}")
        
        # 显示原始特征范围
        print(f"\n2️⃣  Test Data (Raw Features):")
        for i, name in enumerate(feature_names):
            print(f"   {name:20s}: [{X_density[:, i].min():.6e}, {X_density[:, i].max():.6e}]")
            print(f"   {'':20s}  Mean={X_density[:, i].mean():.6e}, Std={X_density[:, i].std():.6e}")
        
        # 标准化
        X_density_scaled = self.gam_scaler.transform(X_density)
        
        print(f"\n3️⃣  Test Data (After Scaling):")
        for i, name in enumerate(feature_names):
            print(f"   {name:20s}: [{X_density_scaled[:, i].min():.6e}, {X_density_scaled[:, i].max():.6e}]")
            print(f"   {'':20s}  Mean={X_density_scaled[:, i].mean():.6e}, Std={X_density_scaled[:, i].std():.6e}")
        
        # 与训练数据对比
        if self._stage1_X_raw is not None:
            print(f"\n4️⃣  Training Data Reference (Raw Features):")
            for i, name in enumerate(feature_names):
                print(f"   {name:20s}: [{self._stage1_X_raw[:, i].min():.6e}, {self._stage1_X_raw[:, i].max():.6e}]")
                print(f"   {'':20s}  Mean={self._stage1_X_raw[:, i].mean():.6e}, Std={self._stage1_X_raw[:, i].std():.6e}")
            
            print(f"\n5️⃣  Training Data Reference (After Scaling):")
            for i, name in enumerate(feature_names):
                print(f"   {name:20s}: [{self._stage1_X_scaled[:, i].min():.6e}, {self._stage1_X_scaled[:, i].max():.6e}]")
                print(f"   {'':20s}  Mean={self._stage1_X_scaled[:, i].mean():.6e}, Std={self._stage1_X_scaled[:, i].std():.6e}")
        
        # 预测
        y_pred = self.gam_model.predict(X_density_scaled)
        
        print(f"\n6️⃣  GAM Predictions:")
        print(f"   Range: [{y_pred.min():.6e}, {y_pred.max():.6e}]")
        print(f"   Mean={y_pred.mean():.6e}, Std={y_pred.std():.6e}")
        
        if abs(y_pred.max() - y_pred.min()) < 1e-6:
            print(f"\n   ⚠️  WARNING: Predictions are nearly constant!")
        
        # 检查 Scaler 参数
        print(f"\n7️⃣  StandardScaler Parameters:")
        print(f"   Mean (used for scaling): {self.gam_scaler.mean_}")
        print(f"   Std  (used for scaling): {self.gam_scaler.scale_}")
        
        print(f"{'='*70}\n")
        
        return X_density, X_density_scaled, y_pred
    
    def plot_stage2_diagnosis(self, figsize=None, save_path=None):
        """
        诊断 Stage 2 SVM 空间残差拟合效果
        
        按不同 sufficiency 分行，每行 4 列：
        - 第1列：原始 R² 空间分布（训练数据）
        - 第2列：残差空间分布（真实残差，待预测）
        - 第3列：SVM 预测的残差空间分布
        - 第4列：真实残差 vs 预测残差散点图
        
        Args:
            figsize: 图片大小，默认自动（每行高度 4）
            save_path: 保存路径，根据扩展名自动识别格式（svg/png），None 则不保存
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score, mean_absolute_error
        
        if self._stage2_data is None:
            print("⚠️  No Stage 2 diagnosis data available. Run fit() first.")
            return
        
        df = self._stage2_data
        
        # 获取唯一的 sufficiency_bin 值（如果有）
        if 'sufficiency_bin' in df.columns:
            unique_bins = sorted(df['sufficiency_bin'].unique())
            n_rows = len(unique_bins)
        else:
            # 如果没有 sufficiency_bin，用单行
            unique_bins = [None]
            n_rows = 1
        
        # 设置图片大小
        if figsize is None:
            figsize = (16, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, 4, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)  # 确保 axes 是 2D
        
        for row_idx, suff_bin in enumerate(unique_bins):
            # 筛选当前 sufficiency_bin 的数据
            if suff_bin is not None:
                df_subset = df[df['sufficiency_bin'] == suff_bin]
                suff_mean = 10 ** df_subset['sufficiency_log'].mean()
                title_suffix = f"Sufficiency ≈ {suff_mean:.0f}"
            else:
                df_subset = df
                title_suffix = "All Data"
            
            lon = df_subset['longitude'].values
            lat = df_subset['latitude'].values
            r2_true = df_subset['r2'].values
            residual_true = df_subset['residual_true'].values
            residual_pred = df_subset['residual_pred'].values
            
            # 计算残差预测性能
            residual_r2 = r2_score(residual_true, residual_pred)
            residual_mae = mean_absolute_error(residual_true, residual_pred)
            
            # 计算残差的统一颜色范围（第2、3列使用相同范围以便对比）
            residual_min = min(residual_true.min(), residual_pred.min())
            residual_max = max(residual_true.max(), residual_pred.max())
            # 对称化范围（使0在中心）
            residual_abs_max = max(abs(residual_min), abs(residual_max))
            residual_vmin = -residual_abs_max
            residual_vmax = residual_abs_max
            
            # 第1列：原始 R² 空间分布
            ax1 = axes[row_idx, 0]
            scatter1 = ax1.scatter(lon, lat, c=r2_true, cmap='viridis', s=40, alpha=0.7, edgecolors='k', linewidths=0.5)
            ax1.set_xlabel('Longitude', fontsize=10)
            ax1.set_ylabel('Latitude', fontsize=10)
            ax1.set_title(f'Original R²\n{title_suffix}', fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('R² (True)', fontsize=9)
            ax1.text(0.02, 0.98, f'n={len(df_subset)}', transform=ax1.transAxes,
                    verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 第2列：真实残差空间分布
            ax2 = axes[row_idx, 1]
            scatter2 = ax2.scatter(lon, lat, c=residual_true, cmap='coolwarm', s=40, alpha=0.7, edgecolors='k', linewidths=0.5, vmin=residual_vmin, vmax=residual_vmax)
            ax2.set_xlabel('Longitude', fontsize=10)
            ax2.set_ylabel('Latitude', fontsize=10)
            ax2.set_title(f'Residual (True)\nR² - GAM pred', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('Residual', fontsize=9)
            ax2.text(0.02, 0.98, f'Range: [{residual_true.min():.3f}, {residual_true.max():.3f}]', 
                    transform=ax2.transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            # 第3列：SVM 预测的残差空间分布
            ax3 = axes[row_idx, 2]
            scatter3 = ax3.scatter(lon, lat, c=residual_pred, cmap='coolwarm', s=40, alpha=0.7, edgecolors='k', linewidths=0.5, vmin=residual_vmin, vmax=residual_vmax)
            ax3.set_xlabel('Longitude', fontsize=10)
            ax3.set_ylabel('Latitude', fontsize=10)
            ax3.set_title(f'Residual (SVM pred)\nR²={residual_r2:.3f}', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            cbar3 = plt.colorbar(scatter3, ax=ax3)
            cbar3.set_label('Residual (pred)', fontsize=9)
            ax3.text(0.02, 0.98, f'MAE={residual_mae:.3f}', 
                    transform=ax3.transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            # 第4列：真实 vs 预测散点图
            ax4 = axes[row_idx, 3]
            ax4.scatter(residual_true, residual_pred, s=30, alpha=0.6, c='steelblue', edgecolors='k', linewidths=0.5)
            
            # 添加 y=x 参考线
            lim_min = min(residual_true.min(), residual_pred.min())
            lim_max = max(residual_true.max(), residual_pred.max())
            ax4.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect fit', alpha=0.7)
            
            ax4.set_xlabel('Residual (True)', fontsize=10)
            ax4.set_ylabel('Residual (SVM pred)', fontsize=10)
            ax4.set_title(f'True vs Predicted\nR²={residual_r2:.3f}, MAE={residual_mae:.3f}', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='best', fontsize=8)
            ax4.set_aspect('equal', adjustable='box')
        
        # 总标题
        mode_text = "Single Sufficiency" if self.single_sufficiency else "Multiple Sufficiency"
        svm_features = ', '.join(self.stage2_features)
        fig.suptitle(f'Stage 2 SVM Diagnosis | Mode: {mode_text} | Features: [{svm_features}]', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # 保存图片（如果指定了路径）
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            file_format = os.path.splitext(save_path)[1][1:]  # 获取扩展名（不含点）
            plt.savefig(save_path, format=file_format, dpi=300, bbox_inches='tight')
            print(f"✓ Stage 2 diagram saved to: {save_path}")
        
        plt.show()
        
        # 打印详细统计
        print(f"\n{'='*70}")
        print(f"Stage 2 SVM Diagnosis Report")
        print(f"{'='*70}")
        print(f"Mode: {mode_text}")
        print(f"Training samples: {len(df)}")
        print(f"SVM features: {svm_features}")
        print(f"Resolution: {self.resolution}")
        print(f"")
        print(f"Overall Performance:")
        overall_r2 = r2_score(df['residual_true'], df['residual_pred'])
        overall_mae = mean_absolute_error(df['residual_true'], df['residual_pred'])
        print(f"  Residual prediction R²: {overall_r2:.6f}")
        print(f"  Residual prediction MAE: {overall_mae:.6f}")
        print(f"  Residual range: [{df['residual_true'].min():.3f}, {df['residual_true'].max():.3f}]")
        print(f"  Original R² range: [{df['r2'].min():.3f}, {df['r2'].max():.3f}]")
        print(f"{'='*70}\n")
    
    def predict(self, X_test):
        """
        预测（完整双阶段）
        
        Args:
            X_test: 测试数据，必须包含 4 列: [longitude, latitude, sufficiency_log, sparsity]
            
        Returns:
            y_pred: 预测值 (GAM 预测 + SVM 残差预测)
        """
        if self.gam_model is None or self.svm_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # 假设输入顺序：[longitude, latitude, sufficiency_log, sparsity]
        feature_map = {
            'longitude': 0,
            'latitude': 1,
            'sufficiency_log': 2,
            'sparsity': 3
        }
        
        # Stage 2: SVM 预测残差（使用可配置的特征）
        X_spatial = np.column_stack([X_test[:, feature_map[f]] for f in self.stage2_features])
        X_spatial_scaled = self.svm_scaler.transform(X_spatial)
        residual_pred = self.svm_model.predict(X_spatial_scaled)
        
        if self.svm_only:
            # SVM-only 模式：直接返回 SVM 预测
            return residual_pred
        else:
            # 正常模式：Stage 1 GAM 预测（根据 sufficiency 数量选择特征）
            if self.single_sufficiency:
                # 只用 sparsity
                X_density = X_test[:, feature_map['sparsity']].reshape(-1, 1)
            else:
                # 用 sufficiency_log 和 sparsity
                X_density = np.column_stack([
                    X_test[:, feature_map['sufficiency_log']],
                    X_test[:, feature_map['sparsity']]
                ])
            
            X_density_scaled = self.gam_scaler.transform(X_density)
            y_gam_pred = self.gam_model.predict(X_density_scaled)
            
            # 最终预测 = GAM 预测 + 残差预测
            y_pred = y_gam_pred + residual_pred
            
            return y_pred
    
    def predict_stage1_only(self, X_test):
        """
        仅使用第一阶段 GAM (Density) 预测
        
        Args:
            X_test: 测试数据，支持两种格式：
                   - 2 列: [sufficiency_log, sparsity] 直接是density特征
                   - 4 列: [longitude, latitude, sufficiency_log, sparsity] 从中提取density特征
            
        Returns:
            y_pred: 仅 GAM 的预测值（不包括 SVM 残差）
        """
        if self.gam_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        n_cols = X_test.shape[1]
        
        # 根据输入列数决定如何提取特征
        if n_cols == 2:
            # 直接是 [sufficiency_log, sparsity]
            if self.single_sufficiency:
                X_density = X_test[:, 1].reshape(-1, 1)  # 只用 sparsity
            else:
                X_density = X_test  # 用全部两列
        elif n_cols == 4:
            # 从 [longitude, latitude, sufficiency_log, sparsity] 中提取
            if self.single_sufficiency:
                X_density = X_test[:, 3].reshape(-1, 1)  # 只用 sparsity
            else:
                X_density = X_test[:, 2:4]  # 提取 [sufficiency_log, sparsity]
        else:
            raise ValueError(f"X_test must have 2 or 4 columns, got {n_cols}")
        
        X_density_scaled = self.gam_scaler.transform(X_density)
        y_gam_pred = self.gam_model.predict(X_density_scaled)
        
        return y_gam_pred
    
    def predict_stage2_only(self, X_test):
        """
        仅使用第二阶段 SVM (Spatial) 预测残差
        
        Args:
            X_test: 测试数据，必须包含 4 列: [longitude, latitude, sufficiency_log, sparsity]
            
        Returns:
            residual_pred: 仅 SVM 的残差预测值
        """
        if self.svm_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # 假设输入顺序：[longitude, latitude, sufficiency_log, sparsity]
        feature_map = {
            'longitude': 0,
            'latitude': 1,
            'sufficiency_log': 2,
            'sparsity': 3
        }
        
        # Stage 2: SVM 预测残差
        X_spatial = np.column_stack([X_test[:, feature_map[f]] for f in self.stage2_features])
        X_spatial_scaled = self.svm_scaler.transform(X_spatial)
        residual_pred = self.svm_model.predict(X_spatial_scaled)
        
        return residual_pred


def _fit_single_model(model_type, X_train, X_test, y_train, y_test, metric='r2', spline=8, lam=1, return_model=False):
    """训练单个模型的辅助函数
    
    Args:
        model_type: 模型类型，支持 'linear', 'ridge', 'elasticnet', 'rf', 'gbr', 
                   'lightgbm', 'svm', 'svm_with_constraint', 'gam_free', 'gam_monotonic', etc.
        metric: 'r2' or 'correlation'
        spline: GAM样条函数节点数
        lam: GAM正则化参数
        return_model: 如果True，返回(score, model_obj, scaler)；如果False，只返回score
    """
    model_obj = None
    scaler = None
    
    try:
        if model_type == 'linear':
            # scaler = StandardScaler()
            # X_train = scaler.fit_transform(X_train)
            # X_test = scaler.transform(X_test) 

            model_obj = LinearRegression()
            model_obj.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)
            
        elif model_type == 'ridge':
            from sklearn.linear_model import Ridge
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model_obj = Ridge(alpha=1.0, random_state=42)
            model_obj.fit(X_train_scaled, y_train)
            y_pred = model_obj.predict(X_test_scaled)
            
        elif model_type == 'elasticnet':
            from sklearn.linear_model import ElasticNet
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model_obj = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            model_obj.fit(X_train_scaled, y_train)
            y_pred = model_obj.predict(X_test_scaled)
            
        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model_obj = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model_obj.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)
            
        elif model_type == 'gbr':
            from sklearn.ensemble import GradientBoostingRegressor
            model_obj = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            model_obj.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)
            
        elif model_type == 'lightgbm':
            if not HAS_LIGHTGBM:
                if return_model:
                    return -999, None, None
                return -999
            model_obj = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, 
                                random_state=42, verbose=-1)
            model_obj.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)
            
        elif model_type == 'svm':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model_obj = SVR(kernel='rbf', C=1.0, gamma='scale')
            model_obj.fit(X_train_scaled, y_train)
            y_pred = model_obj.predict(X_test_scaled)
            
        elif model_type == 'svm_with_constraint':
            # SVM with light monotonic constraints on density and sufficiency
            # Strategy: Train SVM, then apply soft monotonic regularization
            # REQUIRES: 4 features [longitude, latitude, sufficiency_log, sparsity]
            # n_features = X_train.shape[1] 
            # if n_features != 4:
            #     raise ValueError(
            #         f"[longitude, latitude, sufficiency_log, sparsity], but got {n_features} features. "
            #     )
            
            # scaler = StandardScaler()
            # X_train_scaled = scaler.fit_transform(X_train)
            # X_test_scaled = scaler.transform(X_test)
            
            # # Train base SVM model
            # model_obj = SVR(kernel='rbf', C=1.0, gamma='scale')
            # model_obj.fit(X_train_scaled, y_train)
            # y_pred_raw = model_obj.predict(X_test_scaled)
            
            # Apply constraint: higher sufficiency/sparsity should have higher R²
            # Use isotonic regression on these features as soft guidance
            from sklearn.isotonic import IsotonicRegression
            
            # Create a combined "density score" from sufficiency and sparsity
            # This is used to guide monotonic adjustment
            density_score_train = X_train[:, 2] + X_train[:, 3]  # sufficiency_log + sparsity
            density_score_test = X_test[:, 2] + X_test[:, 3]
            
            # Fit isotonic model as a guide (soft constraint)
            iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')
            iso_reg.fit(density_score_train, y_train)
            y_iso = iso_reg.predict(density_score_test)
            
            # Blend SVM prediction with isotonic guidance (constraint weight adjustable)
            constraint_weight = 1  # High constraint weight
            y_pred = y_iso#(1 - constraint_weight) * y_pred_raw + constraint_weight * y_iso
            
        elif model_type == 'gam_free':
            from pygam import LinearGAM, s, te
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            spline=8
            lam=1
            # 根据特征数量自适应构建GAM
            n_features = X_train.shape[1]
            if n_features == 2:
                model_obj = LinearGAM(
                    s(0, n_splines=spline, lam=lam) + 
                    s(1,  n_splines=spline, lam=lam) 
                    +te(0, 1, n_splines=[max(8, spline//4), max(8, spline//4)], lam=lam*2)
                )
            elif n_features == 4:
                model_obj = LinearGAM(
                    s(0, n_splines=spline, lam=lam) + 
                    s(1, n_splines=spline, lam=lam) + 
                    s(2, n_splines=spline, lam=lam) + 
                    s(3, n_splines=spline, lam=lam)
                                            +
                    te(0, 1, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2)
                    +
                    te(2, 3, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2)
                    +
                    te(0, 2, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2)
                    +
                    te(1, 2, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2)
                    +
                    te(0, 3, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2)
                    +
                    te(1, 3, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2)

                )
            else:
                # 通用情况：根据特征数量动态生成
                terms = [s(i, n_splines=8, lam=1) for i in range(n_features)]
                model_obj = LinearGAM(sum(terms[1:], terms[0]))  # sum with initial value
            
            model_obj.fit(X_train_scaled, y_train)
            y_pred = model_obj.predict(X_test_scaled)
        
        elif model_type == 'gam_monotonic':
            # GAM with monotonic constraints
            from pygam import LinearGAM, s, te
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            n_features = X_train.shape[1]
            if n_features == 2:
                model_obj = LinearGAM(
                    s(0, constraints='monotonic_inc', n_splines=spline, lam=lam) + 
                    s(1, constraints='monotonic_inc', n_splines=spline, lam=lam) +
                    te(0, 1, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2)
                )
            elif n_features == 4:
                model_obj = LinearGAM(
                    s(0, n_splines=spline, lam=lam) +  # longitude (no constraint)
                    s(1, n_splines=spline, lam=lam) +  # latitude (no constraint)
                    s(2, constraints='monotonic_inc', n_splines=spline, lam=lam) +  # sufficiency_log
                    s(3, constraints='monotonic_inc', n_splines=spline, lam=lam) +  # sparsity
                    te(0, 1, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2) +
                    te(2, 3, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2) +
                    te(0, 2, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2) +
                    te(1, 2, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2) +
                    te(0, 3, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2) +
                    te(1, 3, n_splines=[max(4, spline//4), max(4, spline//4)], lam=lam*2)
                )
            else:
                # 通用情况：所有特征都加monotonic
                terms = [s(i, constraints='monotonic_inc', n_splines=spline, lam=lam) for i in range(n_features)]
                model_obj = LinearGAM(sum(terms[1:], terms[0]))
            
            model_obj.fit(X_train_scaled, y_train)
            y_pred = model_obj.predict(X_test_scaled)
        
        elif model_type == 'gam_without_interaction':
            # GAM without interaction terms (only main effects with monotonic constraints)
            from pygam import LinearGAM, s
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            n_features = X_train.shape[1]
            if n_features == 2:
                model_obj = LinearGAM(
                    s(0, n_splines=spline, lam=lam) + 
                    s(1,  n_splines=spline, lam=lam)
                )
            elif n_features == 4:
                model_obj = LinearGAM(
                    s(0, n_splines=spline, lam=lam) +  # longitude (no constraint)
                    s(1, n_splines=spline, lam=lam) +  # latitude (no constraint)
                    s(2,  n_splines=spline, lam=lam) +  # sufficiency_log
                    s(3,  n_splines=spline, lam=lam)     # sparsity
                )
            else:
                # 通用情况
                terms = [s(i,  n_splines=spline, lam=lam) for i in range(n_features)]
                model_obj = LinearGAM(sum(terms[1:], terms[0]))
            
            model_obj.fit(X_train_scaled, y_train)
            y_pred = model_obj.predict(X_test_scaled)
        
        elif model_type == 'gam_mono_noint':
            # GAM with Monotonic constraints + No Interaction (same as gam_without_interaction)
            from pygam import LinearGAM, s
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            n_features = X_train.shape[1]
            if n_features == 2:
                model_obj = LinearGAM(
                    s(0, constraints='monotonic_inc', n_splines=spline, lam=lam) + 
                    s(1, constraints='monotonic_inc', n_splines=spline, lam=lam)
                )
            elif n_features == 4:
                model_obj = LinearGAM(
                    s(0, n_splines=spline, lam=lam) +  # longitude (no constraint)
                    s(1, n_splines=spline, lam=lam) +  # latitude (no constraint)
                    s(2, constraints='monotonic_inc', n_splines=spline, lam=lam) +  # sufficiency_log
                    s(3, constraints='monotonic_inc', n_splines=spline, lam=lam)     # sparsity
                )
            else:
                # 通用情况
                terms = [s(i, constraints='monotonic_inc', n_splines=spline, lam=lam) for i in range(n_features)]
                model_obj = LinearGAM(sum(terms[1:], terms[0]))
            
            model_obj.fit(X_train_scaled, y_train)
            y_pred = model_obj.predict(X_test_scaled)
            
        elif model_type == 'interpolation':
            # IDW (Inverse Distance Weighting) 空间插值
            # 创建 IDW 插值器，提供统一的 .predict() 接口
            model_obj = IDWInterpolator(X_train, y_train)
            y_pred = model_obj.predict(X_test)
        
        else:
            if return_model:
                return -999, None, None
            return -999
        
        # 根据metric选择评估方式
        if metric == 'correlation':
            from scipy.stats import pearsonr
            corr, _ = pearsonr(y_test, y_pred)
            score = corr
        else:  # r2
            score = r2_score(y_test, y_pred)
        
        if return_model:
            return score, model_obj, scaler
        else:
            return score
            
    except Exception as e:
        # 抛出详细的错误信息，而不是返回 -999
        raise RuntimeError(f"Failed to fit {model_type} model: {str(e)}") from e

def eval_baseline_comparison(df, sparsity_bins, split_by='grid', 
                             lam=5.0, spline=14, use_seeds=True, metric='r2', full_features='Density', 
                             resolution=None, split_method='spatial', train_by='grid', evaluate_by='grid',spline_order=3,
                             model_list=None, clip=None):
    """
    基于统一数据分割的基线模型对比评估
    
    所有基线方法都基于统一的空间数据分割进行公平比较
    
    Args:
        df: 原始数据框
        sparsity_bins: sparsity分bin数量
        split_by: 'grid' or 'station' - 空间分割方式
        lam: GAM正则化参数
        spline: GAM样条函数节点数
        use_seeds: 是否使用seeds聚合
        metric: 'r2' or 'correlation' - 评估指标选择
        full_features: 'Density', 'Spatial', or 'Full'
                      'Density': 使用[sufficiency_log, sparsity]
                      'Spatial': 使用[longitude, latitude]
                      'Full': 使用全部4个特征[longitude, latitude, sufficiency_log, sparsity]
        resolution: [lon_bins, lat_bins], 默认[10, 10] - 控制spatial grid的分辨率
        split_method: 训练/测试数据分割方式
                     'spatial': 按 split_by 进行空间分割（默认）
                     'sampling': 按 sufficiency×sparsity bins 分割
        train_by: 训练数据聚合方式
                 - 'grid': 按网格聚合训练数据（默认）
                 - 'station': 按站点聚合训练数据
                 - 'sampling': 按 sufficiency×sparsity bins 聚合训练数据
        evaluate_by: 测试数据聚合方式，可以是字符串或列表
                    - 字符串: 'grid', 'station', 或 'sampling'
                    - 列表: ['grid', 'station', 'sampling'] - 将对每种方式分别评估并输出表格
        model_list: 要训练的模型列表。如果为 None，则训练所有基线模型
                   默认: ['linear', 'ridge', 'elasticnet', 'rf', 'gbr', 'lightgbm', 'svm',
                          'gam_free', 'gam_monotonic', 'gam_without_interaction', 'gam_mono_noint', 
                          'interpolation', 'two_stage', 'two_stage_1']
                   其中双阶段模型包括：
                   - 'two_stage': 完整双阶段模型
                     * Stage 1: GAM_Monotonic (基于 density 特征预测基础 R²)
                     * Stage 2: SVM (基于 spatial 特征预测残差)
                     * Final: 预测 = GAM 预测 + SVM 残差预测
                   - 'two_stage_1': 仅使用第一阶段
                     * Stage 1: GAM_Monotonic (基于 density 特征预测基础 R²)
                     * Final: 预测 = GAM 预测（不包括 SVM 残差）
                   - 'two_stage_2': 仅使用第二阶段
                     * Stage 2: SVM (基于 spatial 特征预测残差)
                     * Final: 预测 = SVM 残差预测（相对于 GAM 基线的偏差）
        
    Returns:
        dict: 如果 evaluate_by 是字符串，返回单个结果字典
              如果 evaluate_by 是列表，返回 {evaluate_method: results_dict} 的嵌套字典
    """
    if resolution is None:
        resolution = [10, 10]
    
    # 设置默认模型列表
    if model_list is None:
        model_list = [
            'linear', 'ridge', 'elasticnet', 'rf', 'gbr', 'lightgbm', 'svm',
            'gam_free', 'gam_monotonic', 'gam_without_interaction', 'gam_mono_noint', 
            'interpolation', 'two_stage', 'two_stage_1', 'two_stage_2'
        ]
    
    # 转换 evaluate_by 为列表
    if isinstance(evaluate_by, str):
        evaluate_by_list = [evaluate_by]
        return_single = True
    else:
        evaluate_by_list = list(evaluate_by)
        return_single = False
    
    # 检查数据
    model_columns = [col for col in df.columns if col.startswith('predicted_')]
    model_names = [col.replace('predicted_', '') for col in model_columns]
    
    # ========================================
    # Step 1: Split raw data and prepare bin intervals
    # ========================================
    df_copy = df.copy()
    
    # 先计算全局 bin intervals（用于后续聚合）
    print(f"\n=== Step 1: Calculate bin intervals and split data ===")
    bins_intervals = find_bins_intervals(df_copy, sparsity_bins)
    
    print(f"  Split method: {split_method}")
    if split_method == 'spatial':
        # Spatial split by grid or station
        df_train_raw, df_test_raw = spatial_split(df_copy, split_by=split_by, train_ratio=0.7, seed=42)
        
    elif split_method == 'sampling':
        # Sampling-based split (by sufficiency×sparsity bins)
        df_train_raw, df_test_raw, train_ids, test_ids, _ = sampling_split_raw(
            df_copy, 
            sparsity_bins=sparsity_bins, 
            train_ratio=0.7, 
            seed=42,
            use_seeds=use_seeds
        )
    print(f"  Train samples: {len(df_train_raw)}, Test samples: {len(df_test_raw)}")
    
    # ========================================
    # Step 2: 聚合训练数据（统一生成所有特征）
    # ========================================
    print(f"\n=== Step 2: Aggregate train data ===")
    
    # 为 Interpolation 准备纯 spatial 训练数据（所有模式共用，不受 train_by 影响）
    print(f"  Preparing pure spatial data for interpolation baseline (split_by={split_by})...")
    spatial_train_dict_pure = prepare_spatial_features(
        df_train_raw, model_names,
        split_by=split_by,
        include_sampling_features=False,  # 不包含 sampling features
        bins_intervals=None,
        resolution=resolution,
        clip=clip  # 传递截断参数
    )
    
    # 根据 train_by 选择聚合方式（用于其他模型）
    if train_by == 'sampling':
        print(f"  Aggregating train data by sufficiency×sparsity bins...")
        train_data_dict = prepare_sampling_features(
            bins_intervals, df_train_raw, model_names, use_seeds,
            include_spatial_coords=True,
            clip=clip  # 传递截断参数
        )
    else:
        # train_by = 'grid' or 'station'
        train_split_by = 'station' if train_by == 'station' else 'grid'
        print(f"  Aggregating train data by spatial {train_split_by} with resolution {resolution}...")
        train_data_dict = prepare_spatial_features(
            df_train_raw, model_names, 
            split_by=train_split_by, 
            include_sampling_features=True,  # 始终生成所有特征
            bins_intervals=bins_intervals,
            resolution=resolution,
            clip=clip  # 传递截断参数
        )
    
    # ========================================
    # Step 3: 训练模型（只训练一次）
    # ========================================
    print(f"\n=== Step 3: Training models (train_by={train_by}, features={full_features}) ===")
    
    # 特征选择配置
    feature_configs = {
        'Full': ['longitude', 'latitude', 'sufficiency_log', 'sparsity'],
        'Spatial': ['longitude', 'latitude'],
        'Density': ['sufficiency_log', 'sparsity']
    }
    feature_cols = feature_configs[full_features]
    
    # 存储训练好的模型
    trained_models = {}
    
    for model_name in tqdm(model_names, desc="Training models"):
        model_train_data = train_data_dict.get(model_name, pd.DataFrame())
        spatial_train_data = spatial_train_dict_pure.get(model_name, pd.DataFrame())
        
        if len(model_train_data) < 5:
            trained_models[model_name] = None
            continue
        
        # 准备训练数据
        X_train = model_train_data[feature_cols].values
        y_train = model_train_data['r2'].values
        
        # 训练所有模型（只训练一次！）
        model_dict = {}
        for model_type in model_list:
            if model_type == 'interpolation':
                # Interpolation: 使用纯 spatial 数据封装训练数据
                X_train_spatial = spatial_train_data[['longitude', 'latitude']].values
                y_train_spatial = spatial_train_data['r2'].values
                _, trained_model, scaler = _fit_single_model(
                    'interpolation', X_train_spatial, X_train_spatial, y_train_spatial, y_train_spatial,
                    metric=metric, spline=spline, lam=lam, return_model=True
                )
                model_dict['interpolation'] = {'model': trained_model, 'scaler': scaler, 'type': 'interpolation'}
            elif model_type in ['two_stage', 'two_stage_1', 'two_stage_2']:
                # Two-Stage Model: GAM (Density) + SVM (Spatial Residual)
                # two_stage_1: Only use Stage 1 (GAM) prediction
                # two_stage_2: Only use Stage 2 (SVM) residual prediction
                try:
                    two_stage_model = TwoStageModel(spline=spline, lam=lam, resolution=resolution, diagnose=False, spline_order=spline_order)
                    stage1_score, stage2_score = two_stage_model.fit(
                        df_train_raw, model_name, bins_intervals, 
                        use_seeds=use_seeds, split_by=split_by, metric=metric
                    )
                    model_dict[model_type] = {'model': two_stage_model, 'scaler': None, 'type': model_type}
                except ValueError as e:
                    # ValueError 通常是致命错误（如数据不足），应该抛出
                    error_msg = str(e)
                    if "Insufficient training samples" in error_msg:
                        raise  # 数据不足错误，直接抛出
                    else:
                        print(f"    Warning: {model_type} training failed for {model_name}: {error_msg}")
                except Exception as e:
                    print(f"    Warning: {model_type} training failed for {model_name}: {e}")
                    model_dict[model_type] = None
            else:
                # 普通模型：训练
                _, trained_model, scaler = _fit_single_model(
                    model_type, X_train, X_train, y_train, y_train,
                    metric=metric, spline=spline, lam=lam, return_model=True
                )
                model_dict[model_type] = {'model': trained_model, 'scaler': scaler, 'type': model_type}
        
        trained_models[model_name] = model_dict
    
    # ========================================
    # Step 4: 在不同测试数据上评估（评估多次）
    # ========================================
    all_results = {}
    
    for eval_method in evaluate_by_list:
        print(f"\n{'='*80}")
        print(f"=== Evaluating on: {eval_method.upper()} ===")
        print(f"{'='*80}")
        
        # 准备测试数据
        if eval_method == 'sampling':
            print(f"  Aggregating test data by sufficiency×sparsity bins...")
            test_data_dict = prepare_sampling_features(
                bins_intervals, df_test_raw, model_names, use_seeds,
                include_spatial_coords=True,
                clip=clip  # 传递截断参数
            )
        else:
            test_split_by = 'station' if eval_method == 'station' else 'grid'
            print(f"  Aggregating test data by spatial {test_split_by} with resolution {resolution}...")
            test_data_dict = prepare_spatial_features(
                df_test_raw, model_names, 
                split_by=test_split_by, 
                include_sampling_features=True,
                bins_intervals=bins_intervals,
                resolution=resolution,
                clip=clip  # 传递截断参数
            )
        
        # 使用训练好的模型进行评估
        results = _evaluate_trained_models(
            trained_models, test_data_dict, model_names, 
            model_list, feature_cols, metric
        )
        
        # 打印当前评估方式的结果
        _print_performance_table(results, model_list=model_list, metric=metric)
        all_results[eval_method] = results
    
    # 返回结果
    if return_single:
        return all_results[evaluate_by_list[0]]
    else:
        return all_results


def analyze_model_performance(df, sparsity_bins=None, use_seeds=True, include_spatial=True, 
                             lam=None, spline=None, target_model='Ours', metric='r2'):
    """
    完整的模型性能分析 - Original 模式（备份函数）
    
    使用原始方法：sufficiency/sparsity 各自random split, spatial 各自random split
    此函数保留作为备份，用于与新方法对比
    
    Args:
        df: 原始数据框
        sparsity_bins: sparsity分bin数量，如果None则自动寻找最优
        use_seeds: 是否使用seeds聚合
        include_spatial: 是否包含空间分析
        lam: GAM正则化参数，如果None则自动搜索最优值
        spline: GAM样条函数节点数，如果None则自动搜索最优值
        target_model: 用于GAM参数优化的目标模型
        metric: 'r2' or 'correlation' - 评估指标选择
        
    Returns:
        dict: 包含所有模型性能结果的字典
    """
    resolution = [10, 10]  # Fixed resolution for original mode
    
    print("=== Ozone Reconstruction Model Performance Analysis (Original Mode - Backup) ===")
    
    # 检查数据
    model_columns = [col for col in df.columns if col.startswith('predicted_')]
    has_spatial = include_spatial and all(col in df.columns for col in ['longitude', 'latitude'])
    
    print(f"Available models: {[col.replace('predicted_', '') for col in model_columns]}")
    print(f"Spatial analysis: {'Enabled' if has_spatial else 'Disabled'}")
    
    # 1. 确定最优sparsity_bins
    if sparsity_bins is None:
        print("Finding optimal sparsity bins...")
        sparsity_bins = find_optimal_sparsity_bins(df, use_seeds=use_seeds, target_model=target_model)
    
    # 2. 确定GAM超参数
    final_lam = lam
    final_spline = spline
    
    if lam is None or spline is None:
        print("Optimizing GAM hyperparameters...")
        try:
            optimal_lam, optimal_spline, gam_score = find_optimal_gam_params(
                df, sparsity_bins, target_model=target_model, use_seeds=use_seeds
            )
            
            if lam is None:
                final_lam = optimal_lam
            if spline is None:
                final_spline = optimal_spline
                
            print(f"Optimal GAM parameters: lam={final_lam:.3f}, spline={final_spline}")
            
        except Exception as e:
            print(f"GAM optimization failed: {e}")
            print("Using default GAM parameters...")
            final_lam = final_lam or 5.0
            final_spline = final_spline or 14
    else:
        print(f"Using specified GAM parameters: lam={final_lam:.3f}, spline={final_spline}")
    
    # 3. 计算R²数据
    print("Calculating R² data with bins...")
    df_bins = calculate_r2_with_bins(df, sparsity_bins, use_seeds=use_seeds, verbose=False)
    
    if len(df_bins) == 0:
        print("No data generated for model comparison!")
        return {}
    
    # 4. 训练GAM模型
    print("Fitting GAM models with optimized parameters...")
    gam_results = fit_gam_models(df, sparsity_bins, use_seeds=use_seeds, 
                                spline=final_spline, lam=final_lam, verbose=False)
    
    # 5. 训练其他模型
    print("Training comparison models...")
    results = {}
    
    for model_name in tqdm(df_bins['model'].unique(), desc="Processing models"):
        model_data = df_bins[df_bins['model'] == model_name].copy()
        
        if len(model_data) < 10:
            continue
        
        # 准备数据
        model_data['sufficiency_log'] = np.log10(model_data['sufficiency'])
        X = model_data[['sufficiency_log', 'sparsity']].values
        y = model_data['r2'].values
        
        # 数据分割（原始方法：随机split）
        if len(X) < 15:
            X_train = X_test = X
            y_train = y_test = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 训练所有模型
        model_results = {}
        for model_type in ['linear', 'ridge', 'elasticnet', 'rf', 'gbr', 'lightgbm', 'svm', 'gam_free']:
            model_results[model_type] = _fit_single_model(model_type, X_train, X_test, y_train, y_test)
        
        # 添加GAM结果
        model_results['gam_monotonic'] = gam_results.get(model_name, {}).get('r2_score', -999)
        
        # 空间分析（原始方法：独立random split）
        if has_spatial:
            spatial_data = _calculate_spatial_r2(df, model_name, resolution=resolution)
            if len(spatial_data) > 0:
                X_spatial = spatial_data[['longitude', 'latitude']].values
                y_spatial = spatial_data['r2'].values
                
                if len(X_spatial) >= 15:
                    X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(
                        X_spatial, y_spatial, test_size=0.3, random_state=42)
                else:
                    X_train_sp = X_test_sp = X_spatial
                    y_train_sp = y_test_sp = y_spatial
                
                model_results['interpolation'] = _fit_single_model('interpolation', X_train_sp, X_test_sp, y_train_sp, y_test_sp)
            else:
                model_results['interpolation'] = -999
        
        results[model_name] = model_results
    
    # 6. 打印结果
    default_model_list = [
        'linear', 'ridge', 'elasticnet', 'rf', 'gbr', 'lightgbm', 'svm',
        'gam_free', 'gam_monotonic', 'gam_without_interaction', 'gam_mono_noint'
    ]
    if has_spatial:
        default_model_list.append('interpolation')
    _print_performance_table(results, model_list=default_model_list, metric=metric)
    
    # 7. 打印使用的参数
    print(f"\nUsed parameters:")
    print(f"  Sparsity bins: {sparsity_bins}")
    print(f"  GAM lam: {final_lam:.3f}")
    print(f"  GAM spline: {final_spline}")
    print(f"  Target model for optimization: {target_model}")
    
    return results

def sampling_split_raw(df, sparsity_bins, train_ratio=0.85, seed=42, use_seeds=True):
    """
    按sampling bins (sufficiency×sparsity组合)分割原始数据
    
    Args:
        df: 原始数据
        sparsity_bins: sparsity分bin数量
        train_ratio: 训练集比例
        seed: 随机种子
        use_seeds: 是否使用seeds聚合
    
    Returns:
        df_train, df_test, train_ids, test_ids, bins_intervals
    """
    # Step 1: 计算bin intervals
    bins_intervals = find_bins_intervals(df, sparsity_bins)
    sparsity_bins_edges, suff_to_bin = bins_intervals
    
    # Step 2: 给原始df分配bins
    df_copy = df.copy()
    df_copy['sparsity_bin'] = pd.cut(df_copy['sparsity'], bins=sparsity_bins_edges, labels=False)
    df_copy['sufficiency_bin'] = df_copy['sufficiency'].map(suff_to_bin)
    df_copy = df_copy.dropna(subset=['sparsity_bin', 'sufficiency_bin'])
    
    # Step 3: 创建pseudo ID并split（使用bin ID而不是原始值！）
    df_copy['pseudo_id'] = 'suf_' + df_copy['sufficiency_bin'].astype(str) + '_spa_' + df_copy['sparsity_bin'].astype(str)
    
    unique_ids = df_copy['pseudo_id'].unique()
    np.random.seed(seed)
    train_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * train_ratio), replace=False)
    test_ids = np.array([pid for pid in unique_ids if pid not in train_ids])
    
    df_train = df_copy[df_copy['pseudo_id'].isin(train_ids)].copy()
    df_test = df_copy[df_copy['pseudo_id'].isin(test_ids)].copy()
    
    return df_train, df_test, train_ids, test_ids, bins_intervals


def sampling_split(df_bins_all, train_ratio=0.85, seed=42):
    """
    [DEPRECATED] 按sampling bins (sufficiency×sparsity组合)分割数据
    
    此函数已废弃，请使用 sampling_split_raw() 代替。
    该函数用于旧流程（先聚合全部数据，再分割），不推荐使用。
    
    Args:
        df_bins_all: 已经按bins聚合的数据（来自calculate_r2_with_bins）
        train_ratio: 训练集比例
        seed: 随机种子
    
    Returns:
        df_bins_train, df_bins_test, train_ids, test_ids
    """
    import warnings
    warnings.warn(
        "sampling_split() is deprecated. Use sampling_split_raw() instead for splitting raw data before aggregation.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # 创建pseudo ID（每个sufficiency/sparsity组合）
    df_bins_all['pseudo_id'] = 'suf_' + df_bins_all['sufficiency'].astype(str) + '_spa_' + df_bins_all['sparsity'].astype(str)
    
    unique_ids = df_bins_all['pseudo_id'].unique()
    
    # 按pseudo ID split
    np.random.seed(seed)
    train_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * train_ratio), replace=False)
    test_ids = np.array([pid for pid in unique_ids if pid not in train_ids])
    
    df_bins_train = df_bins_all[df_bins_all['pseudo_id'].isin(train_ids)].copy()
    df_bins_test = df_bins_all[df_bins_all['pseudo_id'].isin(test_ids)].copy()
    
    return df_bins_train, df_bins_test, train_ids, test_ids


def spatial_split(df, split_by='grid', train_ratio=0.85, seed=42):
    """
    按spatial方式分割数据
    
    Args:
        df: 原始数据
        split_by: 'grid' or 'station'
        train_ratio: 训练集比例
        seed: 随机种子
    
    Returns:
        df_train, df_test
    """
    df_copy = df.copy()
    
    if split_by == 'grid':
        # 按10x10空间网格分割
        df_copy['longitude_bin'] = pd.cut(df_copy['longitude'], bins=10, labels=False)
        df_copy['latitude_bin'] = pd.cut(df_copy['latitude'], bins=10, labels=False)
        df_copy['split_id'] = 'grid_' + df_copy['longitude_bin'].astype(str) + '_' + df_copy['latitude_bin'].astype(str)
    else:  # station
        if 'Site_number' in df_copy.columns:
            df_copy['split_id'] = df_copy['Site_number']
        else:
            # 用经纬度组合作为站点ID
            df_copy['split_id'] = df_copy.groupby(['longitude', 'latitude']).ngroup()
    
    df_copy = df_copy.dropna(subset=['split_id'])
    
    unique_ids = df_copy['split_id'].unique()
    np.random.seed(seed)
    train_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * train_ratio), replace=False)
    test_ids = np.array([sid for sid in unique_ids if sid not in train_ids])
    
    df_train = df_copy[df_copy['split_id'].isin(train_ids)].copy()
    df_test = df_copy[df_copy['split_id'].isin(test_ids)].copy()
    
    return df_train, df_test


def find_bins_intervals(df, sparsity_bins):
    """
    在全局df上找到bin边界
    
    Returns:
        sparsity_bins_edges, sufficiency_to_bin_mapping
    """
    # Sparsity bin边界
    sparsity_cut_result = pd.cut(df['sparsity'], bins=sparsity_bins, labels=False, retbins=True)
    sparsity_bins_edges = sparsity_cut_result[1]
    
    # Sufficiency bin映射（模仿calculate_r2_with_bins）
    unique_suff_values = np.sort(df['sufficiency'].unique())
    n_unique = len(unique_suff_values)
    sufficiency_bins_num = n_unique
    values_per_bin = n_unique // sufficiency_bins_num
    remainder = n_unique % sufficiency_bins_num
    
    suff_to_bin = {}
    current_idx = 0
    for bin_idx in range(sufficiency_bins_num):
        current_bin_size = values_per_bin + (1 if bin_idx < remainder else 0)
        for i in range(current_bin_size):
            if current_idx < n_unique:
                suff_to_bin[unique_suff_values[current_idx]] = bin_idx
                current_idx += 1
    
    return sparsity_bins_edges, suff_to_bin


def prepare_sampling_features(bins_intervals, df, model_names, use_seeds=False, include_spatial_coords=False, clip=None):
    """
    准备训练数据：按sufficiency/sparsity bins聚合
    
    Args:
        bins_intervals: (sparsity_bins_edges, suff_to_bin)
        df: 训练数据
        model_names: 模型名称列表
        include_spatial_coords: 如果True，也包含longitude/latitude均值
        clip: R² 截断范围 [min, max]，在计算 R² 后立即截断
              例如 [-0.5, 1.0] 表示将 R² 限制在 [-0.5, 1.0] 范围内
              None 表示不截断
    
    Returns:
        {model_name: DataFrame with columns [sufficiency_log, sparsity, r2] 
                     or [sufficiency_log, sparsity, longitude, latitude, r2] if include_spatial_coords=True}
    """
    sparsity_bins_edges, suff_to_bin = bins_intervals
    
    df_copy = df.copy()
    df_copy['sparsity_bin'] = pd.cut(df_copy['sparsity'], bins=sparsity_bins_edges, labels=False)
    df_copy['sufficiency_bin'] = df_copy['sufficiency'].map(suff_to_bin)
    df_copy = df_copy.dropna(subset=['sparsity_bin', 'sufficiency_bin'])
    
    results_dict = {}
    
    for model_name in model_names:
        model_col = f'predicted_{model_name}'
        if model_col not in df_copy.columns:
            continue
        
        results = []
        group_cols = ['sufficiency_bin', 'sparsity_bin']
        
        for name, group in df_copy.groupby(group_cols):
            if len(group) < 10:
                continue
            
            observed = group['observed'].values
            predicted = group[model_col].values
            valid_mask = ~(np.isnan(observed) | np.isnan(predicted) | 
                          np.isinf(observed) | np.isinf(predicted))
            
            if valid_mask.sum() < 5:
                continue
            
            from sklearn.metrics import r2_score
            r2 = r2_score(observed[valid_mask], predicted[valid_mask])
            
            # 🔧 Clip R² 值（在源数据阶段）
            if clip is not None and len(clip) == 2:
                r2 = np.clip(r2, clip[0], clip[1])
            
            result = {
                'sufficiency_log': np.log10(group['sufficiency'].mean()),
                'sparsity': group['sparsity'].mean(),
                'sufficiency_bin': name[0],  # 保留 bin 信息
                'sparsity_bin': name[1],     # 保留 bin 信息
                'r2': r2
            }
            
            if include_spatial_coords:
                result['longitude'] = group['longitude'].mean()
                result['latitude'] = group['latitude'].mean()
            
            results.append(result)
        
        results_dict[model_name] = pd.DataFrame(results)
    
    return results_dict


def prepare_sampling_features_space(bins_intervals, df, model_names, split_by='grid', resolution=None, use_seeds=False, clip=None):
    """
    两阶段聚合：先按空间网格聚合，再按sufficiency/sparsity bins聚合
    
    这是 prepare_sampling_features 的空间增强版本：
    - prepare_sampling_features: 直接对原始数据按 sufficiency×sparsity 聚合
    - prepare_sampling_features_space: 先空间聚合 → 再 sufficiency×sparsity 聚合
    
    注意：此函数依赖于 prepare_spatial_features 返回的 bin 信息，避免反向计算
          sufficiency (10^sufficiency_log) 时的浮点数精度损失。
    
    Args:
        bins_intervals: (sparsity_bins_edges, suff_to_bin) - 传递给 prepare_spatial_features
        df: 训练数据
        model_names: 模型名称列表
        split_by: 'grid' or 'station'
        resolution: [lon_bins, lat_bins], 默认[10, 10]
        use_seeds: 兼容参数（未使用）
        clip: R² 截断范围 [min, max]，传递给 prepare_spatial_features
    
    Returns:
        {model_name: DataFrame with columns [sufficiency_log, sparsity, longitude, latitude, r2]}
    """
    # Step 1: 先调用 prepare_spatial_features 获取空间聚合数据（包含 density 特征）
    spatial_results = prepare_spatial_features(
        df, model_names,
        split_by=split_by,
        include_sampling_features=True,
        bins_intervals=bins_intervals,
        resolution=resolution,
        clip=clip  # 传递截断参数
    )
    
    # Step 2: 再对空间聚合结果按 sufficiency×sparsity 聚合
    results_dict = {}
    
    for model_name, df_spatial in spatial_results.items():
        if len(df_spatial) == 0:
            results_dict[model_name] = pd.DataFrame()
            continue
        
        # 检查是否有 bin 信息（从 prepare_spatial_features 传递过来）
        if 'sufficiency_bin' not in df_spatial.columns or 'sparsity_bin' not in df_spatial.columns:
            raise ValueError("spatial_results must contain 'sufficiency_bin' and 'sparsity_bin' columns")
        
        # 直接使用已有的 bin 列，避免反向计算精度损失
        df_copy = df_spatial.copy()
        df_copy = df_copy.dropna(subset=['sparsity_bin', 'sufficiency_bin'])
        
        # 按 sufficiency_bin × sparsity_bin 聚合
        results = []
        for name, group in df_copy.groupby(['sufficiency_bin', 'sparsity_bin']):
            if len(group) < 1:
                continue
            
            result = {
                'sufficiency_log': group['sufficiency_log'].mean(),
                'sparsity': group['sparsity'].mean(),
                'longitude': group['longitude'].mean(),
                'latitude': group['latitude'].mean(),
                'sufficiency_bin': name[0],  # 保留 bin 信息
                'sparsity_bin': name[1],     # 保留 bin 信息
                'r2': group['r2'].mean()  # 对已经聚合的 R² 再取平均
            }
            results.append(result)
        
        results_dict[model_name] = pd.DataFrame(results)
    
    return results_dict


def prepare_spatial_features(df, model_names, split_by='grid', include_sampling_features=True, bins_intervals=None, resolution=None, outlier_threshold=None, clip=None):
    """
    准备测试数据：按spatial方式聚合，可选地包含sampling特征
    
    Args:
        df: 测试数据
        model_names: 模型名称列表
        split_by: 'grid' or 'station'
        include_sampling_features: 是否包含sufficiency_log和sparsity特征
        bins_intervals: 如果include_sampling_features=True，需要提供bin边界
        resolution: [lon_bins, lat_bins], 默认[10, 10]
        outlier_threshold: R² 异常值处理（已废弃，建议使用 clip）
                          - 数字（如 -0.5）：将 R² <= threshold 的值替换为 threshold
                          - 字符串（如 "0_remove", "-0.5_remove"）：移除 R² <= threshold 的数据点
                          - None：不进行任何处理
        clip: R² 截断范围 [min, max]，在计算 R² 后立即截断
              例如 [-0.5, 1.0] 表示将 R² 限制在 [-0.5, 1.0] 范围内
              None 表示不截断
    
    Returns:
        {model_name: DataFrame with columns [longitude, latitude, (sparsity, sufficiency_log), r2]}
    """
    if resolution is None:
        resolution = [10, 10]
    
    df_copy = df.copy()
    
    if split_by == 'grid':
        df_copy['longitude_bin'] = pd.cut(df_copy['longitude'], bins=resolution[0], labels=False)
        df_copy['latitude_bin'] = pd.cut(df_copy['latitude'], bins=resolution[1], labels=False)
        group_cols = ['longitude_bin', 'latitude_bin']
    else:  # station
        if 'Site_number' in df_copy.columns:
            df_copy['station_id'] = df_copy['Site_number']
        else:
            df_copy['station_id'] = df_copy.groupby(['longitude', 'latitude']).ngroup()
        group_cols = ['station_id']
    
    df_copy = df_copy.dropna(subset=group_cols)
    
    # 如果需要sampling特征，分配bins
    if include_sampling_features and bins_intervals is not None:
        sparsity_bins_edges, suff_to_bin = bins_intervals
        df_copy['sparsity_bin'] = pd.cut(df_copy['sparsity'], bins=sparsity_bins_edges, labels=False)
        df_copy['sufficiency_bin'] = df_copy['sufficiency'].map(suff_to_bin)
        group_cols = group_cols + ['sufficiency_bin']
    
    results_dict = {}
    
    for model_name in model_names:
        model_col = f'predicted_{model_name}'
        if model_col not in df_copy.columns:
            continue
        
        results = []
        
        for name, group in df_copy.groupby(group_cols):
            if len(group) < 3:
                continue
            
            observed = group['observed'].values
            predicted = group[model_col].values
            valid_mask = ~(np.isnan(observed) | np.isnan(predicted) | 
                          np.isinf(observed) | np.isinf(predicted))
            
            if valid_mask.sum() < 5:
                continue
            
            from sklearn.metrics import r2_score
            r2 = r2_score(observed[valid_mask], predicted[valid_mask])
            
            # 🔧 Clip R² 值（在源数据阶段，优先于 outlier_threshold）
            if clip is not None and len(clip) == 2:
                r2 = np.clip(r2, clip[0], clip[1])
            
            # Handle outlier threshold (legacy, 建议使用 clip)
            if outlier_threshold is not None:
                # Check if it's a string with "_remove" suffix
                if isinstance(outlier_threshold, str) and outlier_threshold.endswith('_remove'):
                    # Extract threshold value and remove if R² <= threshold
                    threshold_str = outlier_threshold.replace('_remove', '')
                    try:
                        threshold_val = float(threshold_str)
                        if r2 <= threshold_val:
                            continue  # Skip this data point (remove)
                    except ValueError:
                        pass  # Invalid format, ignore
                elif isinstance(outlier_threshold, (int, float)):
                    # Numeric threshold: replace value
                    if r2 <= outlier_threshold:
                        r2 = outlier_threshold
            
            result = {
                'longitude': group['longitude'].mean(),
                'latitude': group['latitude'].mean(),
                'r2': r2
            }
            
            if include_sampling_features:
                result['sparsity'] = group['sparsity'].mean()
                result['sufficiency_log'] = np.log10(group['sufficiency'].mean())
                # 保留 bin 信息（用于后续聚合，避免反向计算精度损失）
                result['sparsity_bin'] = group['sparsity_bin'].iloc[0]  # bin 在组内是相同的
                result['sufficiency_bin'] = group['sufficiency_bin'].iloc[0]
            
            results.append(result)
        
        results_dict[model_name] = pd.DataFrame(results)
    
    return results_dict
  

def _calculate_spatial_r2(df, model_name, resolution=None):
    """计算空间R²数据的辅助函数"""
    if resolution is None:
        resolution = [10, 10]
    
    model_col = f'predicted_{model_name}'
    if model_col not in df.columns:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['longitude_bin'] = pd.cut(df_copy['longitude'], bins=resolution[0], labels=False)
    df_copy['latitude_bin'] = pd.cut(df_copy['latitude'], bins=resolution[1], labels=False)
    
    results = []
    for name, group in df_copy.groupby(['longitude_bin', 'latitude_bin']):
        if len(group) < 10 or pd.isna(name[0]) or pd.isna(name[1]):
            continue
        
        observed = group['observed'].values
        predicted = group[model_col].values
        valid_mask = ~(np.isnan(observed) | np.isnan(predicted) | 
                      np.isinf(observed) | np.isinf(predicted))
        
        if valid_mask.sum() >= 5:
            from sklearn.metrics import r2_score
            r2 = r2_score(observed[valid_mask], predicted[valid_mask])
            results.append({
                'longitude': group['longitude'].mean(),
                'latitude': group['latitude'].mean(),
                'r2': r2
            })
    
    return pd.DataFrame(results)

def _evaluate_trained_models(trained_models, test_data_dict, model_names, model_list, feature_cols, metric):
    """
    使用训练好的模型在测试数据上评估
    
    Args:
        trained_models: {model_name: {model_type: {'model': ..., 'scaler': ...}}}
        test_data_dict: {model_name: test_dataframe}
        model_names: 模型名称列表
        model_list: 要评估的模型类型列表
        feature_cols: 特征列名列表
        metric: 'r2' or 'correlation'
    
    Returns:
        results: {model_name: {model_type: score}}
    """
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    
    results = {}
    
    for model_name in model_names:
        model_results = {}
        model_dict = trained_models.get(model_name)
        model_test_data = test_data_dict.get(model_name, pd.DataFrame())
        
        if model_dict is None or len(model_test_data) < 3:
            for model_type in model_list:
                model_results[model_type] = -999
            results[model_name] = model_results
            continue
        
        # 准备测试数据
        X_test_normal = model_test_data[feature_cols].values
        y_test = model_test_data['r2'].values
        
        # 用训练好的模型预测并评估
        for model_type in model_list:
            model_info = model_dict.get(model_type)
            if model_info is None:
                model_results[model_type] = -999
                continue
            
            model_obj = model_info['model']
            scaler = model_info['scaler']
            
            # 预测并评估
            try:
                # 选择测试数据
                if model_type == 'interpolation':
                    # Interpolation: 使用 lon/lat
                    X_test = model_test_data[['longitude', 'latitude']].values
                elif model_type in ['two_stage', 'two_stage_1', 'two_stage_2']:
                    # Two-Stage: 需要所有 4 个特征 [lon, lat, suff_log, sparsity]
                    required_cols = ['longitude', 'latitude', 'sufficiency_log', 'sparsity']
                    if not all(col in model_test_data.columns for col in required_cols):
                        model_results[model_type] = -999
                        continue
                    X_test = model_test_data[required_cols].values
                else:
                    # 其他模型：使用指定特征
                    X_test = X_test_normal
                
                # 统一调用 .predict() 接口
                if scaler is not None:
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model_obj.predict(X_test_scaled)
                else:
                    # 特殊处理：two_stage 系列
                    if model_type == 'two_stage_1' and hasattr(model_obj, 'predict_stage1_only'):
                        y_pred = model_obj.predict_stage1_only(X_test)
                    elif model_type == 'two_stage_2' and hasattr(model_obj, 'predict_stage2_only'):
                        y_pred = model_obj.predict_stage2_only(X_test)
                    else:
                        y_pred = model_obj.predict(X_test)
                
                # 计算评估指标
                if metric == 'correlation':
                    score, _ = pearsonr(y_test, y_pred)
                else:
                    score = r2_score(y_test, y_pred)
                
                model_results[model_type] = score
            except Exception as e:
                model_results[model_type] = -999
        
        results[model_name] = model_results
    
    return results


def _print_performance_table(results, model_list, metric='r2'):
    """打印性能表格的辅助函数"""
    metric_name = 'R²' if metric == 'r2' else 'Correlation'
    
    def format_score(score):
        return f"{score:.4f}" if score != -999 else "FAILED"
    
    # 模型名称映射（用于表头显示）
    model_display_names = {
        'linear': 'Linear', 'ridge': 'Ridge', 'elasticnet': 'ElasticNet',
        'rf': 'RF', 'gbr': 'GBR', 'lightgbm': 'LightGBM', 'svm': 'SVM',
        'gam_free': 'GAM_Free', 'gam_monotonic': 'GAM_Mono',
        'gam_without_interaction': 'GAM_NoInt', 'gam_mono_noint': 'GAM_MonoNoInt',
        'interpolation': 'Interp', 'two_stage': 'TwoStage', 'two_stage_1': 'Stage1', 
        'two_stage_2': 'Stage2_Resid'
    }
    
    # 表头
    headers = ['Model'] + [model_display_names.get(m, m) for m in model_list]
    col_width = 12
    table_width = 15 + col_width * len(model_list)
    
    print(f"\n{'='*table_width}")
    print(f"Model Performance Comparison ({metric_name})")
    print(f"{'='*table_width}")
    print(f"{headers[0]:<15} " + " ".join(f"{h:<{col_width}}" for h in headers[1:]))
    print("-" * table_width)
    
    # 数据行
    for model_name, result in results.items():
        row = [f"{model_name:<15}"]
        for method in model_list:
            row.append(f"{format_score(result.get(method, -999)):<{col_width}}")
        print(" ".join(row))
    
    # 计算平均行
    averages = {}
    for method in model_list:
        valid_scores = [result[method] for result in results.values() if method in result and result[method] != -999]
        averages[method] = np.mean(valid_scores) if valid_scores else -999
    
    # 打印平均行
    print("-" * table_width)
    avg_row = [f"{'AVERAGE':<15}"]
    for method in model_list:
        avg_row.append(f"{format_score(averages[method]):<{col_width}}")
    print(" ".join(avg_row))
    
    print("="*table_width)
    print("Note: 'FAILED' indicates model training failed")

def compare_gam_types(df, sparsity_bins, target_model='Ours', use_seeds=True, 
                     spline=14, lam=5, remove_outliers=True):
    """
    比较不同单调函数类型的GAM模型效果
    
    Args:
        df: 数据框
        sparsity_bins: sparsity分bin数量
        target_model: 目标模型名称
        use_seeds: 是否使用seeds聚合
        spline: 样条函数节点数
        lam: 正则化参数
        remove_outliers: 是否移除异常值
        
    Returns:
        dict: 各种GAM类型的拟合结果
    """
    print(f"🔄 Comparing different monotonic GAM types for {target_model}...")
    
    gam_types = ['spline', 'linear', 'mixed', 'polynomial']
    results = {}
    
    for gam_type in gam_types:
        print(f"\n📊 Testing {gam_type} GAM...")
        
        try:
            fitted_models = fit_gam_models(
                df, sparsity_bins, use_seeds=use_seeds, 
                spline=spline, lam=lam, verbose=False,
                remove_outliers=remove_outliers, gam_type=gam_type
            )
            
            if target_model in fitted_models:
                model_info = fitted_models[target_model]
                r2_val = model_info['r2_score']
                results[gam_type] = {
                    'r2_score': r2_val,
                    'model_info': model_info,
                    'status': 'success'
                }
                print(f"   ✅ {gam_type}: R² = {r2_val:.4f}")
            else:
                results[gam_type] = {'status': 'failed', 'error': 'Model not found'}
                print(f"   ❌ {gam_type}: Failed - Model not found")
                
        except Exception as e:
            results[gam_type] = {'status': 'failed', 'error': str(e)}
            print(f"   ❌ {gam_type}: Failed - {str(e)}")
    
    # 显示排名
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}
    if successful_results:
        print(f"\n🏆 GAM Type Rankings (by R²):")
        sorted_results = sorted(successful_results.items(), 
                              key=lambda x: x[1]['r2_score'], reverse=True)
        
        for i, (gam_type, info) in enumerate(sorted_results, 1):
            r2_val = info['r2_score']
            print(f"   {i}. {gam_type.upper():<12}: R² = {r2_val:.4f}")
    
    return results


# =============================================================================
# Plotting Helper Functions
# =============================================================================

def calculate_station_r2(df_analysis, model_name, sufficiency):
    """
    Calculate R² for each station (longitude, latitude).
    
    Args:
        df_analysis: DataFrame with observed and predicted values
        model_name: Name of the model (e.g., 'lightgbm')
        sufficiency: Sufficiency value to filter data
    
    Returns:
        DataFrame with columns: longitude, latitude, r2
    """
    df_plot = df_analysis[df_analysis['sufficiency'] == sufficiency].copy()
    
    predicted_col = f'predicted_{model_name}'
    if predicted_col not in df_plot.columns:
        print(f"Warning: Column '{predicted_col}' not found in df_analysis")
        return pd.DataFrame(columns=['longitude', 'latitude', 'r2'])
    
    station_r2_list = []
    for (lon, lat), group in df_plot.groupby(['longitude', 'latitude']):
        if len(group) > 1:
            r2 = r2_score(group['observed'], group[predicted_col])
            station_r2_list.append({'longitude': lon, 'latitude': lat, 'r2': r2})
    
    return pd.DataFrame(station_r2_list)


def plot_observation_points(ax, station_data, cmap, vmin, vmax, marker_size=15):
    """
    Plot station observation points on the map.
    
    Args:
        ax: matplotlib Axes object
        station_data: DataFrame with longitude, latitude, r2 columns
        cmap: matplotlib colormap
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        marker_size: Size of scatter points
    
    Returns:
        scatter: matplotlib scatter object
    """
    if len(station_data) == 0:
        print("Warning: No valid stations to plot")
        return None
    
    scatter = ax.scatter(
        station_data['longitude'],
        station_data['latitude'],
        c=station_data['r2'],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=marker_size,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )
    return scatter


def plot_grid_average(ax, df_analysis, model_name, sufficiency, cmap, vmin, vmax, grid_shape=(15, 20)):
    """
    Plot grid-averaged accuracy by calculating R² from all points within each grid cell.
    
    Args:
        ax: matplotlib Axes object
        df_analysis: DataFrame with observed, predicted, longitude, latitude columns
        model_name: Name of the model
        sufficiency: Sufficiency value to filter data
        cmap: matplotlib colormap
        vmin, vmax: Colormap range
        grid_shape: Tuple (n_lat_bins, n_lon_bins)
    
    Returns:
        pcolormesh: matplotlib pcolormesh object
    """
    df_plot = df_analysis[df_analysis['sufficiency'] == sufficiency].copy()
    
    predicted_col = f'predicted_{model_name}'
    if predicted_col not in df_plot.columns or len(df_plot) == 0:
        print(f"Warning: No data for {model_name}")
        return None
    
    n_lat_bins, n_lon_bins = grid_shape
    
    lon_min, lon_max = df_plot['longitude'].min(), df_plot['longitude'].max()
    lat_min, lat_max = df_plot['latitude'].min(), df_plot['latitude'].max()
    
    lon_bins = np.linspace(lon_min, lon_max, n_lon_bins + 1)
    lat_bins = np.linspace(lat_min, lat_max, n_lat_bins + 1)
    
    df_plot['lon_bin'] = np.digitize(df_plot['longitude'], lon_bins) - 1
    df_plot['lat_bin'] = np.digitize(df_plot['latitude'], lat_bins) - 1
    df_plot['lon_bin'] = df_plot['lon_bin'].clip(0, n_lon_bins - 1)
    df_plot['lat_bin'] = df_plot['lat_bin'].clip(0, n_lat_bins - 1)
    
    grid_r2 = np.full((n_lat_bins, n_lon_bins), np.nan)
    
    for (lat_bin, lon_bin), group in df_plot.groupby(['lat_bin', 'lon_bin']):
        if len(group) > 1:
            r2 = r2_score(group['observed'], group[predicted_col])
            grid_r2[int(lat_bin), int(lon_bin)] = r2
    
    mesh = ax.pcolormesh(lon_bins, lat_bins, grid_r2, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, shading='flat')
    return mesh


def plot_anomaly_stations(ax, station_data, marker_size=20, marker_color='red'):
    """
    Plot anomaly stations (R² < 0).
    
    Args:
        ax: matplotlib Axes object
        station_data: DataFrame with longitude, latitude, r2 columns
        marker_size: Size of scatter points
        marker_color: Color for anomaly markers
    
    Returns:
        scatter: matplotlib scatter object
    """
    anomaly_data = station_data[station_data['r2'] < 0].copy()
    
    if len(anomaly_data) == 0:
        print("No anomaly stations (R² < 0) found")
        return None
    
    print(f"Found {len(anomaly_data)} anomaly stations with R² < 0")
    
    scatter = ax.scatter(
        anomaly_data['longitude'],
        anomaly_data['latitude'],
        c=marker_color,
        s=marker_size,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.8,
        marker='x',
        label=f'Anomaly (R²<0): {len(anomaly_data)}'
    )
    ax.legend(loc='upper right', fontsize=8)
    return scatter


def plot_station_points(ax, df_data, color='#2C2C2C', alpha=0.3, size=6, marker=None):
    """
    Plot station distribution points.
    
    Args:
        ax: matplotlib Axes object
        df_data: DataFrame with longitude and latitude columns
        color: Point color
        alpha: Transparency
        size: Point size
        marker: Marker style
    
    Returns:
        scatter: scatter plot object
    """
    unique_locs = df_data.groupby(['longitude', 'latitude']).size()
    data_lons = unique_locs.index.get_level_values('longitude')
    data_lats = unique_locs.index.get_level_values('latitude')
    
    scatter = ax.scatter(data_lons, data_lats, c=color, alpha=alpha, s=size, 
                        edgecolors='none', marker='o' if marker is None else marker)
    return scatter


def plot_multipolygon_edges(ax, gdf, edge_color='black', linewidth=1, alpha=1):
    """
    Plot polygon edges from a GeoDataFrame.
    
    Args:
        ax: matplotlib Axes object
        gdf: GeoDataFrame with geometry
        edge_color: Edge line color
        linewidth: Edge line width
        alpha: Transparency
    """
    for geometry in gdf.geometry:
        if geometry.geom_type == 'Polygon':
            x, y = geometry.exterior.coords.xy
            ax.plot(x, y, color=edge_color, linewidth=linewidth, alpha=alpha)
        elif geometry.geom_type == 'MultiPolygon':
            for polygon in geometry.geoms:
                x, y = polygon.exterior.coords.xy
                ax.plot(x, y, color=edge_color, linewidth=linewidth, alpha=alpha)


def plot_sparsity_quantile_contour(ax, ds_sparsity, count_number=3, levels=None, colors=None, 
                                   linewidth=2.5, alpha=1, show_labels=True, label_format=None, 
                                   line_styles=None):
    """
    Plot sparsity density quantile contours.
    
    Args:
        ax: matplotlib Axes object
        ds_sparsity: xarray Dataset with sparsity values
        count_number: Number of quantile lines (used only if levels=None)
        levels: List of specific level values (overrides quantiles)
        colors: List of colors for each contour line
        linewidth: Line width (single value or list)
        alpha: Transparency
        show_labels: Whether to show contour labels
        label_format: Label format string
        line_styles: List of line styles
    
    Returns:
        contours: list of contour objects
    """
    import matplotlib.pyplot as plt
    
    lons = ds_sparsity.longitude.values
    lats = ds_sparsity.latitude.values
    sparsity_grid = ds_sparsity['sparsity'].values
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    if levels is not None:
        quantile_levels = np.array(levels)
        count_number = len(levels)
        valid_values = sparsity_grid[~np.isnan(sparsity_grid)]
        percentiles = [np.sum(valid_values <= level) / len(valid_values) * 100 for level in levels] if len(valid_values) > 0 else [0] * len(levels)
    else:
        valid_values = sparsity_grid[~np.isnan(sparsity_grid)]
        if len(valid_values) == 0:
            print("Warning: No valid sparsity values found")
            return []
        percentiles = np.linspace(0, 100, count_number + 2)[1:-1]
        quantile_levels = np.percentile(valid_values, percentiles)
        percentiles = percentiles[::-1]
        quantile_levels = quantile_levels[::-1]
    
    if colors is None or len(colors) != count_number:
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, count_number))
    
    if line_styles is None or len(line_styles) != count_number:
        line_styles = ['-'] * count_number
    
    if isinstance(linewidth, (list, tuple, np.ndarray)):
        linewidths = list(linewidth) if len(linewidth) == count_number else [linewidth[0]] * count_number
    else:
        linewidths = [linewidth] * count_number
    
    contours = []
    for i, (level, color, percentile, line_style, lw) in enumerate(zip(quantile_levels, colors, percentiles, line_styles, linewidths)):
        contour = ax.contour(lon_grid, lat_grid, sparsity_grid, levels=[level], colors=[color], linewidths=lw, alpha=alpha, linestyles=line_style)
        
        if show_labels:
            if label_format is not None:
                label_text = label_format.format(percentile=percentile, level=level, index=i+1)
            else:
                label_text = f'{level:.2f}' if levels is not None else f'P{percentile:.0f}'
            ax.clabel(contour, inline=True, fontsize=8, fmt=label_text)
        
        contours.append(contour)
    
    return contours


def _get_model_info(fitted_models, model_name, model_type=None):
    """
    Extract model_info from fitted_models dictionary.
    
    Args:
        fitted_models: Dictionary of fitted models
        model_name: Model name
        model_type: Model type (optional, for API compatibility)
    
    Returns:
        model_info: Dictionary with 'model', 'scaler', 'full_features', etc.
    """
    if model_name not in fitted_models:
        raise ValueError(f"Model '{model_name}' not found in fitted_models")
    return fitted_models[model_name]


def predict_accuracy_grid(model_info, prediction_input):
    """
    Predict accuracy values using the model.
    
    Args:
        model_info: Dictionary containing model and scaler
        prediction_input: Input data for prediction
    
    Returns:
        accuracy_pred: Predicted accuracy values
    """
    if isinstance(model_info, dict):
        model = model_info['model']
        scaler = model_info.get('scaler')
        model_type = model_info.get('model_type')
    else:
        model = model_info
        scaler = None
        model_type = None
    
    if scaler is not None:
        prediction_input = scaler.transform(prediction_input)
    
    if model_type == 'two_stage_1' and hasattr(model, 'predict_stage1_only'):
        return model.predict_stage1_only(prediction_input)
    elif model_type == 'two_stage_2' and hasattr(model, 'predict_stage2_only'):
        return model.predict_stage2_only(prediction_input)
    else:
        return model.predict(prediction_input)


def _prepare_accuracy_grid(fitted_models, model_name, ds_sparsity, sufficiency_value, model_type=None):
    """
    Prepare accuracy grid for a given sufficiency value.
    
    Args:
        fitted_models: Dictionary of fitted models
        model_name: Model name
        ds_sparsity: xarray Dataset with sparsity values
        sufficiency_value: Fixed sufficiency value
        model_type: Model type (optional)
    
    Returns:
        lon_grid, lat_grid, accuracy_grid: Coordinate grids and predicted accuracy
    """
    model_info = _get_model_info(fitted_models, model_name, model_type)
    
    lons = ds_sparsity.longitude.values
    lats = ds_sparsity.latitude.values
    sparsity_grid = ds_sparsity['sparsity'].values
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    full_features = model_info.get('full_features', 'Density')
    sparsity_values = sparsity_grid.ravel()
    
    if isinstance(full_features, list):
        lon_values = lon_grid.ravel()
        lat_values = lat_grid.ravel()
        sufficiency_values = np.full(len(sparsity_values), np.log10(sufficiency_value))
        
        feature_map = {
            'longitude': lon_values,
            'latitude': lat_values,
            'sufficiency_log': sufficiency_values,
            'sparsity': sparsity_values
        }
        model_input = np.column_stack([feature_map[feat] for feat in full_features])
        
    elif full_features == 'Spatial':
        lon_values = lon_grid.ravel()
        lat_values = lat_grid.ravel()
        model_input = np.column_stack([lon_values, lat_values])
        
    elif full_features == 'Full':
        lon_values = lon_grid.ravel()
        lat_values = lat_grid.ravel()
        sufficiency_log = np.full(len(sparsity_values), np.log10(sufficiency_value))
        model_input = np.column_stack([lon_values, lat_values, sufficiency_log, sparsity_values])
        
    else:  # Density
        sufficiency_log = np.full(len(sparsity_values), np.log10(sufficiency_value))
        model_input = np.column_stack([sufficiency_log, sparsity_values])
    
    valid_mask = ~np.isnan(sparsity_values)
    
    if np.sum(valid_mask) == 0:
        accuracy_grid = np.full(sparsity_grid.shape, np.nan)
    else:
        valid_model_input = model_input[valid_mask]
        accuracy_pred_valid = predict_accuracy_grid(model_info, valid_model_input)
        
        accuracy_pred = np.full(len(sparsity_values), np.nan)
        accuracy_pred[valid_mask] = accuracy_pred_valid
        accuracy_grid = accuracy_pred.reshape(sparsity_grid.shape)
    
    return lon_grid, lat_grid, accuracy_grid


def plot_accuracy_hexbin(ax, fitted_models, model_name, ds_sparsity, sufficiency_value,
                         gridsize=30, cmap='viridis', vmin=None, vmax=None, alpha=1,
                         add_colorbar=True, colorbar_label='Accuracy', threshold=None,
                         levels=None, contour_colors='black', contour_linewidths=1,
                         contour_alpha=1.0, contour_linestyles='solid', show_labels=True, model_type=None):
    """
    Plot accuracy hexbin grid.
    
    Args:
        ax: matplotlib Axes object
        fitted_models: Dictionary of fitted models
        model_name: Model name
        ds_sparsity: xarray Dataset with sparsity values
        sufficiency_value: Fixed sufficiency value
        gridsize: Hexbin grid size
        cmap: Colormap
        vmin, vmax: Colormap range
        alpha: Transparency
        add_colorbar: Whether to add colorbar
        colorbar_label: Colorbar label
        threshold: Values below this are shown in gray
        levels: Contour levels
        contour_colors: Contour line colors
        contour_linewidths: Contour line widths
        contour_alpha: Contour transparency
        contour_linestyles: Contour line styles
        show_labels: Whether to show contour labels
        model_type: Model type
    
    Returns:
        hexbin plot object(s)
    """
    import matplotlib.cm as cm
    
    lon_grid, lat_grid, accuracy_grid = _prepare_accuracy_grid(
        fitted_models, model_name, ds_sparsity, sufficiency_value, model_type=model_type)
    
    lon_points = lon_grid.ravel()
    lat_points = lat_grid.ravel()
    accuracy_points = accuracy_grid.ravel()
    
    valid_mask = ~np.isnan(accuracy_points)
    lon_points = lon_points[valid_mask]
    lat_points = lat_points[valid_mask]
    accuracy_points = accuracy_points[valid_mask]
    
    if len(accuracy_points) == 0:
        print(f"Warning: No valid accuracy values for {model_name}")
        return None
    
    if threshold is None and vmin is not None:
        threshold = vmin
    
    if isinstance(cmap, str):
        cmap_obj = cm.get_cmap(cmap).copy()
    else:
        cmap_obj = cmap.copy()
    cmap_obj.set_under('lightgray')
    
    if threshold is not None:
        accuracy_points_modified = accuracy_points.copy()
        below_threshold_mask = accuracy_points < threshold
        if np.any(below_threshold_mask):
            accuracy_points_modified[below_threshold_mask] = threshold - 1e-10
        accuracy_points = accuracy_points_modified
    
    hexbin_plot = ax.hexbin(lon_points, lat_points, C=accuracy_points,
                           gridsize=gridsize, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                           alpha=alpha, reduce_C_function=np.mean)
    
    contour_plot = None
    if levels is not None:
        if isinstance(contour_linewidths, (list, tuple, np.ndarray)):
            linewidths_to_use = contour_linewidths if len(contour_linewidths) == len(levels) else contour_linewidths[0]
        else:
            linewidths_to_use = contour_linewidths
        
        contour_plot = ax.contour(lon_grid, lat_grid, accuracy_grid,
                                 levels=levels, colors=contour_colors,
                                 linewidths=linewidths_to_use,
                                 alpha=contour_alpha, linestyles=contour_linestyles)
        if show_labels:
            ax.clabel(contour_plot, inline=True, fontsize=8, fmt='%.1f')
    
    if add_colorbar:
        import matplotlib.pyplot as plt
        cbar = plt.colorbar(hexbin_plot, ax=ax, label=colorbar_label, extend='min')
        if contour_plot is not None:
            return hexbin_plot, cbar, contour_plot
        return hexbin_plot, cbar
    
    if contour_plot is not None:
        return hexbin_plot, contour_plot
    return hexbin_plot
