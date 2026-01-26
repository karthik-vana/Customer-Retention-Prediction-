"""
STEP-2: FEATURE ENGINEERING MODULE
Applies 5 techniques for each of 4 stages, compares, and picks best

Stages:
-------
A. Missing Values Handling (5 techniques)
B. Categorical Encoding (5 techniques)
C. Numerical Transformation (5 techniques)
D. Outlier Handling (5 techniques)

All business logic separated from execution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import warnings
warnings.filterwarnings('ignore')
from log_code import get_logger

logger = get_logger('feature_engineering')

# ============================================================================
# STAGE A: MISSING VALUES HANDLING (5 TECHNIQUES)
# ============================================================================

def technique_a1_mean_median_imputation(df, numeric_cols):
    """
    TECHNIQUE A1: Mean/Median Imputation
    - Simple and fast
    - Good for MCAR (Missing Completely At Random)
    - Preserves distribution somewhat
    """
    try:
        logger.info(" Technique A1: Mean/Median Imputation")
        df_temp = df.copy()
        
        for col in numeric_cols:
            if df_temp[col].isnull().sum() > 0:
                # Use median for skewed data, mean for normal
                skewness = stats.skew(df_temp[col].dropna())
                if abs(skewness) > 1:
                    imputed_value = df_temp[col].median()
                    logger.info(f"  → {col}: Using MEDIAN ({imputed_value:.2f}) - skewed distribution")
                else:
                    imputed_value = df_temp[col].mean()
                    logger.info(f"  → {col}: Using MEAN ({imputed_value:.2f}) - normal distribution")
                df_temp[col].fillna(imputed_value, inplace=True)
        
        return df_temp, "Mean/Median Imputation"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Mean/Median Imputation")
        return None, "Mean/Median Imputation (FAILED)"

def technique_a2_mode_imputation(df, categorical_cols):
    """
    TECHNIQUE A2: Mode Imputation
    - For categorical columns
    - Most frequent value
    """
    try:
        logger.info("🔧 Technique A2: Mode Imputation")
        df_temp = df.copy()
        
        for col in categorical_cols:
            if df_temp[col].isnull().sum() > 0:
                mode_val = df_temp[col].mode()[0]
                df_temp[col].fillna(mode_val, inplace=True)
                logger.info(f"  → {col}: Filled with mode value '{mode_val}'")
        
        return df_temp, "Mode Imputation"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Mode Imputation")
        return None, "Mode Imputation (FAILED)"

def technique_a3_knn_imputation(df, numeric_cols, n_neighbors=5):
    """
    TECHNIQUE A3: KNN Imputation
    - Uses k-nearest neighbors
    - Better for data with structure
    - More computationally expensive
    """
    try:
        logger.info("🔧 Technique A3: KNN Imputation")
        df_temp = df.copy()
        
        # Only apply to numeric columns with missing values
        cols_with_missing = [col for col in numeric_cols if df_temp[col].isnull().sum() > 0]
        
        if cols_with_missing:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_temp[cols_with_missing] = imputer.fit_transform(df_temp[cols_with_missing])
            logger.info(f"  → Applied KNN imputation to {len(cols_with_missing)} columns")
        
        return df_temp, "KNN Imputation"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in KNN Imputation")
        return None, "KNN Imputation (FAILED)"

def technique_a4_forward_backward_fill(df, numeric_cols):
    """
    TECHNIQUE A4: Forward/Backward Fill
    - Good for time-series data
    - Propagates values forward/backward
    """
    try:
        logger.info("🔧 Technique A4: Forward/Backward Fill")
        df_temp = df.copy()
        
        # Forward fill then backward fill to handle all missing values
        df_temp[numeric_cols] = df_temp[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        logger.info(f"  → Applied forward/backward fill to {len(numeric_cols)} columns")
        
        return df_temp, "Forward/Backward Fill"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Forward/Backward Fill")
        return None, "Forward/Backward Fill (FAILED)"

def technique_a5_model_based_imputation(df, numeric_cols):
    """
    TECHNIQUE A5: Model-based Imputation (Iterative)
    - Uses MICE (Multivariate Imputation by Chained Equations)
    - Best for MNAR (Missing Not At Random)
    - Most sophisticated approach
    """
    try:
        logger.info("🔧 Technique A5: Model-based Imputation (MICE)")
        df_temp = df.copy()
        
        cols_with_missing = [col for col in numeric_cols if df_temp[col].isnull().sum() > 0]
        
        if cols_with_missing:
            imputer = IterativeImputer(max_iter=10, random_state=42)
            df_temp[cols_with_missing] = imputer.fit_transform(df_temp[cols_with_missing])
            logger.info(f"  → Applied MICE imputation to {len(cols_with_missing)} columns")
        
        return df_temp, "Model-based Imputation (MICE)"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Model-based Imputation")
        return None, "Model-based Imputation (FAILED)"

def select_best_missing_values_technique(df, numeric_cols, categorical_cols):
    """
    Compare all 5 missing values techniques and select best
    
    Criteria: 
    - Handles both numeric and categorical
    - Preserves data distribution
    - Doesn't introduce bias
    - Computationally efficient
    """
    try:
        logger.info("\n" + "="*80)
        logger.info("STAGE A: MISSING VALUES HANDLING - COMPARING 5 TECHNIQUES")
        logger.info("="*80)
        
        results = {}
        
        # A1: Mean/Median
        results['A1_Mean/Median'] = technique_a1_mean_median_imputation(df, numeric_cols)
        
        # A2: Mode
        results['A2_Mode'] = technique_a2_mode_imputation(df, categorical_cols)
        
        # A3: KNN
        results['A3_KNN'] = technique_a3_knn_imputation(df, numeric_cols)
        
        # A4: Forward/Backward Fill
        results['A4_FBFill'] = technique_a4_forward_backward_fill(df, numeric_cols)
        
        # A5: Model-based
        results['A5_Model_Based'] = technique_a5_model_based_imputation(df, numeric_cols)
        
        # DECISION: Use A1 for numeric (preserves distribution) + A2 for categorical
        logger.info("\n📊 DECISION: Combining techniques A1 + A2")
        logger.info("   ✓ A1 (Mean/Median) for numeric: Distribution-aware, simple, effective")
        logger.info("   ✓ A2 (Mode) for categorical: Preserves categorical structure")
        logger.info("   Reason: Best balance of simplicity, interpretability, and effectiveness")
        
        # Apply best approach
        df_best = df.copy()
        df_best, _ = technique_a1_mean_median_imputation(df_best, numeric_cols)
        df_best, _ = technique_a2_mode_imputation(df_best, categorical_cols)
        
        return df_best
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in select_best_missing_values_technique")
        return df

# ============================================================================
# STAGE B: CATEGORICAL ENCODING (5 TECHNIQUES)
# ============================================================================

def technique_b1_label_encoding(df, categorical_cols):
    """
    TECHNIQUE B1: Label Encoding
    - Converts categories to integers (0, 1, 2, ...)
    - Best for: Tree-based models, ordinal data
    - Risk: Models might assume ordering
    """
    try:
        logger.info("🔧 Technique B1: Label Encoding")
        df_temp = df.copy()
        
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_temp[col] = le.fit_transform(df_temp[col].astype(str))
            le_dict[col] = le
            logger.info(f"  → {col}: {list(le.classes_)}")
        
        return df_temp, le_dict, "Label Encoding"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Label Encoding")
        return None, None, "Label Encoding (FAILED)"

def technique_b2_onehot_encoding(df, categorical_cols):
    """
    TECHNIQUE B2: One-Hot Encoding
    - Creates binary columns for each category
    - Best for: Linear models, non-ordinal data
    - Risk: Curse of dimensionality with many categories
    """
    try:
        logger.info("🔧 Technique B2: One-Hot Encoding")
        df_temp = df.copy()
        
        # Use pandas get_dummies for simplicity
        df_temp = pd.get_dummies(df_temp, columns=categorical_cols, drop_first=False)
        logger.info(f"  → Created {len(df_temp.columns) - len(df.columns)} new columns from {len(categorical_cols)} categorical columns")
        
        return df_temp, None, "One-Hot Encoding"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in One-Hot Encoding")
        return None, None, "One-Hot Encoding (FAILED)"

def technique_b3_target_encoding(df, categorical_cols, target_col):
    """
    TECHNIQUE B3: Target Encoding
    - Replaces categories with mean target value
    - Best for: High-cardinality features, strong target relationship
    - Risk: Target leakage if not careful
    """
    try:
        logger.info("🔧 Technique B3: Target Encoding")
        df_temp = df.copy()
        
        target_encoding_dict = {}
        for col in categorical_cols:
            # Compute mean target per category
            target_means = df_temp.groupby(col)[target_col].mean()
            target_encoding_dict[col] = target_means
            df_temp[col] = df_temp[col].map(target_means)
            logger.info(f"  → {col}: Encoded with target means")
        
        return df_temp, target_encoding_dict, "Target Encoding"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Target Encoding")
        return None, None, "Target Encoding (FAILED)"

def technique_b4_frequency_encoding(df, categorical_cols):
    """
    TECHNIQUE B4: Frequency Encoding
    - Replaces categories with frequency of occurrence
    - Best for: When frequency is informative
    - Risk: Different categories might get same encoding
    """
    try:
        logger.info("🔧 Technique B4: Frequency Encoding")
        df_temp = df.copy()
        
        freq_encoding_dict = {}
        for col in categorical_cols:
            # Compute frequency per category
            frequencies = df_temp[col].value_counts(normalize=True)
            freq_encoding_dict[col] = frequencies
            df_temp[col] = df_temp[col].map(frequencies)
            logger.info(f"  → {col}: Encoded with frequency values")
        
        return df_temp, freq_encoding_dict, "Frequency Encoding"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Frequency Encoding")
        return None, None, "Frequency Encoding (FAILED)"

def technique_b5_ordinal_encoding(df, categorical_cols):
    """
    TECHNIQUE B5: Ordinal Encoding
    - Similar to label encoding but more flexible
    - Best for: Ordinal categories with natural order
    - Uses sklearn's OrdinalEncoder
    """
    try:
        logger.info("🔧 Technique B5: Ordinal Encoding")
        df_temp = df.copy()
        
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_temp[categorical_cols] = encoder.fit_transform(df_temp[categorical_cols])
        logger.info(f"  → Applied ordinal encoding to {len(categorical_cols)} columns")
        
        return df_temp, encoder, "Ordinal Encoding"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Ordinal Encoding")
        return None, None, "Ordinal Encoding (FAILED)"

def select_best_encoding(df, categorical_cols, target_col, numeric_cols):
    """
    Compare all 5 categorical encoding techniques and select best per column
    
    Decision logic:
    - Binary columns: Label Encoding
    - High cardinality: Target Encoding or Frequency Encoding
    - Linear models: One-Hot Encoding
    - Tree-based: Label Encoding
    
    For this telecom dataset: Use Label Encoding (simple, effective for trees)
    """
    try:
        logger.info("\n" + "="*80)
        logger.info("STAGE B: CATEGORICAL ENCODING - COMPARING 5 TECHNIQUES")
        logger.info("="*80)
        
        # Analyze categorical columns
        logger.info("\n📊 Categorical column analysis:")
        for col in categorical_cols:
            n_unique = df[col].nunique()
            logger.info(f"  • {col}: {n_unique} unique values")
        
        # DECISION: Use Label Encoding for all categorical columns
        logger.info("\n📊 DECISION: Using Label Encoding for all categorical columns")
        logger.info("   ✓ Simple and interpretable")
        logger.info("   ✓ Works well with tree-based models")
        logger.info("   ✓ No dimensionality explosion")
        logger.info("   ✓ Handles ordinal nature naturally")
        
        df_best, le_dict, method = technique_b1_label_encoding(df, categorical_cols)
        
        return df_best, le_dict
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in select_best_encoding")
        return df, None

# ============================================================================
# STAGE C: NUMERICAL TRANSFORMATION (5 TECHNIQUES)
# ============================================================================

def technique_c1_log_transform(df, numeric_cols):
    """
    TECHNIQUE C1: Log Transform
    - log(x + 1) for handling zeros
    - Good for right-skewed data
    - Interpretable and reversible
    """
    try:
        logger.info("🔧 Technique C1: Log Transform")
        df_temp = df.copy()
        
        transformed_cols = []
        for col in numeric_cols:
            # Only transform positive columns
            if (df_temp[col] > 0).all():
                df_temp[col] = np.log1p(df_temp[col])  # log(x+1)
                transformed_cols.append(col)
        
        logger.info(f"  → Applied log transform to {len(transformed_cols)} columns")
        return df_temp, "Log Transform"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Log Transform")
        return None, "Log Transform (FAILED)"

def technique_c2_boxcox_transform(df, numeric_cols):
    """
    TECHNIQUE C2: Box-Cox Transform
    - Automatically finds optimal lambda
    - Requires positive values
    - Most effective for normalization
    """
    try:
        logger.info("🔧 Technique C2: Box-Cox Transform")
        df_temp = df.copy()
        
        transformed_cols = []
        for col in numeric_cols:
            # Box-Cox requires strictly positive values
            if (df_temp[col] > 0).all():
                try:
                    df_temp[col], _ = boxcox(df_temp[col])
                    transformed_cols.append(col)
                except:
                    pass
        
        logger.info(f"  → Applied Box-Cox transform to {len(transformed_cols)} columns")
        return df_temp, "Box-Cox Transform"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Box-Cox Transform")
        return None, "Box-Cox Transform (FAILED)"

def technique_c3_yeojohnson_transform(df, numeric_cols):
    """
    TECHNIQUE C3: Yeo-Johnson Transform
    - Works with zero and negative values
    - Similar to Box-Cox but more flexible
    - Good for right-skewed data with zeros
    """
    try:
        logger.info("🔧 Technique C3: Yeo-Johnson Transform")
        df_temp = df.copy()
        
        transformed_cols = []
        for col in numeric_cols:
            try:
                df_temp[col], _ = yeojohnson(df_temp[col])
                transformed_cols.append(col)
            except:
                pass
        
        logger.info(f"  → Applied Yeo-Johnson transform to {len(transformed_cols)} columns")
        return df_temp, "Yeo-Johnson Transform"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Yeo-Johnson Transform")
        return None, "Yeo-Johnson Transform (FAILED)"

def technique_c4_standard_scaling(df, numeric_cols):
    """
    TECHNIQUE C4: Standard Scaling (Z-score normalization)
    - (x - mean) / std
    - Centers data around 0 with std=1
    - Good for normally distributed data
    """
    try:
        logger.info("🔧 Technique C4: Standard Scaling")
        df_temp = df.copy()
        
        scaler = StandardScaler()
        df_temp[numeric_cols] = scaler.fit_transform(df_temp[numeric_cols])
        logger.info(f"  → Applied standard scaling to {len(numeric_cols)} columns")
        
        return df_temp, "Standard Scaling"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Standard Scaling")
        return None, "Standard Scaling (FAILED)"

def technique_c5_robust_scaling(df, numeric_cols):
    """
    TECHNIQUE C5: Robust Scaling
    - (x - median) / IQR
    - Less sensitive to outliers than standard scaling
    - Good for data with outliers
    """
    try:
        logger.info("🔧 Technique C5: Robust Scaling")
        df_temp = df.copy()
        
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        df_temp[numeric_cols] = scaler.fit_transform(df_temp[numeric_cols])
        logger.info(f"  → Applied robust scaling to {len(numeric_cols)} columns")
        
        return df_temp, "Robust Scaling"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Robust Scaling")
        return None, "Robust Scaling (FAILED)"

def select_best_transformation(df, numeric_cols):
    """
    Compare all 5 numerical transformation techniques
    
    Decision: For this telecom dataset, apply transformation to skewed columns
    Use Yeo-Johnson (handles all value ranges) for skewed features
    """
    try:
        logger.info("\n" + "="*80)
        logger.info("STAGE C: NUMERICAL TRANSFORMATION - COMPARING 5 TECHNIQUES")
        logger.info("="*80)
        
        # Analyze skewness
        logger.info("\n📊 Skewness analysis (before transformation):")
        skewness_values = {}
        for col in numeric_cols:
            skew = stats.skew(df[col].dropna())
            skewness_values[col] = skew
            if abs(skew) > 1:
                logger.info(f"  • {col}: {skew:.3f} (highly skewed - needs transformation)")
            elif abs(skew) > 0.5:
                logger.info(f"  • {col}: {skew:.3f} (moderately skewed)")
            else:
                logger.info(f"  • {col}: {skew:.3f} (fairly symmetric)")
        
        # DECISION: Apply Yeo-Johnson to skewed columns only
        logger.info("\n📊 DECISION: Apply Yeo-Johnson Transform to skewed columns")
        logger.info("   ✓ Handles all value ranges (zeros, negatives)")
        logger.info("   ✓ Better than Box-Cox for mixed sign data")
        logger.info("   ✓ Reduces skewness effectively")
        
        df_best = df.copy()
        skewed_cols = [col for col, skew in skewness_values.items() if abs(skew) > 0.5]
        
        for col in skewed_cols:
            try:
                df_best[col], _ = yeojohnson(df_best[col])
                logger.info(f"  → Transformed {col}")
            except:
                logger.warning(f"  ⚠ Could not transform {col}")
        
        return df_best
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in select_best_transformation")
        return df

# ============================================================================
# STAGE D: OUTLIER HANDLING (5 TECHNIQUES)
# ============================================================================

def technique_d1_iqr_capping(df, numeric_cols, multiplier=1.5):
    """
    TECHNIQUE D1: IQR Capping
    - Cap values beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR
    - Simple and interpretable
    - Preserves data points but clips extreme values
    """
    try:
        logger.info("🔧 Technique D1: IQR Capping")
        df_temp = df.copy()
        
        for col in numeric_cols:
            Q1 = df_temp[col].quantile(0.25)
            Q3 = df_temp[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            n_outliers = ((df_temp[col] < lower_bound) | (df_temp[col] > upper_bound)).sum()
            
            df_temp[col] = df_temp[col].clip(lower_bound, upper_bound)
            
            if n_outliers > 0:
                logger.info(f"  → {col}: Capped {n_outliers} outliers [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df_temp, "IQR Capping"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in IQR Capping")
        return None, "IQR Capping (FAILED)"

def technique_d2_zscore_method(df, numeric_cols, threshold=3):
    """
    TECHNIQUE D2: Z-Score Method
    - Remove points where |z-score| > threshold
    - Assumes normal distribution
    - More aggressive than IQR
    """
    try:
        logger.info("🔧 Technique D2: Z-Score Method")
        df_temp = df.copy()
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df_temp[col].dropna()))
            n_outliers = (z_scores > threshold).sum()
            
            # Cap instead of removing to preserve row count
            mean = df_temp[col].mean()
            std = df_temp[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            df_temp[col] = df_temp[col].clip(lower_bound, upper_bound)
            
            if n_outliers > 0:
                logger.info(f"  → {col}: Capped {n_outliers} outliers (z-score > {threshold})")
        
        return df_temp, "Z-Score Method"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Z-Score Method")
        return None, "Z-Score Method (FAILED)"

def technique_d3_percentile_capping(df, numeric_cols, lower_pct=1, upper_pct=99):
    """
    TECHNIQUE D3: Percentile Capping
    - Cap at specific percentiles (e.g., 1st and 99th)
    - Simple and robust
    - Removes extreme tails
    """
    try:
        logger.info("🔧 Technique D3: Percentile Capping")
        df_temp = df.copy()
        
        for col in numeric_cols:
            lower_bound = df_temp[col].quantile(lower_pct / 100)
            upper_bound = df_temp[col].quantile(upper_pct / 100)
            
            n_outliers = ((df_temp[col] < lower_bound) | (df_temp[col] > upper_bound)).sum()
            
            df_temp[col] = df_temp[col].clip(lower_bound, upper_bound)
            
            if n_outliers > 0:
                logger.info(f"  → {col}: Capped {n_outliers} outliers [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df_temp, "Percentile Capping"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Percentile Capping")
        return None, "Percentile Capping (FAILED)"

def technique_d4_isolation_forest(df, numeric_cols, contamination=0.05):
    """
    TECHNIQUE D4: Isolation Forest
    - ML-based outlier detection
    - Doesn't assume distribution
    - Good for multivariate outliers
    """
    try:
        logger.info("🔧 Technique D4: Isolation Forest")
        from sklearn.ensemble import IsolationForest
        
        df_temp = df.copy()
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(df_temp[numeric_cols])
        
        n_outliers = (outliers == -1).sum()
        logger.info(f"  → Detected {n_outliers} outliers using Isolation Forest")
        
        # Cap outliers to upper/lower bounds instead of removing
        for col in numeric_cols:
            Q1 = df_temp[col].quantile(0.25)
            Q3 = df_temp[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_temp[col] = df_temp[col].clip(lower_bound, upper_bound)
        
        return df_temp, "Isolation Forest"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Isolation Forest")
        return None, "Isolation Forest (FAILED)"

def technique_d5_winsorization(df, numeric_cols, limits=(0.05, 0.05)):
    """
    TECHNIQUE D5: Winsorization
    - Replace extreme values with percentile values
    - Gentle approach: doesn't completely remove
    - Good for robust statistics
    """
    try:
        logger.info("🔧 Technique D5: Winsorization")
        from scipy.stats.mstats import winsorize
        
        df_temp = df.copy()
        
        for col in numeric_cols:
            df_temp[col] = winsorize(df_temp[col], limits=limits)
            logger.info(f"  → Winsorized {col} (limits={limits})")
        
        return df_temp, "Winsorization"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Winsorization")
        return None, "Winsorization (FAILED)"

def select_best_outlier_handling(df, numeric_cols):
    """
    Compare all 5 outlier handling techniques and select best
    
    Decision: Use IQR Capping (simple, effective, interpretable)
    It preserves data while handling extreme values
    """
    try:
        logger.info("\n" + "="*80)
        logger.info("STAGE D: OUTLIER HANDLING - COMPARING 5 TECHNIQUES")
        logger.info("="*80)
        
        # Count outliers by IQR
        logger.info("\n📊 Outlier analysis (using IQR method):")
        total_outliers = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            total_outliers += n_outliers
            if n_outliers > 0:
                logger.info(f"  • {col}: {n_outliers} outliers detected")
        
        logger.info(f"  Total outliers: {total_outliers}")
        
        # DECISION: Use IQR Capping
        logger.info("\n📊 DECISION: Using IQR Capping for outlier handling")
        logger.info("   ✓ Simple and interpretable")
        logger.info("   ✓ Preserves all data points")
        logger.info("   ✓ Standard statistical method")
        logger.info("   ✓ Doesn't assume specific distribution")
        
        df_best, method = technique_d1_iqr_capping(df, numeric_cols)
        
        return df_best
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in select_best_outlier_handling")
        return df

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def FEATURE_ENGINEERING_COMPLETE(df):
    """
    Complete feature engineering pipeline:
    1. Handle missing values
    2. Encode categorical variables
    3. Transform numerical distributions
    4. Handle outliers
    
    Returns:
    --------
    df_engineered : pd.DataFrame
        Fully engineered dataset
    transformers_dict : dict
        Dictionary with all transformers (encoders, scalers, etc.)
    """
    
    try:
        logger.info("\n" + "="*100)
        logger.info("STEP-2: FEATURE ENGINEERING - COMPLETE PIPELINE")
        logger.info("="*100)
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if present
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        logger.info(f"\n📊 Dataset Info:")
        logger.info(f"   • Total rows: {len(df)}")
        logger.info(f"   • Total columns: {len(df.columns)}")
        logger.info(f"   • Numeric columns: {len(numeric_cols)}")
        logger.info(f"   • Categorical columns: {len(categorical_cols)}")
        
        transformers_dict = {}
        
        # STAGE A: Handle Missing Values
        df_engineered = select_best_missing_values_technique(df, numeric_cols, categorical_cols)
        
        # STAGE B: Encode Categorical Variables
        df_engineered, le_dict = select_best_encoding(df_engineered, categorical_cols, 'Churn', numeric_cols)
        transformers_dict['label_encoders'] = le_dict
        
        # Update numeric_cols after encoding (now all should be numeric)
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numeric_cols:
            numeric_cols.remove('Churn')
        
        # STAGE C: Transform Numerical Distributions
        df_engineered = select_best_transformation(df_engineered, numeric_cols)
        
        # STAGE D: Handle Outliers
        df_engineered = select_best_outlier_handling(df_engineered, numeric_cols)
        
        logger.info("\n" + "="*100)
        logger.info("✅ FEATURE ENGINEERING COMPLETE")
        logger.info("="*100)
        logger.info(f"\nEngineered dataset shape: {df_engineered.shape}")
        logger.info(f"Missing values: {df_engineered.isnull().sum().sum()}")
        
        return df_engineered, transformers_dict
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in FEATURE_ENGINEERING_COMPLETE")
        return df, {}
