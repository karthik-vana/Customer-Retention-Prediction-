"""
STEP-3: FEATURE SELECTION MODULE
Applies 5 techniques in EACH of 3 categories, compares, and picks best

Categories:
-----------
1. Filter Methods (5 techniques)
2. Correlation Based (5 techniques)
3. Hypothesis Testing (5 techniques)

All business logic separated from execution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import pointbiserialr, chi2_contingency, mannwhitneyu, ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')
from log_code import get_logger

logger = get_logger('feature_selection')

# ============================================================================
# CATEGORY 1: FILTER METHODS (5 TECHNIQUES)
# ============================================================================

def filter_1_variance_threshold(df, threshold=0.01):
    """
    FILTER 1: Variance Threshold
    - Remove features with low variance
    - Assumption: High variance = informative
    """
    try:
        logger.info("🔍 Filter Method 1: Variance Threshold")
        from sklearn.feature_selection import VarianceThreshold
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        selector = VarianceThreshold(threshold=threshold)
        selected = selector.fit_transform(numeric_df)
        selected_cols = numeric_df.columns[selector.get_support()].tolist()
        removed_cols = numeric_df.columns[~selector.get_support()].tolist()
        
        logger.info(f"  → Selected {len(selected_cols)} features (removed {len(removed_cols)} low-variance)")
        if removed_cols:
            logger.info(f"    Removed: {removed_cols}")
        
        return selected_cols, "Variance Threshold"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Variance Threshold")
        return df.select_dtypes(include=[np.number]).columns.tolist(), "Variance Threshold (FAILED)"

def filter_2_chi_square(df, target_col, k=10):
    """
    FILTER 2: Chi-Square Test
    - For categorical features vs categorical target
    - Measures dependence
    """
    try:
        logger.info("🔍 Filter Method 2: Chi-Square Test")
        
        # Use only categorical features
        cat_df = df.select_dtypes(include=['object']).copy()
        if target_col in cat_df.columns:
            cat_df = cat_df.drop(target_col, axis=1)
        
        if len(cat_df.columns) == 0:
            logger.warning("  → No categorical features for chi-square test")
            return [], "Chi-Square (No categorical features)"
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        scores = {}
        for col in cat_df.columns:
            if col != target_col:
                # Encode target
                target_encoded = le.fit_transform(df[target_col])
                feature_encoded = le.fit_transform(cat_df[col])
                
                # Chi-square test
                chi2_stat, p_value = chi2_contingency(pd.crosstab(feature_encoded, target_encoded))[:2]
                scores[col] = chi2_stat
        
        # Sort by score
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_cols = [f[0] for f in sorted_features[:min(k, len(sorted_features))]]
        
        logger.info(f"  → Selected {len(selected_cols)} categorical features (chi-square)")
        
        return selected_cols, "Chi-Square Test"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Chi-Square Test")
        return [], "Chi-Square (FAILED)"

def filter_3_mutual_information(df, target_col, k=10):
    """
    FILTER 3: Mutual Information
    - Measures dependence without assuming linearity
    - Works with both numeric and categorical
    """
    try:
        logger.info("🔍 Filter Method 3: Mutual Information")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        if len(numeric_df.columns) == 0:
            logger.warning("  → No numeric features for MI calculation")
            return [], "Mutual Information (No numeric features)"
        
        # Convert target to numeric
        le = LabelEncoder()
        target_encoded = le.fit_transform(df[target_col])
        
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(numeric_df.columns)))
        selector.fit(numeric_df, target_encoded)
        selected_cols = numeric_df.columns[selector.get_support()].tolist()
        
        logger.info(f"  → Selected {len(selected_cols)} features (mutual information)")
        
        return selected_cols, "Mutual Information"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Mutual Information")
        return df.select_dtypes(include=[np.number]).columns.tolist(), "MI (FAILED)"

def filter_4_anova_f_test(df, target_col, k=10):
    """
    FILTER 4: ANOVA F-test
    - For numeric features vs categorical target
    - Tests if group means are significantly different
    """
    try:
        logger.info("🔍 Filter Method 4: ANOVA F-test")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        if len(numeric_df.columns) == 0:
            logger.warning("  → No numeric features for ANOVA")
            return [], "ANOVA (No numeric features)"
        
        le = LabelEncoder()
        target_encoded = le.fit_transform(df[target_col])
        
        selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_df.columns)))
        selector.fit(numeric_df, target_encoded)
        selected_cols = numeric_df.columns[selector.get_support()].tolist()
        
        logger.info(f"  → Selected {len(selected_cols)} features (ANOVA F-test)")
        
        return selected_cols, "ANOVA F-test"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in ANOVA F-test")
        return df.select_dtypes(include=[np.number]).columns.tolist(), "ANOVA (FAILED)"

def filter_5_information_gain(df, target_col, k=10):
    """
    FILTER 5: Information Gain
    - Uses tree-based feature importances
    - Like mutual information but from tree perspective
    """
    try:
        logger.info("🔍 Filter Method 5: Information Gain (Tree-based)")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        if len(numeric_df.columns) == 0:
            logger.warning("  → No numeric features for information gain")
            return [], "Information Gain (No features)"
        
        le = LabelEncoder()
        target_encoded = le.fit_transform(df[target_col])
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(numeric_df, target_encoded)
        
        feature_importance = pd.DataFrame({
            'feature': numeric_df.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_cols = feature_importance.head(min(k, len(numeric_df.columns)))['feature'].tolist()
        
        logger.info(f"  → Selected {len(selected_cols)} features (information gain)")
        for feat, imp in zip(selected_cols[:5], feature_importance.head(5)['importance']):
            logger.info(f"    • {feat}: {imp:.4f}")
        
        return selected_cols, "Information Gain"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Information Gain")
        return df.select_dtypes(include=[np.number]).columns.tolist(), "Info Gain (FAILED)"

# ============================================================================
# CATEGORY 2: CORRELATION BASED (5 TECHNIQUES)
# ============================================================================

def correlation_1_pearson(df, target_col, threshold=0.1):
    """
    CORRELATION 1: Pearson Correlation
    - Linear correlation coefficient
    - Range: -1 to 1
    """
    try:
        logger.info("🔗 Correlation Method 1: Pearson Correlation")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        # Convert target to numeric for correlation
        le = LabelEncoder()
        target_numeric = le.fit_transform(df[target_col])
        
        correlations = {}
        for col in numeric_df.columns:
            corr = numeric_df[col].corr(pd.Series(target_numeric))
            correlations[col] = abs(corr)
        
        selected_cols = [col for col, corr in correlations.items() if corr >= threshold]
        selected_cols = sorted(selected_cols, key=lambda x: correlations[x], reverse=True)
        
        logger.info(f"  → Selected {len(selected_cols)} features (Pearson corr > {threshold})")
        
        return selected_cols, "Pearson Correlation"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Pearson Correlation")
        return [], "Pearson (FAILED)"

def correlation_2_spearman(df, target_col, threshold=0.1):
    """
    CORRELATION 2: Spearman Correlation
    - Rank-based correlation (monotonic relationships)
    - More robust to outliers than Pearson
    """
    try:
        logger.info("🔗 Correlation Method 2: Spearman Correlation")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        le = LabelEncoder()
        target_numeric = le.fit_transform(df[target_col])
        
        correlations = {}
        for col in numeric_df.columns:
            corr, _ = stats.spearmanr(numeric_df[col], target_numeric)
            correlations[col] = abs(corr)
        
        selected_cols = [col for col, corr in correlations.items() if corr >= threshold]
        selected_cols = sorted(selected_cols, key=lambda x: correlations[x], reverse=True)
        
        logger.info(f"  → Selected {len(selected_cols)} features (Spearman corr > {threshold})")
        
        return selected_cols, "Spearman Correlation"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Spearman Correlation")
        return [], "Spearman (FAILED)"

def correlation_3_kendall(df, target_col, threshold=0.1):
    """
    CORRELATION 3: Kendall Correlation
    - Tau statistic
    - Measures ordinal association
    - More computationally expensive but robust
    """
    try:
        logger.info("🔗 Correlation Method 3: Kendall Correlation")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        le = LabelEncoder()
        target_numeric = le.fit_transform(df[target_col])
        
        correlations = {}
        for col in numeric_df.columns:
            corr, _ = stats.kendalltau(numeric_df[col], target_numeric)
            correlations[col] = abs(corr)
        
        selected_cols = [col for col, corr in correlations.items() if corr >= threshold]
        selected_cols = sorted(selected_cols, key=lambda x: correlations[x], reverse=True)
        
        logger.info(f"  → Selected {len(selected_cols)} features (Kendall corr > {threshold})")
        
        return selected_cols, "Kendall Correlation"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Kendall Correlation")
        return [], "Kendall (FAILED)"

def correlation_4_vif(df, threshold=5):
    """
    CORRELATION 4: VIF (Variance Inflation Factor)
    - Multicollinearity detection
    - VIF > 5 indicates high multicollinearity
    - Removes features that are highly correlated with others
    """
    try:
        logger.info("🔗 Correlation Method 4: VIF (Variance Inflation Factor)")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        vif_data = pd.DataFrame()
        vif_data['feature'] = numeric_df.columns
        vif_data['VIF'] = [variance_inflation_factor(numeric_df.values, i) for i in range(numeric_df.shape[1])]
        
        # Select features with VIF < threshold
        selected_cols = vif_data[vif_data['VIF'] < threshold]['feature'].tolist()
        
        logger.info(f"  → Selected {len(selected_cols)} features (VIF < {threshold})")
        high_vif = vif_data[vif_data['VIF'] >= threshold]
        if len(high_vif) > 0:
            logger.info(f"    Removed {len(high_vif)} features with high multicollinearity")
        
        return selected_cols, "VIF"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in VIF")
        return df.select_dtypes(include=[np.number]).columns.tolist(), "VIF (FAILED)"

def correlation_5_heatmap_threshold(df, target_col, threshold=0.15):
    """
    CORRELATION 5: Heatmap Threshold
    - Visual correlation analysis
    - Remove features with low correlation to target
    - Also removes highly correlated feature pairs
    """
    try:
        logger.info("🔗 Correlation Method 5: Heatmap Threshold")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        le = LabelEncoder()
        target_numeric = le.fit_transform(df[target_col])
        numeric_df['__target__'] = target_numeric
        
        corr_matrix = numeric_df.corr()
        target_corr = corr_matrix['__target__'].drop('__target__')
        
        # Select features with correlation to target > threshold
        selected_cols = [col for col, corr in target_corr.items() if abs(corr) >= threshold]
        selected_cols = sorted(selected_cols, key=lambda x: abs(target_corr[x]), reverse=True)
        
        logger.info(f"  → Selected {len(selected_cols)} features (|correlation| > {threshold})")
        
        return selected_cols, "Heatmap Threshold"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Heatmap Threshold")
        return [], "Heatmap (FAILED)"

# ============================================================================
# CATEGORY 3: HYPOTHESIS TESTING (5 TECHNIQUES)
# ============================================================================

def hypothesis_1_t_test(df, target_col, threshold=0.05):
    """
    HYPOTHESIS 1: T-Test
    - For numeric features vs binary target
    - H0: Two groups have same mean
    - Selects features where groups differ significantly
    """
    try:
        logger.info("📊 Hypothesis Test 1: T-Test")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        # Get target groups
        target_values = df[target_col].unique()
        if len(target_values) != 2:
            logger.warning(f"  → T-test requires binary target. Found {len(target_values)} classes")
            return [], "T-Test (Non-binary target)"
        
        le = LabelEncoder()
        target_encoded = le.fit_transform(df[target_col])
        
        group0 = numeric_df[target_encoded == 0]
        group1 = numeric_df[target_encoded == 1]
        
        selected_cols = []
        for col in numeric_df.columns:
            t_stat, p_value = stats.ttest_ind(group0[col].dropna(), group1[col].dropna())
            if p_value < threshold:
                selected_cols.append(col)
        
        logger.info(f"  → Selected {len(selected_cols)} features (t-test p < {threshold})")
        
        return selected_cols, "T-Test"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in T-Test")
        return [], "T-Test (FAILED)"

def hypothesis_2_chi_square_test(df, target_col, threshold=0.05):
    """
    HYPOTHESIS 2: Chi-Square Test
    - For categorical features vs categorical target
    - H0: Variables are independent
    - Selects dependent features
    """
    try:
        logger.info("📊 Hypothesis Test 2: Chi-Square Test")
        
        cat_df = df.select_dtypes(include=['object']).copy()
        if target_col in cat_df.columns:
            cat_df = cat_df.drop(target_col, axis=1)
        
        if len(cat_df.columns) == 0:
            logger.warning("  → No categorical features for chi-square")
            return [], "Chi-Square (No categorical)"
        
        le = LabelEncoder()
        target_encoded = le.fit_transform(df[target_col])
        
        selected_cols = []
        for col in cat_df.columns:
            feature_encoded = le.fit_transform(cat_df[col])
            chi2_stat, p_value = chi2_contingency(pd.crosstab(feature_encoded, target_encoded))[:2]
            if p_value < threshold:
                selected_cols.append(col)
        
        logger.info(f"  → Selected {len(selected_cols)} categorical features (chi-square p < {threshold})")
        
        return selected_cols, "Chi-Square Test"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Chi-Square Test")
        return [], "Chi-Square Test (FAILED)"

def hypothesis_3_mannwhitney_u(df, target_col, threshold=0.05):
    """
    HYPOTHESIS 3: Mann-Whitney U Test
    - Non-parametric alternative to t-test
    - Tests if distributions differ
    - Doesn't assume normality
    """
    try:
        logger.info("📊 Hypothesis Test 3: Mann-Whitney U Test")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        target_values = df[target_col].unique()
        if len(target_values) != 2:
            logger.warning(f"  → Mann-Whitney requires binary target. Found {len(target_values)} classes")
            return [], "Mann-Whitney (Non-binary target)"
        
        le = LabelEncoder()
        target_encoded = le.fit_transform(df[target_col])
        
        group0 = numeric_df[target_encoded == 0]
        group1 = numeric_df[target_encoded == 1]
        
        selected_cols = []
        for col in numeric_df.columns:
            u_stat, p_value = mannwhitneyu(group0[col].dropna(), group1[col].dropna())
            if p_value < threshold:
                selected_cols.append(col)
        
        logger.info(f"  → Selected {len(selected_cols)} features (Mann-Whitney p < {threshold})")
        
        return selected_cols, "Mann-Whitney U"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Mann-Whitney U")
        return [], "Mann-Whitney (FAILED)"

def hypothesis_4_ks_test(df, target_col, threshold=0.05):
    """
    HYPOTHESIS 4: Kolmogorov-Smirnov Test
    - Tests if two distributions are different
    - Non-parametric
    - For continuous data
    """
    try:
        logger.info("📊 Hypothesis Test 4: Kolmogorov-Smirnov Test")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        target_values = df[target_col].unique()
        if len(target_values) != 2:
            logger.warning(f"  → KS test requires binary target. Found {len(target_values)} classes")
            return [], "KS Test (Non-binary)"
        
        le = LabelEncoder()
        target_encoded = le.fit_transform(df[target_col])
        
        group0 = numeric_df[target_encoded == 0]
        group1 = numeric_df[target_encoded == 1]
        
        selected_cols = []
        for col in numeric_df.columns:
            ks_stat, p_value = ks_2samp(group0[col].dropna(), group1[col].dropna())
            if p_value < threshold:
                selected_cols.append(col)
        
        logger.info(f"  → Selected {len(selected_cols)} features (KS test p < {threshold})")
        
        return selected_cols, "KS Test"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in KS Test")
        return [], "KS Test (FAILED)"

def hypothesis_5_point_biserial(df, target_col, threshold=0.1):
    """
    HYPOTHESIS 5: Point-Biserial Correlation
    - For numeric feature vs binary target
    - Correlation coefficient for continuous-binary relationship
    """
    try:
        logger.info("📊 Hypothesis Test 5: Point-Biserial Correlation")
        
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if 'Churn' in numeric_df.columns:
            numeric_df = numeric_df.drop('Churn', axis=1)
        
        target_values = df[target_col].unique()
        if len(target_values) != 2:
            logger.warning(f"  → Point-Biserial requires binary target. Found {len(target_values)} classes")
            return [], "Point-Biserial (Non-binary)"
        
        le = LabelEncoder()
        target_encoded = le.fit_transform(df[target_col])
        
        selected_cols = []
        for col in numeric_df.columns:
            corr, p_value = pointbiserialr(target_encoded, numeric_df[col])
            if p_value < 0.05 and abs(corr) >= threshold:
                selected_cols.append(col)
        
        logger.info(f"  → Selected {len(selected_cols)} features (Point-Biserial |r| > {threshold})")
        
        return selected_cols, "Point-Biserial"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Point-Biserial")
        return [], "Point-Biserial (FAILED)"

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def FEATURE_SELECTION_COMPLETE(df, target_col='Churn'):
    """
    Complete feature selection pipeline:
    1. Filter Methods (5 techniques)
    2. Correlation Based (5 techniques)
    3. Hypothesis Testing (5 techniques)
    
    Select best features using consensus approach
    
    Returns:
    --------
    final_features : list
        List of selected feature columns
    selection_report : dict
        Report of all methods and their selections
    """
    
    try:
        logger.info("\n" + "="*100)
        logger.info("STEP-3: FEATURE SELECTION - COMPLETE PIPELINE")
        logger.info("="*100)
        
        logger.info(f"\n📊 Dataset Info:")
        logger.info(f"   • Total rows: {len(df)}")
        logger.info(f"   • Total columns: {len(df.columns)}")
        logger.info(f"   • Target: {target_col}")
        
        selection_report = {}
        all_selected_features = []
        
        # =====================================================================
        # CATEGORY 1: FILTER METHODS
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("CATEGORY 1: FILTER METHODS (5 TECHNIQUES)")
        logger.info("="*80)
        
        filter_results = {}
        
        features, method = filter_1_variance_threshold(df)
        filter_results['Variance Threshold'] = features
        all_selected_features.extend(features)
        
        features, method = filter_2_chi_square(df, target_col)
        filter_results['Chi-Square'] = features
        all_selected_features.extend(features)
        
        features, method = filter_3_mutual_information(df, target_col)
        filter_results['Mutual Information'] = features
        all_selected_features.extend(features)
        
        features, method = filter_4_anova_f_test(df, target_col)
        filter_results['ANOVA F-test'] = features
        all_selected_features.extend(features)
        
        features, method = filter_5_information_gain(df, target_col)
        filter_results['Information Gain'] = features
        all_selected_features.extend(features)
        
        selection_report['Filter Methods'] = filter_results
        
        # =====================================================================
        # CATEGORY 2: CORRELATION BASED
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("CATEGORY 2: CORRELATION BASED (5 TECHNIQUES)")
        logger.info("="*80)
        
        correlation_results = {}
        
        features, method = correlation_1_pearson(df, target_col)
        correlation_results['Pearson'] = features
        all_selected_features.extend(features)
        
        features, method = correlation_2_spearman(df, target_col)
        correlation_results['Spearman'] = features
        all_selected_features.extend(features)
        
        features, method = correlation_3_kendall(df, target_col)
        correlation_results['Kendall'] = features
        all_selected_features.extend(features)
        
        features, method = correlation_4_vif(df)
        correlation_results['VIF'] = features
        all_selected_features.extend(features)
        
        features, method = correlation_5_heatmap_threshold(df, target_col)
        correlation_results['Heatmap'] = features
        all_selected_features.extend(features)
        
        selection_report['Correlation Based'] = correlation_results
        
        # =====================================================================
        # CATEGORY 3: HYPOTHESIS TESTING
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("CATEGORY 3: HYPOTHESIS TESTING (5 TECHNIQUES)")
        logger.info("="*80)
        
        hypothesis_results = {}
        
        features, method = hypothesis_1_t_test(df, target_col)
        hypothesis_results['T-Test'] = features
        all_selected_features.extend(features)
        
        features, method = hypothesis_2_chi_square_test(df, target_col)
        hypothesis_results['Chi-Square Test'] = features
        all_selected_features.extend(features)
        
        features, method = hypothesis_3_mannwhitney_u(df, target_col)
        hypothesis_results['Mann-Whitney U'] = features
        all_selected_features.extend(features)
        
        features, method = hypothesis_4_ks_test(df, target_col)
        hypothesis_results['KS Test'] = features
        all_selected_features.extend(features)
        
        features, method = hypothesis_5_point_biserial(df, target_col)
        hypothesis_results['Point-Biserial'] = features
        all_selected_features.extend(features)
        
        selection_report['Hypothesis Testing'] = hypothesis_results
        
        # =====================================================================
        # CONSENSUS-BASED FINAL SELECTION
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("CONSENSUS-BASED FINAL SELECTION")
        logger.info("="*80)
        
        # Count frequency of each feature across all methods
        from collections import Counter
        feature_counts = Counter(all_selected_features)
        
        logger.info(f"\n📊 Feature frequency across {len(all_selected_features)} selections:")
        logger.info(f"   Total unique features proposed: {len(feature_counts)}")
        
        # Select features that appear in at least 3 methods (consensus)
        min_consensus = max(3, len(all_selected_features) // 20)  # At least 3 methods or 5% of all selections
        final_features = sorted([
            feat for feat, count in feature_counts.items() if count >= min_consensus
        ], key=lambda x: feature_counts[x], reverse=True)
        
        logger.info(f"\n✅ DECISION: Using consensus approach (min {min_consensus} votes)")
        logger.info(f"   Selected {len(final_features)} features from {len(feature_counts)} proposed")
        logger.info(f"\nFinal Selected Features:")
        for i, feat in enumerate(final_features, 1):
            logger.info(f"   {i}. {feat} (selected by {feature_counts[feat]} methods)")
        
        # Always include target column
        if target_col not in final_features:
            final_features.append(target_col)
        
        logger.info("\n" + "="*100)
        logger.info("✅ FEATURE SELECTION COMPLETE")
        logger.info("="*100)
        logger.info(f"\nFinal dataset will have {len(final_features)} features")
        
        return final_features, selection_report
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in FEATURE_SELECTION_COMPLETE")
        # Return all numeric columns as fallback
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col not in numeric_cols:
            numeric_cols.append(target_col)
        return numeric_cols, {}
