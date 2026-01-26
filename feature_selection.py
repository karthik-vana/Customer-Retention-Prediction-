"""
feature_selection.py - Feature Selection Module
========================================
Beginner-friendly feature selection using multiple methods

Functions:
- FEATURE_SELECTION_PROCESS: Main feature selection pipeline
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif, SelectKBest
import warnings
warnings.filterwarnings('ignore')

class FEATURE_SELECTION_PROCESS:
    """
    Feature selection using multiple methods
    
    Methods:
    - Mutual Information
    - Chi-Square Test
    - ANOVA F-test
    - Correlation Analysis
    """
    
    def __init__(self, X, y):
        """
        Initialize with features and target
        
        Args:
            X: Feature matrix (dataframe)
            y: Target variable (series or array)
        """
        self.X = X.copy()
        self.y = y.copy()
        self.selected_features = None
        self.feature_scores = None
        print("\n[FEATURE SELECTION INITIALIZED]")
    
    def mutual_information_selection(self, k=10):
        """
        Select features using Mutual Information
        
        What it does:
        - Measures how much each feature tells us about the target
        - Higher score = feature is more informative
        
        Args:
            k: Number of top features to select
            
        Returns:
            list: Top k feature names
        """
        print("\n[Method 1: Mutual Information Selection]")
        
        try:
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
            mi_df = pd.DataFrame({
                'Feature': self.X.columns,
                'MI_Score': mi_scores
            }).sort_values('MI_Score', ascending=False)
            
            top_features = mi_df.head(k)['Feature'].tolist()
            
            print(f"  [OK] Top {k} features by Mutual Information:")
            for i, (feat, score) in enumerate(zip(top_features, mi_df.head(k)['MI_Score'].values), 1):
                print(f"    {i}. {feat}: {score:.4f}")
            
            return top_features
        
        except Exception as e:
            print(f"  [FAILED] Mutual Information failed: {str(e)}")
            return None
    
    def chi_square_selection(self, k=10):
        """
        Select features using Chi-Square test
        
        What it does:
        - Tests if each feature is independent from target
        - Higher score = feature is dependent on target (useful)
        
        Args:
            k: Number of top features to select
            
        Returns:
            list: Top k feature names
        """
        print("\n[Method 2: Chi-Square Test Selection]")
        
        try:
            # Chi-square requires non-negative values
            X_chi = self.X.copy()
            
            # Shift negative values to make them non-negative
            for col in X_chi.columns:
                if (X_chi[col] < 0).any():
                    min_val = X_chi[col].min()
                    X_chi[col] = X_chi[col] - min_val
            
            # Calculate chi-square scores
            chi_scores = chi2(X_chi, self.y)[0]
            chi_df = pd.DataFrame({
                'Feature': self.X.columns,
                'Chi2_Score': chi_scores
            }).sort_values('Chi2_Score', ascending=False)
            
            top_features = chi_df.head(k)['Feature'].tolist()
            
            print(f"  [OK] Top {k} features by Chi-Square:")
            for i, (feat, score) in enumerate(zip(top_features, chi_df.head(k)['Chi2_Score'].values), 1):
                print(f"    {i}. {feat}: {score:.4f}")
            
            return top_features
        
        except Exception as e:
            print(f"  [FAILED] Chi-Square failed: {str(e)}")
            return None
    
    def anova_f_test_selection(self, k=10):
        """
        Select features using ANOVA F-test
        
        What it does:
        - Tests if feature means differ between classes
        - Higher score = feature is good at separating classes
        
        Args:
            k: Number of top features to select
            
        Returns:
            list: Top k feature names
        """
        print("\n[Method 3: ANOVA F-Test Selection]")
        
        try:
            # Calculate F-scores
            f_scores = f_classif(self.X, self.y)[0]
            f_df = pd.DataFrame({
                'Feature': self.X.columns,
                'F_Score': f_scores
            }).sort_values('F_Score', ascending=False)
            
            top_features = f_df.head(k)['Feature'].tolist()
            
            print(f"  [OK] Top {k} features by ANOVA F-Test:")
            for i, (feat, score) in enumerate(zip(top_features, f_df.head(k)['F_Score'].values), 1):
                print(f"    {i}. {feat}: {score:.4f}")
            
            return top_features
        
        except Exception as e:
            print(f"  [FAILED] ANOVA F-Test failed: {str(e)}")
            return None
    
    def correlation_based_selection(self, threshold=0.05):
        """
        Select features based on correlation with target
        
        What it does:
        - Calculates correlation between each feature and target
        - Keeps features with correlation above threshold
        
        Args:
            threshold: Minimum correlation to keep feature
            
        Returns:
            list: Selected feature names
        """
        print("\n[Method 4: Correlation-Based Selection]")
        
        try:
            # Create dataframe with features and target
            df = self.X.copy()
            df['Target'] = self.y
            
            # Calculate correlation with target
            correlations = df.corr()['Target'].abs().sort_values(ascending=False)
            
            # Remove Target itself
            correlations = correlations[correlations.index != 'Target']
            
            # Select features above threshold
            selected = correlations[correlations > threshold].index.tolist()
            
            print(f"  [OK] Selected {len(selected)} features with correlation > {threshold}:")
            for feat in selected[:10]:
                print(f"    - {feat}: {correlations[feat]:.4f}")
            
            if len(selected) > 10:
                print(f"    ... and {len(selected) - 10} more")
            
            return selected
        
        except Exception as e:
            print(f"  [FAILED] Correlation failed: {str(e)}")
            return None
    
    def ensemble_selection(self, k=10):
        """
        Combine multiple selection methods for robust feature selection
        
        What it does:
        - Uses all 4 methods above
        - Votes on which features to keep
        - More robust than single method
        
        Args:
            k: Number of top features to select
            
        Returns:
            list: Final selected feature names
        """
        print("\n[Ensemble Feature Selection (Voting Method)]")
        
        # Get selections from each method
        mi_features = self.mutual_information_selection(k)
        chi_features = self.chi_square_selection(k)
        anova_features = self.anova_f_test_selection(k)
        corr_features = self.correlation_based_selection(0.05)
        
        # Vote for features (appear in multiple methods)
        all_votes = []
        
        if mi_features:
            all_votes.extend(mi_features)
        if chi_features:
            all_votes.extend(chi_features)
        if anova_features:
            all_votes.extend(anova_features)
        if corr_features:
            all_votes.extend(corr_features)
        
        # Count votes
        from collections import Counter
        vote_counts = Counter(all_votes)
        
        # Get top k by vote count
        top_features = [feat for feat, count in vote_counts.most_common(k)]
        
        print(f"\n  [OK] Final Selected Features ({len(top_features)}):")
        print(f"    {top_features}")
        
        self.selected_features = top_features
        return top_features
    
    def get_selected_features(self):
        """
        Return selected features
        
        Returns:
            list: Selected feature names
        """
        return self.selected_features
    
    def filter_dataset(self, df):
        """
        Filter dataframe to keep only selected features + Churn
        
        Args:
            df: Input dataframe
            
        Returns:
            df: Filtered dataframe with selected features
        """
        if self.selected_features:
            # Keep selected features + Churn column if it exists
            cols_to_keep = self.selected_features.copy()
            if 'Churn' in df.columns and 'Churn' not in cols_to_keep:
                cols_to_keep.append('Churn')
            
            return df[cols_to_keep]
        
        return df
