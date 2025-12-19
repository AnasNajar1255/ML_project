import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')


class DataPreparation:
    """Data preparation with noise injection and task-specific preprocessing"""
    
    def __init__(self, df, target_column, task_type=None, n_noise_features=None, random_state=42):
        """
        Initialize data preparation
        
        Args:
            df: pandas DataFrame
            target_column: name of target column
            task_type: 'regression' or 'classification' (auto-detected if None)
            n_noise_features: number of noise features to add (auto-calculated if None)
            random_state: for reproducibility
        """
        self.df = df.copy()
        self.target_column = target_column
        self.task_type = task_type or self._detect_task_type()
        self.random_state = random_state
        
        # Calculate noise features needed to reach 500 total columns
        n_current_features = df.shape[1] - 1  # Exclude target
        self.n_noise_features = n_noise_features or max(0, 500 - n_current_features)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        
    def _detect_task_type(self):
        """Auto-detect if task is regression or classification"""
        target = self.df[self.target_column]
        
        if target.dtype == 'object' or target.dtype == 'category':
            return 'classification'
        
        n_unique = target.nunique()
        n_samples = len(target)
        
        if n_unique < min(20, n_samples * 0.1):
            return 'classification'
        
        return 'regression'
    
    def analyze_missing_values(self):
        """
        Analyze missing values in the dataset
        
        Returns:
            dict: Missing values analysis with patterns and recommendations
        """
        missing_info = {}
        
        # Count missing values by column
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df)) * 100
        
        missing_summary = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_pct': missing_pct
        }).sort_values('missing_pct', ascending=False)
        
        # Filter only columns with missing values
        missing_cols = missing_summary[missing_summary['missing_count'] > 0]
        
        missing_info['summary'] = missing_cols
        missing_info['total_missing'] = missing_counts.sum()
        missing_info['pct_complete_cases'] = (self.df.dropna().shape[0] / len(self.df)) * 100
        
        # Missing patterns
        if len(missing_cols) > 0:
            missing_info['strategy_recommendation'] = self._recommend_missing_strategy(missing_cols)
        
        return missing_info
    
    def _recommend_missing_strategy(self, missing_cols):
        """Recommend strategy for missing value handling"""
        strategies = {}
        
        for col, row in missing_cols.iterrows():
            missing_pct = row['missing_pct']
            col_dtype = self.df[col].dtype
            
            if missing_pct > 50:
                strategies[col] = "DROP_COLUMN"  # Drop if >50% missing
            elif col_dtype in ['object', 'category']:
                strategies[col] = "FILL_MODE"  # Fill with mode for categorical
            elif missing_pct < 5:
                if self.task_type == 'regression':
                    strategies[col] = "FILL_MEAN"  # Fill with mean for numeric
                else:
                    strategies[col] = "FILL_MEDIAN"  # Fill with median for classification
            else:
                strategies[col] = "FILL_KNN"  # KNN imputation for moderate missing
        
        return strategies
    
    def detect_outliers(self, X):
        """
        Detect outliers using multiple methods
        
        Args:
            X: DataFrame with numeric features
            
        Returns:
            dict: Outlier information by method
        """
        outlier_info = {}
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = X[col]
            outliers = {}
            
            # IQR Method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((series < lower_bound) | (series > upper_bound))
            outliers['iqr'] = {
                'count': iqr_outliers.sum(),
                'pct': (iqr_outliers.sum() / len(series)) * 100,
                'bounds': [lower_bound, upper_bound]
            }
            
            # Z-Score Method (for normal distributions)
            z_scores = np.abs((series - series.mean()) / series.std())
            z_outliers = z_scores > 3
            outliers['zscore'] = {
                'count': z_outliers.sum(),
                'pct': (z_outliers.sum() / len(series)) * 100,
                'threshold': 3
            }
            
            # Modified Z-Score (robust to outliers)
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            modified_z_outliers = np.abs(modified_z_scores) > 3.5
            outliers['modified_zscore'] = {
                'count': modified_z_outliers.sum(),
                'pct': (modified_z_outliers.sum() / len(series)) * 100,
                'threshold': 3.5
            }
            
            outlier_info[col] = outliers
        
        return outlier_info
    
    def handle_missing_values(self, df, strategy=None):
        """
        Handle missing values according to strategy
        
        Args:
            df: DataFrame to clean
            strategy: dict with column-specific strategies or 'auto'
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        df_clean = df.copy()
        
        if strategy is None or strategy == 'auto':
            missing_analysis = self.analyze_missing_values()
            if 'strategy_recommendation' in missing_analysis:
                strategy = missing_analysis['strategy_recommendation']
            else:
                return df_clean
        
        for col, method in strategy.items():
            if col not in df_clean.columns:
                continue
                
            if method == "DROP_COLUMN":
                df_clean = df_clean.drop(columns=[col])
                print(f"   Dropped column '{col}' (>50% missing)")
                
            elif method == "FILL_MEAN":
                fill_value = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(fill_value)
                print(f"   Filled '{col}' with mean: {fill_value:.4f}")
                
            elif method == "FILL_MEDIAN":
                fill_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_value)
                print(f"   Filled '{col}' with median: {fill_value:.4f}")
                
            elif method == "FILL_MODE":
                fill_value = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'missing'
                df_clean[col] = df_clean[col].fillna(fill_value)
                print(f"   Filled '{col}' with mode: {fill_value}")
                
            elif method == "FILL_KNN":
                # Simple KNN-like imputation using correlation
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                if col in numeric_cols and len(numeric_cols) > 1:
                    corr_cols = df_clean[numeric_cols].corr()[col].abs().sort_values(ascending=False)
                    # Use most correlated feature for imputation
                    if len(corr_cols) > 1:
                        best_corr_col = corr_cols.index[1]  # Skip self-correlation
                        missing_mask = df_clean[col].isnull()
                        # Simple linear imputation based on correlation
                        slope = corr_cols[best_corr_col]
                        df_clean.loc[missing_mask, col] = df_clean.loc[missing_mask, best_corr_col] * slope
                        print(f"   Filled '{col}' using correlation with '{best_corr_col}'")
                    else:
                        # Fallback to median
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        print(f"   Filled '{col}' with median (KNN fallback)")
                else:
                    # For categorical, use mode
                    fill_value = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'missing'
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"   Filled '{col}' with mode: {fill_value}")
        
        return df_clean
    
    def handle_outliers(self, X, method='iqr', action='clip'):
        """
        Handle outliers in numeric features
        
        Args:
            X: DataFrame with features
            method: 'iqr', 'zscore', or 'modified_zscore'
            action: 'clip', 'remove', or 'transform'
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        X_clean = X.copy()
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numeric_cols:
            series = X_clean[col]
            
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (series < lower_bound) | (series > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((series - series.mean()) / series.std())
                outlier_mask = z_scores > 3
                lower_bound = series.mean() - 3 * series.std()
                upper_bound = series.mean() + 3 * series.std()
                
            elif method == 'modified_zscore':
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z_scores = 0.6745 * (series - median) / mad
                outlier_mask = np.abs(modified_z_scores) > 3.5
                # Estimate bounds for clipping
                lower_bound = series.quantile(0.01)
                upper_bound = series.quantile(0.99)
            
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                if action == 'clip':
                    X_clean[col] = np.clip(X_clean[col], lower_bound, upper_bound)
                    print(f"   Clipped {n_outliers} outliers in '{col}' to [{lower_bound:.4f}, {upper_bound:.4f}]")
                    
                elif action == 'remove':
                    X_clean = X_clean[~outlier_mask]
                    outliers_removed += n_outliers
                    print(f"   Removed {n_outliers} outlier rows for '{col}'")
                    
                elif action == 'transform':
                    # Log transform for positive skewed data
                    if series.min() > 0 and series.skew() > 1:
                        X_clean[col] = np.log1p(X_clean[col])
                        print(f"   Applied log transform to '{col}' (skew: {series.skew():.2f})")
                    else:
                        # Square root transform
                        if series.min() >= 0:
                            X_clean[col] = np.sqrt(X_clean[col])
                            print(f"   Applied sqrt transform to '{col}'")
        
        if outliers_removed > 0:
            print(f"   Total rows removed due to outliers: {outliers_removed}")
        
        return X_clean
    
    def add_noise_features(self):
        """Add noise features with 1:2 signal-to-noise ratio to reach 500 columns"""
        print(f"Adding {self.n_noise_features} noise features with 1:2 signal/noise ratio...")
        
        n_samples = len(self.df)
        original_features = [col for col in self.df.columns if col != self.target_column]
        
        # 🔧 CORRECTION CONTRAINTE 2: Rapport signal/bruit 1:2 selon méthodologie Zhu et al.
        noise_features = []
        np.random.seed(self.random_state)
        
        for i in range(self.n_noise_features):
            # Sélectionner une variable originale aléatoirement
            if len(original_features) > 0:
                selected_original = np.random.choice(original_features)
                original_values = self.df[selected_original].values
                
                # Créer variable avec rapport signal/bruit 1:2
                # Signal (1 partie) + Bruit (2 parties) = rapport 1:2
                signal_component = original_values
                noise_component = np.random.randn(n_samples) * 2  # 2x plus fort que le signal
                
                # Standardiser pour éviter domination par échelle
                signal_component = (signal_component - np.mean(signal_component)) / (np.std(signal_component) + 1e-8)
                
                # Combinaison avec rapport 1:2
                synthetic_feature = signal_component + noise_component
            else:
                # Fallback: bruit pur si pas de features originales
                synthetic_feature = np.random.randn(n_samples)
            
            noise_features.append(synthetic_feature)
        
        # Créer DataFrame avec les nouvelles features
        noise_cols = [f'noise_signal_1to2_{i}' for i in range(self.n_noise_features)]
        noise_df = pd.DataFrame(np.column_stack(noise_features), columns=noise_cols, index=self.df.index)
        
        # Combine with original data
        df_with_noise = pd.concat([self.df, noise_df], axis=1)
        
        print(f"Shape after noise injection: {df_with_noise.shape}")
        print(f"✅ Signal/Noise ratio 1:2 applied to {self.n_noise_features} features")
        return df_with_noise
    
    def split_train_test(self, df_prepared, train_size=150):
        """Split into train (150 samples) and test (rest)"""
        print(f"Splitting data: train={train_size}, test={len(df_prepared)-train_size}")
        
        X = df_prepared.drop(columns=[self.target_column])
        y = df_prepared[self.target_column]
        
        # Get indices
        all_indices = np.arange(len(X))
        np.random.seed(self.random_state)
        np.random.shuffle(all_indices)
        
        train_indices = all_indices[:train_size]
        test_indices = all_indices[train_size:]
        
        X_train = X.iloc[train_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
        y_train = y.iloc[train_indices].reset_index(drop=True)
        y_test = y.iloc[test_indices].reset_index(drop=True)
        
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_features(self, X_train, X_test):
        """
        Preprocess features: encode categoricals and standardize numerics
        TASK-SPECIFIC preprocessing
        Ensure final shape is exactly (n_samples, 500)
        """
        print(f"\nPreprocessing features for {self.task_type.upper()}...")
        
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"  Numeric features: {len(numeric_features)}")
        print(f"  Categorical features: {len(categorical_features)}")
        
        # Standardize numeric features
        if numeric_features:
            self.scaler = StandardScaler()
            X_train[numeric_features] = self.scaler.fit_transform(X_train[numeric_features])
            X_test[numeric_features] = self.scaler.transform(X_test[numeric_features])
            print(f"  Standardized numeric features")
        
        # Encode categorical features - TASK-SPECIFIC
        if categorical_features:
            if self.task_type == 'regression':
                # For regression: use LabelEncoder (ordinal encoding)
                print(f"  Encoding categorical features (LabelEncoder - Regression)")
                encoders = {}
                for col in categorical_features:
                    le = LabelEncoder()
                    # Combine train and test to fit encoder on all possible values
                    all_values = pd.concat([X_train[col], X_test[col]]).astype(str)
                    le.fit(all_values)
                    # Transform both sets
                    X_train[col] = le.transform(X_train[col].astype(str))
                    X_test[col] = le.transform(X_test[col].astype(str))
                    encoders[col] = le
                self.encoder = encoders
            else:
                # For classification: use OneHotEncoder
                print(f"  Encoding categorical features (OneHotEncoder - Classification)")
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                X_train_cat = ohe.fit_transform(X_train[categorical_features])
                X_test_cat = ohe.transform(X_test[categorical_features])
                
                # Get new column names
                new_cat_cols = ohe.get_feature_names_out(categorical_features)
                
                # Drop original categorical columns and add one-hot encoded
                X_train = X_train.drop(columns=categorical_features)
                X_test = X_test.drop(columns=categorical_features)
                
                X_train = pd.concat([X_train, pd.DataFrame(X_train_cat, columns=new_cat_cols, index=X_train.index)], axis=1)
                X_test = pd.concat([X_test, pd.DataFrame(X_test_cat, columns=new_cat_cols, index=X_test.index)], axis=1)
                
                self.encoder = ohe
                categorical_features = list(new_cat_cols)
        
        # ENSURE EXACTLY 500 FEATURES
        current_features = X_train.shape[1]
        target_features = 500
        
        if current_features < target_features:
            # Add more noise features to reach 500
            n_missing = target_features - current_features
            print(f"  Adding {n_missing} extra noise features to reach exactly 500...")
            
            np.random.seed(42)
            extra_noise_train = np.random.randn(len(X_train), n_missing)
            extra_noise_test = np.random.randn(len(X_test), n_missing)
            
            extra_cols = [f'extra_noise_{i}' for i in range(n_missing)]
            X_train = pd.concat([X_train, pd.DataFrame(extra_noise_train, columns=extra_cols, index=X_train.index)], axis=1)
            X_test = pd.concat([X_test, pd.DataFrame(extra_noise_test, columns=extra_cols, index=X_test.index)], axis=1)
            
        elif current_features > target_features:
            # Remove extra features (keep only first 500)
            n_excess = current_features - target_features
            print(f"  Removing {n_excess} excess features to keep exactly 500...")
            
            cols_to_keep = X_train.columns[:target_features].tolist()
            X_train = X_train[cols_to_keep]
            X_test = X_test[cols_to_keep]
        
        self.feature_names = X_train.columns.tolist()
        
        return X_train, X_test
    
    def prepare(self, train_size=150):
        """
        Execute full preparation pipeline
        
        Args:
            train_size: number of samples for training (default 150)
        
        Returns:
            dict with X_train, X_test, y_train, y_test
        """
        print("\n" + "=" * 80)
        print("DATA PREPARATION PIPELINE")
        print("=" * 80)
        print(f"Task Type: {self.task_type.upper()}")
        print(f"Original shape: {self.df.shape}")
        
        # Step 1: Add noise features
        df_with_noise = self.add_noise_features()
        
        # Step 2: Split train-test
        X_train, X_test, y_train, y_test = self.split_train_test(df_with_noise, train_size=train_size)
        
        # Step 3: Preprocess
        X_train, X_test = self.preprocess_features(X_train, X_test)
        
        # Store results
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"\n[FINAL SHAPES]")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        print("\n" + "=" * 80)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'task_type': self.task_type,
            'n_features': X_train.shape[1]
        }
    
    def get_summary(self):
        """Get preparation summary"""
        if self.X_train is None:
            print("Data not prepared yet. Call prepare() first.")
            return
        
        summary = {
            'task_type': self.task_type,
            'train_samples': self.X_train.shape[0],
            'test_samples': self.X_test.shape[0],
            'n_features': self.X_train.shape[1],
            'n_noise_features': self.n_noise_features,
            'feature_names': self.feature_names[:10],
            'scaler_used': self.scaler is not None,
            'encoder_used': self.encoder is not None,
            'target_train_shape': self.y_train.shape,
            'target_test_shape': self.y_test.shape
        }
        
        if self.task_type == 'classification':
            summary['train_class_dist'] = self.y_train.value_counts().to_dict()
            summary['test_class_dist'] = self.y_test.value_counts().to_dict()
        else:
            try:
                summary['train_target_mean'] = float(self.y_train.mean())
                summary['test_target_mean'] = float(self.y_test.mean())
            except:
                summary['train_target_mean'] = 0
                summary['test_target_mean'] = 0
        
        return summary
    
    def print_summary(self):
        """Print human-readable summary"""
        summary = self.get_summary()
        
        if summary is None:
            return
        
        print("\n" + "=" * 80)
        print("PREPARATION SUMMARY")
        print("=" * 80)
        print(f"Task Type: {summary['task_type'].upper()}")
        print(f"\nData Split:")
        print(f"  Train: {summary['train_samples']} samples")
        print(f"  Test: {summary['test_samples']} samples")
        print(f"\nFeatures:")
        print(f"  Total: {summary['n_features']}")
        print(f"  Noise: {summary['n_noise_features']}")
        print(f"  Scaler applied: {summary['scaler_used']}")
        print(f"  Encoder applied: {summary['encoder_used']}")
        
        if summary['task_type'] == 'classification':
            print(f"\nTarget Distribution (Train):")
            for cls, count in summary['train_class_dist'].items():
                print(f"  {cls}: {count}")
            print(f"\nTarget Distribution (Test):")
            for cls, count in summary['test_class_dist'].items():
                print(f"  {cls}: {count}")
        else:
            if 'train_target_mean' in summary:
                print(f"\nTarget Statistics:")
                print(f"  Train mean: {summary['train_target_mean']:.4f}")
                print(f"  Test mean: {summary['test_target_mean']:.4f}")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    from loading import DataLoader
    
    print("Testing DataPreparation...")
    all_data = DataLoader.load_all_datasets()
    
    # Test 1: Regression
    print("\n" + "=" * 80)
    print("TEST 1: REGRESSION (Boston Housing)")
    print("=" * 80)
    
    boston_data = all_data['boston_housing']['data']
    boston_target = all_data['boston_housing']['target']
    
    prep_reg = DataPreparation(boston_data, boston_target)
    result_reg = prep_reg.prepare(train_size=150)
    prep_reg.print_summary()
    
    # Test 2: Classification
    print("\n" + "=" * 80)
    print("TEST 2: CLASSIFICATION (Breast Cancer)")
    print("=" * 80)
    
    cancer_data = all_data['breast_cancer']['data']
    cancer_target = all_data['breast_cancer']['target']
    
    prep_clf = DataPreparation(cancer_data, cancer_target)
    result_clf = prep_clf.prepare(train_size=150)
    prep_clf.print_summary()
    
    print("\n" + "=" * 80)
    print("PREPARATION TESTS COMPLETED!")
    print("=" * 80)
