import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


class DataUnderstanding:
    """Flexible EDA class that adapts to regression or classification tasks"""
    
    def __init__(self, df, target_column, task_type=None):
        """
        Initialize the EDA analyzer
        
        Args:
            df: pandas DataFrame
            target_column: name of target column
            task_type: 'regression' or 'classification' (auto-detected if None)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.task_type = task_type or self._detect_task_type()
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        
    def _detect_task_type(self):
        """Auto-detect if task is regression or classification"""
        target = self.df[self.target_column]
        
        # Check if target is numeric or categorical
        if target.dtype == 'object' or target.dtype == 'category':
            return 'classification'
        
        # If numeric, check cardinality
        n_unique = target.nunique()
        n_samples = len(target)
        
        # If unique values < 10% of samples and < 20 unique values, likely classification
        if n_unique < min(20, n_samples * 0.1):
            return 'classification'
        
        return 'regression'
    
    def get_basic_info(self):
        """Get basic dataset information"""
        info = {
            'shape': self.df.shape,
            'task_type': self.task_type,
            'target_column': self.target_column,
            'n_features': self.X.shape[1],
            'n_samples': self.df.shape[0]
        }
        return info
    
    def get_missing_values(self):
        """Analyze missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_info = pd.DataFrame({
            'missing_count': missing,
            'missing_pct': missing_pct
        })
        
        return missing_info[missing_info['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    def get_duplicates(self):
        """Analyze duplicate rows"""
        n_duplicates = self.df.duplicated().sum()
        pct_duplicates = (n_duplicates / len(self.df)) * 100
        
        return {
            'n_duplicates': int(n_duplicates),
            'pct_duplicates': float(pct_duplicates)
        }
    
    def get_feature_types(self):
        """Categorize features by type"""
        numeric_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'n_numeric': len(numeric_features),
            'n_categorical': len(categorical_features)
        }
    
    def get_target_stats(self):
        """Get target variable statistics - TASK-SPECIFIC"""
        stats = {
            'dtype': str(self.y.dtype),
            'missing': int(self.y.isnull().sum()),
            'n_unique': int(self.y.nunique())
        }
        
        if self.task_type == 'regression':
            stats.update({
                'mean': float(self.y.mean()),
                'median': float(self.y.median()),
                'std': float(self.y.std()),
                'min': float(self.y.min()),
                'max': float(self.y.max()),
                'q25': float(self.y.quantile(0.25)),
                'q75': float(self.y.quantile(0.75)),
                'skewness': float(self.y.skew()),
                'kurtosis': float(self.y.kurtosis())
            })
        else:
            class_dist = self.y.value_counts().to_dict()
            class_prop = (self.y.value_counts() / len(self.y)).to_dict()
            stats.update({
                'class_distribution': class_dist,
                'class_proportions': class_prop,
                'class_imbalance_ratio': float(self.y.value_counts().max() / self.y.value_counts().min())
            })
        
        return stats
    
    def get_correlations(self):
        """Get feature correlations with target - TASK-SPECIFIC"""
        numeric_X = self.X.select_dtypes(include=[np.number])
        
        if numeric_X.empty:
            return pd.DataFrame()
        
        if self.task_type == 'regression':
            correlations = numeric_X.corrwith(self.y).sort_values(ascending=False)
            return pd.DataFrame({
                'feature': correlations.index,
                'correlation': correlations.values
            })
        else:
            correlations_list = []
            for col in numeric_X.columns:
                try:
                    corr = numeric_X[col].corr(self.y.astype('category').cat.codes)
                    correlations_list.append({'feature': col, 'correlation': float(corr)})
                except:
                    continue
            
            corr_df = pd.DataFrame(correlations_list)
            return corr_df.sort_values('correlation', key=abs, ascending=False) if not corr_df.empty else pd.DataFrame()
    
    def get_feature_distributions(self):
        """Analyze feature distributions"""
        numeric_features = self.X.select_dtypes(include=[np.number]).columns
        
        distributions = {}
        for feat in numeric_features:
            distributions[feat] = {
                'skewness': float(self.X[feat].skew()),
                'kurtosis': float(self.X[feat].kurtosis()),
                'outliers_iqr': int(self._count_outliers_iqr(self.X[feat]))
            }
        
        return distributions
    
    @staticmethod
    def _count_outliers_iqr(series):
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return ((series < lower) | (series > upper)).sum()
    
    def get_categorical_analysis(self):
        """Analyze categorical features - TASK-SPECIFIC"""
        categorical_features = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis = {}
        for feat in categorical_features:
            value_counts = self.X[feat].value_counts()
            
            analysis[feat] = {
                'n_unique': int(self.X[feat].nunique()),
                'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
                'top_value_freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing': int(self.X[feat].isnull().sum())
            }
        
        return analysis
    
    def plot_correlation_matrix(self, figsize=(10, 8), cmap='coolwarm'):
        """Plot correlation matrix heatmap for numeric features"""
        numeric_X = self.X.select_dtypes(include=[np.number])
        
        if numeric_X.empty:
            print("No numeric features to correlate")
            return
        
        # For classification, encode target
        if self.task_type == 'classification':
            y_encoded = pd.factorize(self.y)[0]
        else:
            y_encoded = self.y
        
        # Create correlation matrix with target
        corr_data = pd.concat([numeric_X, pd.Series(y_encoded, index=self.y.index, name=self.target_column)], axis=1)
        corr_matrix = corr_data.corr()
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap, 
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Matrix - {self.task_type.upper()}')
        plt.tight_layout()
        
        return plt
    
    def plot_target_distribution(self, figsize=(10, 5)):
        """Plot target variable distribution - TASK-SPECIFIC"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        if self.task_type == 'regression':
            # Histogram for regression
            axes[0].hist(self.y, bins=30, color='skyblue', edgecolor='black')
            axes[0].set_xlabel('Target Value')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Target Distribution - {self.task_type.upper()}')
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(self.y, dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q Plot')
        else:
            # Bar plot for classification
            class_counts = self.y.value_counts()
            axes[0].bar(range(len(class_counts)), class_counts.values, color='coral', edgecolor='black')
            axes[0].set_xticks(range(len(class_counts)))
            axes[0].set_xticklabels(class_counts.index, rotation=45)
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Class Distribution - {self.task_type.upper()}')
            
            # Pie chart
            axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                       colors=['#ff9999', '#66b3ff'])
            axes[1].set_title('Class Proportions')
        
        plt.tight_layout()
        return plt
    
    def plot_feature_distributions(self, figsize=(15, 10)):
        """Plot distributions of numeric features"""
        numeric_features = self.X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) == 0:
            print("No numeric features to plot")
            return
        
        n_features = len(numeric_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, feat in enumerate(numeric_features):
            axes[idx].hist(self.X[feat], bins=30, color='skyblue', edgecolor='black')
            axes[idx].set_xlabel(feat)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {feat}')
        
        # Hide unused subplots
        for idx in range(len(numeric_features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return plt
    
    def show_all_plots(self):
        """Display all plots for comprehensive EDA"""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS...")
        print("=" * 80)
        
        # Target distribution
        print("\n[1] Plotting target distribution...")
        plt1 = self.plot_target_distribution()
        plt1.show()
        
        # Feature distributions
        print("[2] Plotting feature distributions...")
        plt2 = self.plot_feature_distributions()
        plt2.show()
        
        # Correlation matrix
        print("[3] Plotting correlation matrix...")
        plt3 = self.plot_correlation_matrix()
        plt3.show()
        
        print("\n" + "=" * 80)
        print("ALL VISUALIZATIONS DISPLAYED!")
        print("=" * 80 + "\n")
        report = {
            'BASIC_INFO': self.get_basic_info(),
            'TASK_TYPE': self.task_type.upper(),
            'MISSING_VALUES': self.get_missing_values().to_dict('index'),
            'DUPLICATES': self.get_duplicates(),
            'FEATURE_TYPES': self.get_feature_types(),
            'TARGET_STATS': self.get_target_stats(),
            'FEATURE_DISTRIBUTIONS': self.get_feature_distributions(),
            'CATEGORICAL_ANALYSIS': self.get_categorical_analysis(),
            'CORRELATIONS': self.get_correlations().to_dict('records')
        }
        
        return report
    
    def print_summary(self):
        """Print a concise summary"""
        report = self.get_summary_report()
        basic = report['BASIC_INFO']
        target_stats = report['TARGET_STATS']
        
        print(f"\nDataset: {basic['shape']}")
        print(f"Task Type: {report['TASK_TYPE']}")
        print(f"Target: {basic['target_column']}")
        print(f"Numeric Features: {report['FEATURE_TYPES']['n_numeric']}")
        print(f"Categorical Features: {report['FEATURE_TYPES']['n_categorical']}")
        
        if report['TASK_TYPE'] == 'REGRESSION':
            print(f"\nTarget Stats:")
            print(f"  Mean: {target_stats['mean']:.4f}")
            print(f"  Std: {target_stats['std']:.4f}")
            print(f"  Range: [{target_stats['min']:.4f}, {target_stats['max']:.4f}]")
        else:
            print(f"\nClass Distribution:")
            for cls, count in target_stats['class_distribution'].items():
                print(f"  {cls}: {count}")
        
        if len(report['CORRELATIONS']) > 0:
            print(f"\nTop Correlations:")
            for corr in report['CORRELATIONS'][:3]:
                print(f"  {corr['feature']}: {corr['correlation']:.4f}")


if __name__ == "__main__":
    from loading import DataLoader
    
    print("\n" + "=" * 80)
    print("DATA UNDERSTANDING - COMPREHENSIVE TEST")
    print("=" * 80)
    
    all_data = DataLoader.load_all_datasets()
    
    # Test 1: Regression
    print("\n" + "=" * 80)
    print("TEST 1: REGRESSION (Boston Housing)")
    print("=" * 80)
    
    boston_data = all_data['boston_housing']['data']
    boston_target = all_data['boston_housing']['target']
    eda_reg = DataUnderstanding(boston_data, boston_target)
    eda_reg.print_summary()
    
    print("\nGenerating visualizations for Boston Housing...")
    eda_reg.show_all_plots()
    
    # Test 2: Classification
    print("\n" + "=" * 80)
    print("TEST 2: CLASSIFICATION (Breast Cancer)")
    print("=" * 80)
    
    cancer_data = all_data['breast_cancer']['data']
    cancer_target = all_data['breast_cancer']['target']
    eda_clf = DataUnderstanding(cancer_data, cancer_target)
    eda_clf.print_summary()
    
    print("\nGenerating visualizations for Breast Cancer...")
    eda_clf.show_all_plots()
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")
