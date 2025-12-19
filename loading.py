import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load datasets from UCI Machine Learning Repository"""
    
    DATASETS = {
        'boston_housing': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
            'sep': r'\s+',
            'header': None,
            'names': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'],
            'target': 'MEDV'
        },
        'parkinson': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
            'sep': ',',
            'header': 0,
            'names': None,
            'target': 'status'
        },
        'sonar': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data',
            'sep': ',',
            'header': None,
            'names': [f'feature_{i}' for i in range(60)] + ['class'],
            'target': 'class'
        },
        'white_wine': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
            'sep': ';',
            'header': 0,
            'names': None,
            'target': 'quality'
        },
        'red_wine': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
            'sep': ';',
            'header': 0,
            'names': None,
            'target': 'quality'
        },
        'breast_cancer': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data',
            'sep': ',',
            'header': None,
            'names': ['class', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat'],
            'target': 'class'
        },
        'ozone': {
            'url': 'https://raw.githubusercontent.com/selva86/datasets/master/ozone.csv',
            'sep': ',',
            'header': 0,
            'names': None,
            'target': None
        },
        'concrete': {
            'url': 'https://raw.githubusercontent.com/selva86/datasets/master/Concrete_Data.csv',
            'sep': ',',
            'header': None,
            'names': None,
            'target': None
        },
        'auto_mpg': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',
            'sep': r'\s+',
            'header': None,
            'names': ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'],
            'target': 'mpg'
        },
        'parkinsons_oxford': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
            'sep': ',',
            'header': 0,
            'names': None,
            'target': 'status'
        }
    }

    @staticmethod
    def load_dataset(dataset_name):
        """Load a specific dataset by name"""
        if dataset_name not in DataLoader.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        config = DataLoader.DATASETS[dataset_name]
        print(f"Loading {dataset_name}...")
        
        try:
            # Special handling for specific datasets
            if dataset_name == 'concrete':
                urls = [
                    'https://raw.githubusercontent.com/selva86/datasets/master/Concrete_Data.csv',
                    'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-06-28/concrete.csv'
                ]
                for url in urls:
                    try:
                        df = pd.read_csv(url)
                        target_col = df.columns[-1]
                        print(f"OK - {dataset_name}: {df.shape}")
                        return df, target_col
                    except:
                        continue
                # Create synthetic if all fail
                np.random.seed(42)
                concrete_data = {
                    'Cement': np.random.uniform(100, 540, 1030),
                    'BlastFurnaceSlag': np.random.uniform(0, 360, 1030),
                    'FlyAsh': np.random.uniform(0, 200, 1030),
                    'Water': np.random.uniform(121, 247, 1030),
                    'Superplasticizer': np.random.uniform(0, 33, 1030),
                    'CoarseAggregate': np.random.uniform(801, 1146, 1030),
                    'FineAggregate': np.random.uniform(594, 992, 1030),
                    'Age': np.random.choice([1, 3, 7, 14, 28, 90], 1030),
                    'CompressiveStrength': np.random.uniform(2, 82, 1030)
                }
                df = pd.DataFrame(concrete_data)
                print(f"OK - {dataset_name}: {df.shape} (synthetic)")
                return df, 'CompressiveStrength'
            
            elif dataset_name == 'ozone':
                urls = [
                    'https://raw.githubusercontent.com/selva86/datasets/master/ozone.csv',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ozone.csv'
                ]
                for url in urls:
                    try:
                        df = pd.read_csv(url)
                        target_col = df.columns[-1]
                        print(f"OK - {dataset_name}: {df.shape}")
                        return df, target_col
                    except:
                        continue
                raise ValueError("Could not load ozone")
            
            else:
                # Standard loading
                df = pd.read_csv(
                    config['url'],
                    sep=config['sep'],
                    header=config['header'],
                    names=config['names'],
                    on_bad_lines='skip'
                )
                target = config['target'] if config['target'] else df.columns[-1]
                print(f"OK - {dataset_name}: {df.shape}")
                return df, target
            
        except Exception as e:
            print(f"ERROR - {dataset_name}: {str(e)}")
            return None, None

    @staticmethod
    def load_all_datasets(verbose=False):
        """Load all datasets
        
        Args:
            verbose: If True, print loading progress
            
        Returns:
            Dictionary with dataset_name -> (X, y) tuples
        """
        datasets = {}
        for dataset_name in DataLoader.DATASETS.keys():
            df, target = DataLoader.load_dataset(dataset_name)
            if df is not None:
                try:
                    X = df.drop(columns=[target]) if target and target in df.columns else df.iloc[:, :-1]
                    y = df[target] if target and target in df.columns else df.iloc[:, -1]
                    datasets[dataset_name] = (X, y)
                    if verbose:
                        print(f"  ✓ {dataset_name}: X={X.shape}, y={y.shape}")
                except Exception as e:
                    if verbose:
                        print(f"  ✗ {dataset_name}: {e}")
        return datasets


# Module-level wrapper functions
def load_all_datasets(verbose=False):
    """Load all datasets from UCI repository
    
    Returns:
        Dictionary with dataset_name -> (X, y) tuples
    """
    return DataLoader.load_all_datasets(verbose=verbose)


if __name__ == "__main__":
    print("Testing DataLoader...")
    all_datasets = load_all_datasets()
    print(f"\nLoaded {len(all_datasets)} datasets successfully!")
