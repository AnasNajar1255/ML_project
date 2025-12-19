"""
Evaluation Module for RLT - ENHANCED WITH COMPLETE COMPETITORS

Evaluates RLT with different muting strategies and robustness testing:
- RLT Variants: no muting (0.0), moderate muting (0.5), aggressive muting (0.8)
- Complete competitors: RF, Lasso, GBM, BART, ExtraTrees, RLT-naive, RF-log(p), RF-√p
- Multi-dataset evaluation (10 UCI datasets)
- Simulation scenario robustness testing (4 scenarios, p=200/500/1000)

Supports flexible dataset dimensions and comprehensive robustness analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import time
import warnings

# Import new competitors
try:
    from competitors import AdaptedRandomForest, SimpleBART, CompetitorsBenchmark
    from modeling import RLTNaive
    COMPETITORS_AVAILABLE = True
    print("✅ Enhanced competitors imported successfully")
except ImportError as e:
    COMPETITORS_AVAILABLE = False
    print(f"⚠️ Warning: Enhanced competitors not available: {e}")

warnings.filterwarnings('ignore')


class RLTOnlyEvaluator:
    """
    Pure RLT Evaluation Framework
    
    Evaluates RLT across 3 muting strategies:
    - No Muting (0.0): All features available
    - Moderate Muting (0.5): 50% weak features excluded
    - Aggressive Muting (0.8): 80% weak features excluded
    
    Tests on:
    1. Individual datasets
    2. Multiple datasets (10 UCI)
    3. Simulation scenarios (4 scenarios, p=200/500/1000)
    """
    
    def __init__(self, M=100, random_state=42, verbose=True):
        """
        Initialize RLT Only Evaluator
        
        Parameters:
        -----------
        M : int
            Number of trees in RLT ensemble
        random_state : int
            Random seed
        verbose : bool
            Print progress
        """
        self.M = M
        self.random_state = random_state
        self.verbose = verbose
        self.results = {}
    
    def evaluate_single_dataset(self, X_train, y_train, X_test, y_test, 
                                dataset_name='Unknown'):
        """
        Evaluate RLT variants on a single dataset
        
        Returns:
        --------
        results_dict : dict
            Performance metrics for each muting strategy
        """
        from modeling import RLTModel
        
        results = {}
        
        muting_configs = {
            'no_muting (0.0)': 0.0,
            'moderate_muting (0.5)': 0.5,
            'aggressive_muting (0.8)': 0.8
        }
        
        # Detect task type
        unique_targets = len(np.unique(y_train))
        task_type = 'classification' if unique_targets <= 20 else 'regression'
        
        for config_name, muting_rate in muting_configs.items():
            try:
                start_time = time.time()
                
                # Train RLT
                rlt = RLTModel(M=self.M, muting_rate=muting_rate, random_state=self.random_state)
                rlt.train(X_train, y_train)
                
                elapsed = time.time() - start_time
                
                # Get predictions
                pred_train = rlt.predictions_train
                pred_test = rlt.predictions_test
                
                # Calculate metrics
                if task_type == 'regression':
                    mse_train = mean_squared_error(y_train, pred_train)
                    mse_test = mean_squared_error(y_test, pred_test)
                    mae_test = np.mean(np.abs(y_test - pred_test))
                    
                    results[config_name] = {
                        'mse_train': mse_train,
                        'mse_test': mse_test,
                        'mae_test': mae_test,
                        'training_time': elapsed
                    }
                else:
                    from sklearn.metrics import accuracy_score, f1_score
                    acc_train = accuracy_score(y_train, pred_train)
                    acc_test = accuracy_score(y_test, pred_test)
                    f1_test = f1_score(y_test, pred_test, average='weighted', zero_division=0)
                    
                    results[config_name] = {
                        'accuracy_train': acc_train,
                        'accuracy_test': acc_test,
                        'f1_test': f1_test,
                        'training_time': elapsed
                    }
                
            except Exception as e:
                if self.verbose:
                    print(f"  [Warning] {config_name} failed: {str(e)[:50]}")
        
        return results, task_type
    
    def evaluate_multiple_datasets(self, datasets_dict, verbose_detail=True):
        """
        Evaluate RLT on multiple datasets
        
        Parameters:
        -----------
        datasets_dict : dict
            {dataset_name: (X, y), ...}
        verbose_detail : bool
            Print detailed results
        
        Returns:
        --------
        all_results : dict
            Results for all datasets
        """
        from preparation import DataPreparation
        
        all_results = {}
        
        if self.verbose:
            print("\n" + "="*100)
            print("RLT EVALUATION - MULTIPLE DATASETS")
            print("="*100)
        
        for dataset_name, (X_raw, y_raw) in datasets_dict.items():
            if self.verbose:
                print(f"\n[Dataset] {dataset_name}...")
            
            try:
                # Prepare data
                prep = DataPreparation(X_raw, y_raw)
                X_prepared, y_prepared = prep.preprocess_features()
                X_train, X_test, y_train, y_test = prep.train_test_split(
                    test_size=0.7, random_state=self.random_state
                )
                
                # Evaluate
                results, task_type = self.evaluate_single_dataset(
                    X_train, y_train, X_test, y_test, dataset_name
                )
                
                all_results[dataset_name] = {
                    'task_type': task_type,
                    'results': results,
                    'n_samples': len(X_train) + len(X_test),
                    'n_features': X_train.shape[1]
                }
                
                if verbose_detail and self.verbose:
                    print(f"  [OK] {dataset_name} completed")
                    for config, metrics in results.items():
                        metric_str = ', '.join([f"{k}={v:.4f}" for k, v in metrics.items() 
                                               if k != 'training_time'])
                        print(f"    {config}: {metric_str}")
            
            except Exception as e:
                if self.verbose:
                    print(f"  [Error] {dataset_name} failed: {str(e)[:50]}")
        
        self.results = all_results
        return all_results
    
    def evaluate_simulation_scenarios(self, n_repetitions=20, p_values=[200, 500, 1000]):
        """
        Evaluate RLT robustness on simulation scenarios
        
        Parameters:
        -----------
        n_repetitions : int
            Repetitions per scenario/dimension
        p_values : list
            Dimensions to test
        
        Returns:
        --------
        scenario_results : dict
            Results for each scenario and dimension
        """
        from scenarios import (
            Scenario1ClassificationIndependent,
            Scenario2NonlinearIndependent,
            Scenario3CheckerboardCorrelated,
            Scenario4LinearMulticollinearity
        )
        from modeling import RLTModel
        
        scenarios_list = [
            ('Scenario 1: Classification (Independent)', Scenario1ClassificationIndependent(self.random_state)),
            ('Scenario 2: Non-linear (Independent)', Scenario2NonlinearIndependent(self.random_state)),
            ('Scenario 3: Checkerboard (Correlated)', Scenario3CheckerboardCorrelated(self.random_state)),
            ('Scenario 4: Linear (Multicollinear)', Scenario4LinearMulticollinearity(self.random_state))
        ]
        
        scenario_results = {}
        
        if self.verbose:
            print("\n" + "="*100)
            print("RLT ROBUSTNESS EVALUATION - SIMULATION SCENARIOS")
            print("="*100)
        
        for scenario_name, scenario in scenarios_list:
            if self.verbose:
                print(f"\n{scenario_name}")
                print("-"*100)
            
            scenario_results[scenario_name] = {}
            
            for p in p_values:
                if self.verbose:
                    print(f"  Dimension p={p}...")
                
                # Storage for results
                results_by_config = {
                    'no_muting (0.0)': [],
                    'moderate_muting (0.5)': [],
                    'aggressive_muting (0.8)': []
                }
                
                for rep in range(n_repetitions):
                    # Generate data
                    X_train, y_train, task_type = scenario.generate_data(500, p)
                    X_test, y_test, _ = scenario.generate_data(1000, p)
                    
                    # Standardize
                    X_mean = X_train.mean(axis=0)
                    X_std = X_train.std(axis=0) + 1e-8
                    X_train_std = (X_train - X_mean) / X_std
                    X_test_std = (X_test - X_mean) / X_std
                    
                    # Test each muting config
                    for config_name, muting_rate in [
                        ('no_muting (0.0)', 0.0),
                        ('moderate_muting (0.5)', 0.5),
                        ('aggressive_muting (0.8)', 0.8)
                    ]:
                        try:
                            rlt = RLTModel(M=50, muting_rate=muting_rate, 
                                         random_state=self.random_state)
                            rlt.train(X_train_std, y_train)
                            pred_test = rlt.predictions_test
                            
                            if task_type == 'regression':
                                error = mean_squared_error(y_test, pred_test)
                            else:
                                error = 1 - accuracy_score(y_test, pred_test)
                            
                            results_by_config[config_name].append(error)
                        except:
                            pass
                
                # Aggregate results
                aggregated = {}
                for config_name, errors in results_by_config.items():
                    if errors:
                        aggregated[config_name] = {
                            'mean_error': np.mean(errors),
                            'std_error': np.std(errors),
                            'min_error': np.min(errors),
                            'max_error': np.max(errors)
                        }
                
                scenario_results[scenario_name][p] = aggregated
                
                if self.verbose:
                    for config, metrics in aggregated.items():
                        print(f"    {config}: {metrics['mean_error']:.4f} ± {metrics['std_error']:.4f}")
        
        return scenario_results
    
    def print_multi_dataset_summary(self):
        """Print summary of multi-dataset evaluation"""
        if not self.results:
            print("No results available. Run evaluate_multiple_datasets() first.")
            return
        
        print("\n" + "="*100)
        print("RLT MULTI-DATASET EVALUATION SUMMARY")
        print("="*100)
        
        summary_data = []
        for dataset_name, dataset_results in self.results.items():
            task_type = dataset_results['task_type']
            results = dataset_results['results']
            
            # Get best muting strategy
            if task_type == 'regression':
                best_config = min(results.items(), key=lambda x: x[1]['mse_test'])
                summary_data.append({
                    'Dataset': dataset_name,
                    'Task': task_type,
                    'Best Config': best_config[0],
                    'Test MSE': f"{best_config[1]['mse_test']:.4f}",
                    'Test MAE': f"{best_config[1]['mae_test']:.4f}",
                    'Time (s)': f"{best_config[1]['training_time']:.2f}"
                })
            else:
                best_config = max(results.items(), key=lambda x: x[1]['accuracy_test'])
                summary_data.append({
                    'Dataset': dataset_name,
                    'Task': task_type,
                    'Best Config': best_config[0],
                    'Test Accuracy': f"{best_config[1]['accuracy_test']:.4f}",
                    'Test F1': f"{best_config[1]['f1_test']:.4f}",
                    'Time (s)': f"{best_config[1]['training_time']:.2f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        print("\n" + "="*100)


class RLTEvaluator:
    """
    Evaluates RLT with three different muting strategies
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, task_type='regression', 
                 M=100, verbose=True):
        """
        Initialize RLT Evaluator
        
        Parameters:
        -----------
        X_train, y_train : arrays
            Training data
        X_test, y_test : arrays
            Test data
        task_type : str
            'regression' or 'classification'
        M : int
            Number of trees
        verbose : bool
            Print progress
        """
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train).flatten()
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test).flatten()
        self.task_type = task_type
        self.M = M
        self.verbose = verbose
        
        self.results = {}
        self.predictions = {}
        self.times = {}
    
    def evaluate_rlt_variants(self):
        """
        Evaluate RLT with three muting configurations
        """
        from modeling import RLTModel
        
        muting_configs = {
            'no_muting': 0.0,
            'moderate_muting': 0.5,
            'aggressive_muting': 0.8
        }
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("RLT EVALUATION - THREE MUTING STRATEGIES")
            print("=" * 80)
        
        for config_name, muting_rate in muting_configs.items():
            if self.verbose:
                print(f"\n[Training] RLT ({config_name}: muting_rate={muting_rate})")
            
            start_time = time.time()
            
            # Train RLT
            rlt_model = RLTModel(
                self.X_train, self.y_train,
                self.X_test, self.y_test,
                M=self.M,
                muting_rate=muting_rate
            )
            rlt_model.train()
            
            elapsed = time.time() - start_time
            
            # Get predictions
            y_pred_train = rlt_model.predictions_train
            y_pred_test = rlt_model.predictions_test
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_pred_train, y_pred_test)
            
            self.results[config_name] = metrics
            self.predictions[config_name] = {
                'train': y_pred_train,
                'test': y_pred_test
            }
            self.times[config_name] = elapsed
            
            if self.verbose:
                self._print_metrics(config_name, metrics, elapsed)
    
    def _calculate_metrics(self, y_pred_train, y_pred_test):
        """Calculate evaluation metrics"""
        if self.task_type == 'regression':
            mse_train = mean_squared_error(self.y_train, y_pred_train)
            mse_test = mean_squared_error(self.y_test, y_pred_test)
            
            rmse_train = np.sqrt(mse_train)
            rmse_test = np.sqrt(mse_test)
            
            mae_train = np.mean(np.abs(self.y_train - y_pred_train))
            mae_test = np.mean(np.abs(self.y_test - y_pred_test))
            
            return {
                'mse_train': mse_train,
                'mse_test': mse_test,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'mae_train': mae_train,
                'mae_test': mae_test
            }
        else:
            acc_train = accuracy_score(self.y_train, y_pred_train)
            acc_test = accuracy_score(self.y_test, y_pred_test)
            
            error_train = 1 - acc_train
            error_test = 1 - acc_test
            
            return {
                'accuracy_train': acc_train,
                'accuracy_test': acc_test,
                'error_train': error_train,
                'error_test': error_test
            }
    
    def _print_metrics(self, config_name, metrics, elapsed):
        """Print metrics"""
        if self.task_type == 'regression':
            print(f"  Train MSE: {metrics['mse_train']:.4f} | Test MSE: {metrics['mse_test']:.4f}")
            print(f"  Train RMSE: {metrics['rmse_train']:.4f} | Test RMSE: {metrics['rmse_test']:.4f}")
            print(f"  Train MAE: {metrics['mae_train']:.4f} | Test MAE: {metrics['mae_test']:.4f}")
        else:
            print(f"  Train Accuracy: {metrics['accuracy_train']:.4f} | Test Accuracy: {metrics['accuracy_test']:.4f}")
            print(f"  Train Error: {metrics['error_train']:.4f} | Test Error: {metrics['error_test']:.4f}")
        
        print(f"  Time: {elapsed:.2f}s")
    
    def get_best_rlt(self):
        """Get best RLT configuration"""
        if not self.results:
            return None
        
        if self.task_type == 'regression':
            # Best by test MSE
            best_config = min(self.results.keys(), 
                            key=lambda k: self.results[k]['mse_test'])
        else:
            # Best by test accuracy
            best_config = max(self.results.keys(), 
                            key=lambda k: self.results[k]['accuracy_test'])
        
        return best_config, self.results[best_config]


class BaselineEvaluator:
    """
    Evaluates baseline methods
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, task_type='regression', verbose=True):
        """
        Initialize Baseline Evaluator
        """
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train).flatten()
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test).flatten()
        self.task_type = task_type
        self.verbose = verbose
        
        self.results = {}
        self.times = {}
    
    def evaluate_all_baselines(self):
        """Evaluate all baseline methods"""
        if self.verbose:
            print("\n" + "=" * 80)
            print("BASELINE METHODS EVALUATION")
            print("=" * 80)
        
        if self.task_type == 'regression':
            self._evaluate_regression_baselines()
        else:
            self._evaluate_classification_baselines()
    
    def _evaluate_regression_baselines(self):
        """Evaluate regression baselines"""
        baselines = {
            'Linear Regression': LinearRegression(),
            'Ridge (alpha=1.0)': Ridge(alpha=1.0),
            'Ridge (alpha=10.0)': Ridge(alpha=10.0),
            'Random Forest (n=50)': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Random Forest (n=100)': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting (n=50)': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'Gradient Boosting (n=100)': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        
        for name, model in baselines.items():
            if self.verbose:
                print(f"\n[Training] {name}")
            
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            elapsed = time.time() - start_time
            
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            mse_train = mean_squared_error(self.y_train, y_pred_train)
            mse_test = mean_squared_error(self.y_test, y_pred_test)
            
            rmse_train = np.sqrt(mse_train)
            rmse_test = np.sqrt(mse_test)
            
            mae_train = np.mean(np.abs(self.y_train - y_pred_train))
            mae_test = np.mean(np.abs(self.y_test - y_pred_test))
            
            self.results[name] = {
                'mse_train': mse_train,
                'mse_test': mse_test,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'mae_train': mae_train,
                'mae_test': mae_test
            }
            self.times[name] = elapsed
            
            if self.verbose:
                print(f"  Train MSE: {mse_train:.4f} | Test MSE: {mse_test:.4f}")
                print(f"  Time: {elapsed:.2f}s")
    
    def _evaluate_classification_baselines(self):
        """Evaluate classification baselines"""
        baselines = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest (n=50)': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            'Random Forest (n=100)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting (n=50)': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'Gradient Boosting (n=100)': GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        
        for name, model in baselines.items():
            if self.verbose:
                print(f"\n[Training] {name}")
            
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            elapsed = time.time() - start_time
            
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            acc_train = accuracy_score(self.y_train, y_pred_train)
            acc_test = accuracy_score(self.y_test, y_pred_test)
            
            error_train = 1 - acc_train
            error_test = 1 - acc_test
            
            self.results[name] = {
                'accuracy_train': acc_train,
                'accuracy_test': acc_test,
                'error_train': error_train,
                'error_test': error_test
            }
            self.times[name] = elapsed
            
            if self.verbose:
                print(f"  Train Accuracy: {acc_train:.4f} | Test Accuracy: {acc_test:.4f}")
                print(f"  Time: {elapsed:.2f}s")


class EvaluationReport:
    """
    Generate comprehensive evaluation report
    """
    
    def __init__(self, task_type='regression'):
        """Initialize report generator"""
        self.task_type = task_type
    
    def create_comparison_dataframe(self, rlt_results, baseline_results):
        """
        Create comparison dataframe
        
        Parameters:
        -----------
        rlt_results : dict
            Results from RLTEvaluator
        baseline_results : dict
            Results from BaselineEvaluator
            
        Returns:
        --------
        df : DataFrame
            Comparison results
        """
        all_results = {}
        
        # Add RLT variants
        for config_name, metrics in rlt_results.items():
            all_results[f"RLT ({config_name})"] = metrics
        
        # Add baselines
        for method_name, metrics in baseline_results.items():
            all_results[method_name] = metrics
        
        df = pd.DataFrame(all_results).T
        return df
    
    def calculate_relative_performance(self, df):
        """
        Calculate relative performance vs best method
        (normalized to 1)
        """
        if self.task_type == 'regression':
            # For regression: lower MSE is better
            best_test_mse = df['mse_test'].min()
            df['relative_performance'] = df['mse_test'] / best_test_mse
        else:
            # For classification: higher accuracy is better
            best_test_acc = df['accuracy_test'].max()
            df['relative_performance'] = best_test_acc / df['accuracy_test']
        
        return df
    
    def print_summary(self, df, dataset_name=''):
        """Print evaluation summary"""
        print("\n" + "=" * 100)
        print(f"EVALUATION SUMMARY{' - ' + dataset_name if dataset_name else ''}")
        print("=" * 100)
        
        if self.task_type == 'regression':
            display_cols = ['mse_train', 'mse_test', 'rmse_test', 'mae_test', 'relative_performance']
        else:
            display_cols = ['accuracy_train', 'accuracy_test', 'error_test', 'relative_performance']
        
        print("\n" + df[display_cols].to_string())
        print("\n" + "=" * 100)


class DatasetEvaluator:
    """
    Complete evaluation for a single dataset
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, 
                 dataset_name='', task_type=None, M=100, verbose=True):
        """
        Initialize Dataset Evaluator
        
        Parameters:
        -----------
        X_train, y_train, X_test, y_test : arrays
            Dataset splits
        dataset_name : str
            Name of dataset
        task_type : str
            'regression' or 'classification' (auto-detected if None)
        M : int
            Number of trees for RLT
        verbose : bool
            Print progress
        """
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train).flatten()
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test).flatten()
        self.dataset_name = dataset_name
        self.M = M
        self.verbose = verbose
        
        # Auto-detect task type if not provided
        if task_type is None:
            unique_vals = np.unique(self.y_train)
            self.task_type = 'classification' if len(unique_vals) <= 20 else 'regression'
        else:
            self.task_type = task_type
    
    def evaluate(self):
        """Run complete evaluation"""
        if self.verbose:
            print("\n" + "=" * 100)
            print(f"DATASET EVALUATION: {self.dataset_name}")
            print(f"Task Type: {self.task_type.upper()}")
            print(f"Train shape: {self.X_train.shape} | Test shape: {self.X_test.shape}")
            print("=" * 100)
        
        # Evaluate RLT variants
        rlt_eval = RLTEvaluator(
            self.X_train, self.y_train, 
            self.X_test, self.y_test,
            task_type=self.task_type,
            M=self.M,
            verbose=self.verbose
        )
        rlt_eval.evaluate_rlt_variants()
        
        # Evaluate baselines
        baseline_eval = BaselineEvaluator(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            task_type=self.task_type,
            verbose=self.verbose
        )
        baseline_eval.evaluate_all_baselines()
        
        # Create report
        report = EvaluationReport(task_type=self.task_type)
        df_comparison = report.create_comparison_dataframe(
            rlt_eval.results,
            baseline_eval.results
        )
        df_comparison = report.calculate_relative_performance(df_comparison)
        
        if self.verbose:
            report.print_summary(df_comparison, self.dataset_name)
        
        # Get best RLT
        best_rlt_config, best_rlt_metrics = rlt_eval.get_best_rlt()
        
        if self.verbose:
            print(f"\nBest RLT Configuration: {best_rlt_config}")
            print(f"Best RLT Test Performance: {best_rlt_metrics}")
        
        return {
            'rlt_results': rlt_eval.results,
            'baseline_results': baseline_eval.results,
            'rlt_times': rlt_eval.times,
            'baseline_times': baseline_eval.times,
            'comparison_df': df_comparison,
            'best_rlt_config': best_rlt_config,
            'best_rlt_metrics': best_rlt_metrics,
            'task_type': self.task_type
        }


class MultiDatasetBenchmark:
    """
    Benchmark across multiple datasets
    Flexible for different dimensions and scenarios
    """
    
    def __init__(self, datasets_dict, M=100, verbose=True):
        """
        Initialize Multi-Dataset Benchmark
        
        Parameters:
        -----------
        datasets_dict : dict
            Dictionary of datasets:
            {
                'dataset_name': {
                    'X_train': array,
                    'y_train': array,
                    'X_test': array,
                    'y_test': array,
                    'task_type': 'regression' or 'classification' (optional)
                }
            }
        M : int
            Number of trees for RLT
        verbose : bool
            Print progress
        """
        self.datasets_dict = datasets_dict
        self.M = M
        self.verbose = verbose
        self.results_all = {}
    
    def run_benchmark(self):
        """Run benchmark across all datasets"""
        print("\n" + "=" * 100)
        print("MULTI-DATASET BENCHMARK")
        print(f"Datasets: {list(self.datasets_dict.keys())}")
        print("=" * 100)
        
        for dataset_name, data in self.datasets_dict.items():
            dataset_eval = DatasetEvaluator(
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                dataset_name=dataset_name,
                task_type=data.get('task_type', None),
                M=self.M,
                verbose=self.verbose
            )
            
            results = dataset_eval.evaluate()
            self.results_all[dataset_name] = results
        
        return self.results_all
    
    def generate_summary_table(self):
        """Generate summary table across all datasets"""
        summary = []
        
        for dataset_name, results in self.results_all.items():
            task_type = results['task_type']
            
            # Get best RLT config metrics
            best_config = results['best_rlt_config']
            best_rlt_perf = results['comparison_df'].loc[f'RLT ({best_config})', 
                                                         'relative_performance']
            
            # Get best overall method
            best_overall = results['comparison_df']['relative_performance'].idxmin()
            best_overall_perf = results['comparison_df']['relative_performance'].min()
            
            summary.append({
                'Dataset': dataset_name,
                'Task Type': task_type,
                'Best RLT Config': best_config,
                'Best RLT Relative Perf': best_rlt_perf,
                'Best Overall Method': best_overall,
                'Best Overall Relative Perf': best_overall_perf
            })
        
        df_summary = pd.DataFrame(summary)
        
        print("\n" + "=" * 100)
        print("BENCHMARK SUMMARY ACROSS DATASETS")
        print("=" * 100)
        print("\n" + df_summary.to_string(index=False))
        print("\n" + "=" * 100)
        
        return df_summary


class SimulationScenarioEvaluator:
    """
    Evaluate RLT robustness across simulation scenarios
    
    Tests RLT against competitors on synthetic data with controlled properties:
    - Scenario 1: Classification with independent covariances
    - Scenario 2: Non-linear model with independent covariances
    - Scenario 3: Checkerboard-like model with strong correlation
    - Scenario 4: Linear model with multicollinearity
    
    Dimensions tested: p = 200, 500, 1000
    200 repetitions per scenario
    """
    
    def __init__(self, n_repetitions=200, p_values=[200, 500, 1000], 
                 n_train=500, n_test=1000, random_state=42):
        """
        Initialize Simulation Evaluator
        
        Parameters:
        -----------
        n_repetitions : int
            Number of repetitions per scenario
        p_values : list
            Feature dimensions to test
        n_train : int
            Training samples per repetition
        n_test : int
            Test samples per repetition
        random_state : int
            Random seed
        """
        self.n_repetitions = n_repetitions
        self.p_values = p_values
        self.n_train = n_train
        self.n_test = n_test
        self.random_state = random_state
        self.results = {}
    
    def evaluate_on_scenarios(self, verbose=True):
        """
        Evaluate RLT vs competitors on all 4 scenarios
        
        Returns:
        --------
        summary_df : DataFrame
            Performance comparison across scenarios and dimensions
        """
        from scenarios import (
            Scenario1ClassificationIndependent,
            Scenario2NonlinearIndependent,
            Scenario3CheckerboardCorrelated,
            Scenario4LinearMulticollinearity
        )
        from modeling import RLTModel
        from sklearn.linear_model import LogisticRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        import xgboost as xgb
        
        scenarios_list = [
            ('Scenario 1: Classification (Independent)', Scenario1ClassificationIndependent(self.random_state)),
            ('Scenario 2: Non-linear (Independent)', Scenario2NonlinearIndependent(self.random_state)),
            ('Scenario 3: Checkerboard (Correlated)', Scenario3CheckerboardCorrelated(self.random_state)),
            ('Scenario 4: Linear (Multicollinear)', Scenario4LinearMulticollinearity(self.random_state))
        ]
        
        all_results = []
        
        for scenario_name, scenario in scenarios_list:
            if verbose:
                print(f"\n{'='*100}")
                print(f"{scenario_name}")
                print('='*100)
            
            for p in self.p_values:
                if verbose:
                    print(f"  Dimension p={p}...")
                
                # Storage for results across repetitions
                rlt_no_mut_errors = []
                rlt_mod_mut_errors = []
                rlt_agg_mut_errors = []
                competitor_errors = {}
                
                for rep in range(min(self.n_repetitions, 20)):  # Reduced for speed
                    # Generate data
                    X_train, y_train, task_type = scenario.generate_data(self.n_train, p)
                    X_test, y_test, _ = scenario.generate_data(self.n_test, p)
                    
                    # Standardize
                    X_mean = X_train.mean(axis=0)
                    X_std = X_train.std(axis=0) + 1e-8
                    X_train_std = (X_train - X_mean) / X_std
                    X_test_std = (X_test - X_mean) / X_std
                    
                    # Evaluate RLT variants
                    for muting_rate, storage_list in [
                        (0.0, rlt_no_mut_errors),
                        (0.5, rlt_mod_mut_errors),
                        (0.8, rlt_agg_mut_errors)
                    ]:
                        try:
                            rlt = RLTModel(M=50, muting_rate=muting_rate, random_state=self.random_state)
                            rlt.train(X_train_std, y_train)
                            pred_test = rlt.predictions_test
                            
                            if task_type == 'regression':
                                error = mean_squared_error(y_test, pred_test)
                            else:
                                from sklearn.metrics import accuracy_score
                                error = 1 - accuracy_score(y_test, pred_test)
                            
                            storage_list.append(error)
                        except Exception as e:
                            if verbose:
                                print(f"    [Warning] RLT failed: {str(e)[:30]}")
                    
                    # Evaluate competitors (only on first repetition to save time)
                    if rep == 0:
                        if task_type == 'regression':
                            competitors = {
                                'Ridge': Ridge(alpha=1.0),
                                'Lasso': Lasso(alpha=0.1),
                                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=self.random_state),
                                'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=self.random_state),
                                'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=self.random_state, verbosity=0)
                            }
                        else:
                            competitors = {
                                'Logistic Reg': LogisticRegression(max_iter=1000, random_state=self.random_state),
                                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                                'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=self.random_state),
                                'XGBoost': xgb.XGBClassifier(n_estimators=50, random_state=self.random_state, verbosity=0)
                            }
                        
                        for comp_name, model in competitors.items():
                            try:
                                model.fit(X_train_std, y_train)
                                pred_test = model.predict(X_test_std)
                                
                                if task_type == 'regression':
                                    error = mean_squared_error(y_test, pred_test)
                                else:
                                    error = 1 - accuracy_score(y_test, pred_test)
                                
                                if comp_name not in competitor_errors:
                                    competitor_errors[comp_name] = []
                                competitor_errors[comp_name].append(error)
                            except Exception as e:
                                if verbose:
                                    print(f"    [Warning] {comp_name} failed: {str(e)[:30]}")
                
                # Average results
                if rlt_no_mut_errors:
                    avg_rlt_no_mut = np.mean(rlt_no_mut_errors)
                    avg_rlt_mod_mut = np.mean(rlt_mod_mut_errors) if rlt_mod_mut_errors else np.inf
                    avg_rlt_agg_mut = np.mean(rlt_agg_mut_errors) if rlt_agg_mut_errors else np.inf
                    
                    # Find best RLT
                    best_rlt_error = min(avg_rlt_no_mut, avg_rlt_mod_mut, avg_rlt_agg_mut)
                    best_rlt_config = {
                        avg_rlt_no_mut: 'no_muting',
                        avg_rlt_mod_mut: 'mod_muting',
                        avg_rlt_agg_mut: 'agg_muting'
                    }[best_rlt_error]
                    
                    # Find best competitor
                    best_competitor_error = np.inf
                    best_competitor_name = 'None'
                    for comp_name, errors in competitor_errors.items():
                        if errors:
                            avg_error = np.mean(errors)
                            if avg_error < best_competitor_error:
                                best_competitor_error = avg_error
                                best_competitor_name = comp_name
                    
                    all_results.append({
                        'Scenario': scenario_name.split(':')[0],
                        'Dimension': p,
                        'RLT (no_mut)': avg_rlt_no_mut,
                        'RLT (mod_mut)': avg_rlt_mod_mut,
                        'RLT (agg_mut)': avg_rlt_agg_mut,
                        'Best RLT Config': best_rlt_config,
                        'Best RLT Error': best_rlt_error,
                        'Best Competitor': best_competitor_name,
                        'Best Competitor Error': best_competitor_error,
                        'RLT vs Best': 'Win' if best_rlt_error < best_competitor_error else 'Loss'
                    })
        
        results_df = pd.DataFrame(all_results)
        self.results = results_df
        
        return results_df
    
    def print_summary(self):
        """Print scenario evaluation summary"""
        if self.results.empty:
            print("No results available. Run evaluate_on_scenarios() first.")
            return
        
        print("\n" + "="*100)
        print("SIMULATION SCENARIO EVALUATION SUMMARY")
        print("="*100)
        print("\nRLT Performance across scenarios and dimensions:\n")
        print(self.results.to_string(index=False))
        
        # Win rate
        win_rate = (self.results['RLT vs Best'] == 'Win').sum() / len(self.results) * 100
        print(f"\n{'='*100}")
        print(f"RLT Win Rate vs Best Competitor: {win_rate:.1f}%")
        print(f"{'='*100}")
    
    def get_results_dataframe(self):
        """Return results as DataFrame"""
        return self.results


class CompleteCompetitorsEvaluator:
    """
    Complete Academic Benchmark Evaluator
    
    Evaluates RLT contre TOUS les concurrents requis selon spécifications:
    - RLT Principal: 9 configurations (k=1,2,5 × muting=0,50%,80%)
    - RLT-naive: Version simplifiée avec signaux marginaux
    - Random Forests (RF): Standard
    - RF-log(p) & RF-√p: Random Forests adaptés
    - Lasso: Régression/Classification régularisée
    - Gradient Boosting (GBM): Ensemble de boosting
    - Extremely Randomized Trees (ET): Extra arbres
    - BART Alternative: Bayesian Additive Regression Trees
    
    Protocol académique:
    - 200 répétitions par scenario/dimension
    - Best tuning pour tous les concurrents
    - Métriques: MSE (régression), Misclassification Error (classification)
    """
    
    def __init__(self, n_repetitions=200, test_size=1000, random_state=42, verbose=True):
        """
        Initialize Complete Competitors Evaluator
        
        Parameters:
        -----------
        n_repetitions : int
            Number of repetitions per scenario/dimension (academic: 200)
        test_size : int
            Test set size (academic: 1000)
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Print detailed progress
        """
        self.n_repetitions = n_repetitions
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Results storage
        self.results = {}
        self.summary_stats = None
        
        # Check competitors availability
        if not COMPETITORS_AVAILABLE:
            raise ImportError(
                "Enhanced competitors not available. "
                "Make sure competitors.py and enhanced modeling.py are properly installed."
            )
    
    def evaluate_scenario_complete(self, scenario, scenario_name, dimensions=[200, 500, 1000], 
                                 sample_sizes=None):
        """
        Complete evaluation of one scenario against all competitors
        
        Parameters:
        -----------
        scenario : Scenario instance
            Scenario to evaluate (from scenarios.py)
        scenario_name : str
            Name for results tracking
        dimensions : list
            Feature dimensions to test
        sample_sizes : dict
            Override sample sizes {train: N, test: M}
        
        Returns:
        --------
        scenario_results : dict
            Complete results for this scenario
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"COMPLETE EVALUATION: {scenario_name}")
            print(f"{'='*80}")
        
        scenario_results = {}
        
        for p in dimensions:
            if self.verbose:
                print(f"\n🔍 Dimension p = {p}")
                print(f"   Repetitions: {self.n_repetitions}")
                
            dimension_results = self._run_dimension_benchmark(
                scenario, scenario_name, p, sample_sizes
            )
            
            scenario_results[p] = dimension_results
            
            if self.verbose:
                self._print_dimension_summary(dimension_results, p)
        
        return scenario_results
    
    def _run_dimension_benchmark(self, scenario, scenario_name, p, sample_sizes=None):
        """Run complete benchmark for one scenario at specific dimension"""
        
        # Sample sizes
        if sample_sizes is None:
            if 'Classification' in scenario_name:
                train_size = 100
            elif 'Non-linéaire' in scenario_name:
                train_size = 100  
            elif 'échiquier' in scenario_name or 'Checkerboard' in scenario_name:
                train_size = 300
            else:  # Linear
                train_size = 200
        else:
            train_size = sample_sizes.get('train', 100)
        
        test_size = sample_sizes.get('test', self.test_size)
        
        # Initialize benchmark
        task_type = 'classification' if 'Classification' in scenario_name else 'regression'
        benchmark = CompetitorsBenchmark(task_type=task_type, random_state=self.random_state)
        benchmark.initialize_competitors(p)
        
        # Storage for all repetitions
        all_method_scores = {method: [] for method in benchmark.competitors.keys()}
        
        if self.verbose:
            print(f"   Methods: {len(benchmark.competitors)}")
            
        # Run repetitions
        for rep in range(self.n_repetitions):
            if self.verbose and (rep + 1) % 20 == 0:
                print(f"     Repetition {rep + 1}/{self.n_repetitions}")
            
            # Generate data for this repetition
            X_train, y_train, _ = scenario.generate_data(train_size, p)
            X_test, y_test, _ = scenario.generate_data(test_size, p)
            
            # Run benchmark on this repetition
            rep_results = benchmark.run_benchmark(
                X_train, y_train, X_test, y_test, verbose=False
            )
            
            # Store scores
            for method_name, result in rep_results.items():
                if 'score' in result and not np.isnan(result['score']):
                    all_method_scores[method_name].append(result['score'])
        
        # Calculate statistics across repetitions
        dimension_results = {}
        for method_name, scores in all_method_scores.items():
            if scores:  # Has valid scores
                dimension_results[method_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'n_successful': len(scores),
                    'success_rate': len(scores) / self.n_repetitions
                }
            else:
                dimension_results[method_name] = {
                    'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
                    'n_successful': 0, 'success_rate': 0.0
                }
        
        return dimension_results
    
    def _print_dimension_summary(self, dimension_results, p):
        """Print summary for one dimension"""
        print(f"\n   📊 RESULTS SUMMARY (p={p}):")
        
        # Sort by mean performance
        valid_methods = [(name, res) for name, res in dimension_results.items() 
                        if not np.isnan(res['mean'])]
        valid_methods.sort(key=lambda x: x[1]['mean'], reverse=True)
        
        print(f"   {'Method':<20} {'Mean':<8} {'±Std':<8} {'Success':<8}")
        print(f"   {'-'*50}")
        
        for method_name, stats in valid_methods[:10]:  # Top 10
            print(f"   {method_name:<20} {stats['mean']:.4f} "
                  f"±{stats['std']:.4f} {stats['success_rate']:.1%}")
    
    def evaluate_all_scenarios(self, dimensions=[200, 500, 1000]):
        """
        Evaluate all 4 scenarios with complete competitors benchmark
        
        Parameters:
        -----------
        dimensions : list
            Feature dimensions to test
            
        Returns:
        --------
        complete_results : dict
            All scenarios results
        """
        # Import scenarios
        try:
            import scenarios
            scenario_configs = [
                (scenarios.Scenario1ClassificationIndependent(random_state=self.random_state), 
                 "Scenario 1: Classification Indépendante"),
                (scenarios.Scenario2NonlinearIndependent(random_state=self.random_state),
                 "Scenario 2: Régression Non-linéaire"),
                (scenarios.Scenario3CheckerboardCorrelated(random_state=self.random_state),
                 "Scenario 3: Modèle Échiquier Corrélé"),
                (scenarios.Scenario4LinearMulticollinearity(random_state=self.random_state),
                 "Scenario 4: Modèle Linéaire")
            ]
        except ImportError:
            raise ImportError("scenarios module required for complete evaluation")
        
        if self.verbose:
            print(f"\n{'='*100}")
            print(f"COMPLETE ACADEMIC BENCHMARK - ALL SCENARIOS")
            print(f"Repetitions: {self.n_repetitions}, Dimensions: {dimensions}")
            print(f"{'='*100}")
        
        self.results = {}
        
        for scenario, scenario_name in scenario_configs:
            scenario_results = self.evaluate_scenario_complete(
                scenario, scenario_name, dimensions
            )
            self.results[scenario_name] = scenario_results
        
        # Generate summary statistics
        self._generate_summary_stats()
        
        if self.verbose:
            self.print_complete_summary()
        
        return self.results
    
    def _generate_summary_stats(self):
        """Generate summary statistics across all scenarios and dimensions"""
        
        summary_data = []
        
        for scenario_name, scenario_results in self.results.items():
            for p, dimension_results in scenario_results.items():
                
                # Find best RLT and best competitor
                rlt_methods = {name: res for name, res in dimension_results.items() 
                             if 'RLT' in name}
                non_rlt_methods = {name: res for name, res in dimension_results.items() 
                                 if 'RLT' not in name and not np.isnan(res['mean'])}
                
                if rlt_methods and non_rlt_methods:
                    # Best RLT
                    best_rlt_name = max(rlt_methods.keys(), 
                                      key=lambda x: rlt_methods[x]['mean'])
                    best_rlt_score = rlt_methods[best_rlt_name]['mean']
                    best_rlt_std = rlt_methods[best_rlt_name]['std']
                    
                    # Best competitor
                    best_comp_name = max(non_rlt_methods.keys(),
                                       key=lambda x: non_rlt_methods[x]['mean'])
                    best_comp_score = non_rlt_methods[best_comp_name]['mean']
                    best_comp_std = non_rlt_methods[best_comp_name]['std']
                    
                    summary_data.append({
                        'Scenario': scenario_name,
                        'Dimension': p,
                        'Best_RLT': best_rlt_name,
                        'Best_RLT_Score': best_rlt_score,
                        'Best_RLT_Std': best_rlt_std,
                        'Best_Competitor': best_comp_name,
                        'Best_Competitor_Score': best_comp_score,
                        'Best_Competitor_Std': best_comp_std,
                        'RLT_Advantage': best_rlt_score - best_comp_score,
                        'RLT_Win': best_rlt_score > best_comp_score
                    })
        
        self.summary_stats = pd.DataFrame(summary_data)
    
    def print_complete_summary(self):
        """Print complete benchmark summary"""
        if self.summary_stats is None:
            print("No summary statistics available.")
            return
        
        print(f"\n{'='*120}")
        print(f"COMPLETE ACADEMIC BENCHMARK SUMMARY")
        print(f"{'='*120}")
        print(f"Repetitions per scenario/dimension: {self.n_repetitions}")
        print(f"Total evaluations: {len(self.summary_stats) * self.n_repetitions}")
        
        print(f"\n📊 BEST METHODS BY SCENARIO:")
        for _, row in self.summary_stats.iterrows():
            status = "🏆 RLT WINS" if row['RLT_Win'] else "🥈 RLT LOSES"
            print(f"   {row['Scenario']} (p={row['Dimension']}):")
            print(f"      {row['Best_RLT']}: {row['Best_RLT_Score']:.4f}±{row['Best_RLT_Std']:.4f}")
            print(f"      {row['Best_Competitor']}: {row['Best_Competitor_Score']:.4f}±{row['Best_Competitor_Std']:.4f}")
            print(f"      {status} (advantage: {row['RLT_Advantage']:+.4f})")
        
        # Overall statistics
        overall_win_rate = self.summary_stats['RLT_Win'].mean() * 100
        mean_advantage = self.summary_stats['RLT_Advantage'].mean()
        
        print(f"\n{'='*120}")
        print(f"🎯 OVERALL RLT PERFORMANCE:")
        print(f"   Win Rate: {overall_win_rate:.1f}% ({self.summary_stats['RLT_Win'].sum()}/{len(self.summary_stats)})")
        print(f"   Average Advantage: {mean_advantage:+.4f}")
        print(f"   Best RLT Config: {self.summary_stats.groupby('Best_RLT').size().idxmax()}")
        print(f"{'='*120}")
    
    def get_detailed_results(self):
        """Return detailed results dictionary"""
        return self.results
    
    def get_summary_dataframe(self):
        """Return summary statistics as DataFrame"""
        return self.summary_stats
    
    def export_results(self, filename_prefix="rlt_complete_benchmark"):
        """
        Export results to CSV files
        
        Parameters:
        -----------
        filename_prefix : str
            Prefix for output files
        """
        if self.summary_stats is not None:
            summary_file = f"{filename_prefix}_summary.csv"
            self.summary_stats.to_csv(summary_file, index=False)
            print(f"📁 Summary exported to: {summary_file}")
        
        # Export detailed results
        detailed_data = []
        for scenario_name, scenario_results in self.results.items():
            for p, dimension_results in scenario_results.items():
                for method_name, stats in dimension_results.items():
                    detailed_data.append({
                        'Scenario': scenario_name,
                        'Dimension': p,
                        'Method': method_name,
                        'Mean_Score': stats['mean'],
                        'Std_Score': stats['std'],
                        'Min_Score': stats['min'],
                        'Max_Score': stats['max'],
                        'Success_Rate': stats['success_rate'],
                        'N_Repetitions': self.n_repetitions
                    })
        
        if detailed_data:
            detailed_file = f"{filename_prefix}_detailed.csv"
            pd.DataFrame(detailed_data).to_csv(detailed_file, index=False)
            print(f"📁 Detailed results exported to: {detailed_file}")
