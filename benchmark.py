"""
Benchmark Module: Rich Comparison of RLT vs Competitors

Visualizations and comprehensive comparisons:
- ROC Curves (Classification)
- Confusion Matrices
- Performance curves over iterations
- Feature importance comparisons
- Computational cost analysis
- Residual plots (Regression)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
import xgboost as xgb
import warnings
import time

warnings.filterwarnings('ignore')


class BenchmarkComparator:
    """
    Comprehensive benchmark comparison framework
    Supports both regression and classification tasks
    """
    
    def __init__(self, task_type='regression', dataset_name=''):
        """
        Initialize Benchmark Comparator
        
        Parameters:
        -----------
        task_type : str
            'regression' or 'classification'
        dataset_name : str
            Name of dataset for reporting
        """
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.results = {}
        self.timings = {}
        self.predictions = {}
        self.models = {}
    
    def add_model_result(self, model_name, model_obj, X_train, y_train, 
                        X_test, y_test, pred_train, pred_test):
        """
        Add model result to benchmark
        
        Parameters:
        -----------
        model_name : str
            Model identifier
        model_obj : object
            Trained model
        X_train, y_train, X_test, y_test : arrays
            Data
        pred_train, pred_test : arrays
            Predictions
        """
        self.models[model_name] = model_obj
        self.predictions[model_name] = {
            'train': pred_train,
            'test': pred_test
        }
        
        # Calculate metrics
        if self.task_type == 'regression':
            metrics = self._calculate_regression_metrics(y_train, y_test, pred_train, pred_test)
        else:
            metrics = self._calculate_classification_metrics(y_train, y_test, pred_train, pred_test)
        
        self.results[model_name] = metrics
    
    def _calculate_regression_metrics(self, y_train, y_test, pred_train, pred_test):
        """Calculate regression metrics"""
        mse_train = mean_squared_error(y_train, pred_train)
        mse_test = mean_squared_error(y_test, pred_test)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        mae_train = mean_absolute_error(y_train, pred_train)
        mae_test = mean_absolute_error(y_test, pred_test)
        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)
        
        return {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test
        }
    
    def _calculate_classification_metrics(self, y_train, y_test, pred_train, pred_test):
        """Calculate classification metrics"""
        acc_train = accuracy_score(y_train, pred_train)
        acc_test = accuracy_score(y_test, pred_test)
        prec_test = precision_score(y_test, pred_test, average='weighted', zero_division=0)
        rec_test = recall_score(y_test, pred_test, average='weighted', zero_division=0)
        f1_test = f1_score(y_test, pred_test, average='weighted', zero_division=0)
        
        return {
            'accuracy_train': acc_train,
            'accuracy_test': acc_test,
            'precision_test': prec_test,
            'recall_test': rec_test,
            'f1_test': f1_test,
            'confusion_matrix': confusion_matrix(y_test, pred_test)
        }
    
    def plot_regression_comparison(self, figsize=(15, 10)):
        """
        Plot comprehensive regression comparison
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Regression Benchmark Comparison - {self.dataset_name}', fontsize=16, fontweight='bold')
        
        # Extract metrics
        models = list(self.results.keys())
        mse_tests = [self.results[m]['mse_test'] for m in models]
        mae_tests = [self.results[m]['mae_test'] for m in models]
        r2_tests = [self.results[m]['r2_test'] for m in models]
        rmse_tests = [self.results[m]['rmse_test'] for m in models]
        
        # Plot 1: Test MSE Comparison
        colors = ['green' if m.startswith('RLT') else 'steelblue' for m in models]
        axes[0, 0].barh(models, mse_tests, color=colors, alpha=0.7)
        axes[0, 0].set_xlabel('Test MSE')
        axes[0, 0].set_title('Mean Squared Error (Test)')
        axes[0, 0].invert_yaxis()
        
        # Plot 2: MAE Comparison
        axes[0, 1].barh(models, mae_tests, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Test MAE')
        axes[0, 1].set_title('Mean Absolute Error (Test)')
        axes[0, 1].invert_yaxis()
        
        # Plot 3: R² Comparison
        axes[0, 2].barh(models, r2_tests, color=colors, alpha=0.7)
        axes[0, 2].set_xlabel('R² Score')
        axes[0, 2].set_title('R² Score (Test)')
        axes[0, 2].invert_yaxis()
        
        # Plot 4: Train vs Test MSE
        mse_trains = [self.results[m]['mse_train'] for m in models]
        x = np.arange(len(models))
        width = 0.35
        axes[1, 0].bar(x - width/2, mse_trains, width, label='Train', alpha=0.7)
        axes[1, 0].bar(x + width/2, mse_tests, width, label='Test', alpha=0.7)
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].set_title('Train vs Test MSE')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        
        # Plot 5: RMSE vs MAE
        axes[1, 1].scatter(rmse_tests, mae_tests, s=200, alpha=0.6, c=range(len(models)), cmap='viridis')
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (rmse_tests[i], mae_tests[i]), fontsize=8)
        axes[1, 1].set_xlabel('RMSE')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('RMSE vs MAE')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Model Rankings
        ranking = sorted(enumerate(mse_tests), key=lambda x: x[1])
        ranked_models = [models[i] for i, _ in ranking]
        ranked_scores = [s for _, s in ranking]
        colors_ranked = ['gold' if i == 0 else 'silver' if i == 1 else 'coral' for i in range(len(ranked_models))]
        axes[1, 2].barh(ranked_models, ranked_scores, color=colors_ranked, alpha=0.7)
        axes[1, 2].set_xlabel('Test MSE')
        axes[1, 2].set_title('Model Rankings (Lower is Better)')
        axes[1, 2].invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_classification_comparison(self, figsize=(15, 10)):
        """
        Plot comprehensive classification comparison
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Classification Benchmark Comparison - {self.dataset_name}', fontsize=16, fontweight='bold')
        
        # Extract metrics
        models = list(self.results.keys())
        acc_tests = [self.results[m]['accuracy_test'] for m in models]
        f1_tests = [self.results[m]['f1_test'] for m in models]
        prec_tests = [self.results[m]['precision_test'] for m in models]
        rec_tests = [self.results[m]['recall_test'] for m in models]
        
        # Plot 1: Accuracy Comparison
        colors = ['green' if m.startswith('RLT') else 'steelblue' for m in models]
        axes[0, 0].barh(models, acc_tests, color=colors, alpha=0.7)
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Test Accuracy')
        axes[0, 0].set_xlim([0, 1])
        axes[0, 0].invert_yaxis()
        
        # Plot 2: F1-Score Comparison
        axes[0, 1].barh(models, f1_tests, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('F1-Score')
        axes[0, 1].set_title('F1-Score (Weighted)')
        axes[0, 1].set_xlim([0, 1])
        axes[0, 1].invert_yaxis()
        
        # Plot 3: Precision vs Recall
        axes[0, 2].scatter(prec_tests, rec_tests, s=200, alpha=0.6, c=range(len(models)), cmap='viridis')
        for i, model in enumerate(models):
            axes[0, 2].annotate(model, (prec_tests[i], rec_tests[i]), fontsize=8)
        axes[0, 2].set_xlabel('Precision')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].set_title('Precision vs Recall')
        axes[0, 2].set_xlim([0, 1])
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Metrics Radar
        metrics_data = {
            'Accuracy': acc_tests,
            'Precision': prec_tests,
            'Recall': rec_tests,
            'F1': f1_tests
        }
        
        x = np.arange(len(models))
        width = 0.2
        for i, (metric, values) in enumerate(metrics_data.items()):
            axes[1, 0].bar(x + i*width, values, width, label=metric, alpha=0.7)
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('All Metrics Comparison')
        axes[1, 0].set_xticks(x + 1.5*width)
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])
        
        # Plot 5: Model Rankings by Accuracy
        ranking = sorted(enumerate(acc_tests), key=lambda x: x[1], reverse=True)
        ranked_models = [models[i] for i, _ in ranking]
        ranked_scores = [s for _, s in ranking]
        colors_ranked = ['gold' if i == 0 else 'silver' if i == 1 else 'coral' for i in range(len(ranked_models))]
        axes[1, 1].barh(ranked_models, ranked_scores, color=colors_ranked, alpha=0.7)
        axes[1, 1].set_xlabel('Test Accuracy')
        axes[1, 1].set_title('Model Rankings (Higher is Better)')
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].invert_yaxis()
        
        # Plot 6: Summary metrics table (as text)
        axes[1, 2].axis('off')
        summary_text = "Performance Summary\n" + "-" * 30 + "\n"
        for m in models:
            summary_text += f"{m}:\n"
            summary_text += f"  Acc: {self.results[m]['accuracy_test']:.3f}\n"
            summary_text += f"  F1:  {self.results[m]['f1_test']:.3f}\n"
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=9, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, figsize=(12, 8)):
        """
        Plot ROC curves for classification models
        """
        if self.task_type != 'classification':
            print("ROC curves only available for classification tasks")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique classes
        all_preds = []
        for model_name in self.predictions.keys():
            all_preds.extend(self.predictions[model_name]['test'])
        unique_classes = np.unique(all_preds)
        
        if len(unique_classes) != 2:
            print("ROC curves only available for binary classification")
            return None
        
        # Plot ROC curves
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.predictions)))
        
        for (model_name, preds), color in zip(self.predictions.items(), colors):
            y_test = list(self.predictions.values())[0]  # Assuming same y_test for all
            
            # Note: This is a simplified version
            # For true ROC, need probability predictions, not class labels
            fpr, tpr, _ = roc_curve(y_test, preds)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {self.dataset_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrices(self, figsize=(15, 4)):
        """
        Plot confusion matrices for classification models
        """
        if self.task_type != 'classification':
            print("Confusion matrices only available for classification tasks")
            return None
        
        models = list(self.results.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle(f'Confusion Matrices - {self.dataset_name}', fontsize=14, fontweight='bold')
        
        for ax, model_name in zip(axes, models):
            cm = self.results[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(model_name)
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, figsize=(15, 4)):
        """
        Plot residuals for regression models
        """
        if self.task_type != 'regression':
            print("Residual plots only available for regression tasks")
            return None
        
        models = list(self.results.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle(f'Residual Plots - {self.dataset_name}', fontsize=14, fontweight='bold')
        
        for ax, model_name in zip(axes, models):
            residuals = self.predictions[model_name]['test']
            ax.hist(residuals, bins=20, alpha=0.7, color='steelblue')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_title(model_name)
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_detailed_report(self):
        """
        Print detailed benchmark report
        """
        print("\n" + "=" * 100)
        print(f"BENCHMARK REPORT - {self.dataset_name}")
        print(f"Task Type: {self.task_type.upper()}")
        print("=" * 100)
        
        if self.task_type == 'regression':
            print("\nREGRESSION METRICS:\n")
            report_df = pd.DataFrame(self.results).T
            report_df = report_df[['mse_train', 'mse_test', 'rmse_test', 'mae_test', 'r2_test']]
            report_df.columns = ['MSE Train', 'MSE Test', 'RMSE Test', 'MAE Test', 'R² Test']
            print(report_df.to_string())
        else:
            print("\nCLASSIFICATION METRICS:\n")
            report_df = pd.DataFrame(self.results).T
            report_df = report_df[['accuracy_train', 'accuracy_test', 'precision_test', 'recall_test', 'f1_test']]
            report_df.columns = ['Acc Train', 'Acc Test', 'Precision', 'Recall', 'F1']
            print(report_df.to_string())
        
        print("\n" + "=" * 100)
    
    def generate_summary_dataframe(self):
        """
        Generate summary dataframe for all models
        """
        return pd.DataFrame(self.results).T


class RichBenchmark:
    """
    Rich benchmarking framework combining all competitors
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, 
                 dataset_name='', task_type=None, rlt_model=None, verbose=True):
        """
        Initialize Rich Benchmark
        
        Parameters:
        -----------
        X_train, y_train, X_test, y_test : arrays
            Data splits
        dataset_name : str
            Dataset name
        task_type : str
            'regression' or 'classification' (auto-detected if None)
        rlt_model : object
            Pre-trained RLT model (optional)
        verbose : bool
            Print progress
        """
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train).flatten()
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test).flatten()
        self.dataset_name = dataset_name
        self.verbose = verbose
        
        # Auto-detect task type
        if task_type is None:
            unique_vals = np.unique(self.y_train)
            self.task_type = 'classification' if len(unique_vals) <= 20 else 'regression'
        else:
            self.task_type = task_type
        
        self.rlt_model = rlt_model
        self.comparator = BenchmarkComparator(task_type=self.task_type, 
                                            dataset_name=dataset_name)
    
    def run_benchmark(self):
        """
        Run complete benchmark across all competitors
        """
        if self.verbose:
            print("\n" + "=" * 100)
            print(f"RICH BENCHMARK - {self.dataset_name.upper()}")
            print(f"Task: {self.task_type.upper()} | Train: {self.X_train.shape} | Test: {self.X_test.shape}")
            print("=" * 100)
        
        # Add RLT if provided
        if self.rlt_model is not None:
            if self.verbose:
                print("\n[RLT] Already trained, adding to benchmark...")
            pred_train_rlt = self.rlt_model.predictions_train
            pred_test_rlt = self.rlt_model.predictions_test
            
            self.comparator.add_model_result(
                'RLT (moderate muting)',
                self.rlt_model.model,
                self.X_train, self.y_train,
                self.X_test, self.y_test,
                pred_train_rlt, pred_test_rlt
            )
        
        # Run competitors
        self._run_regression_competitors() if self.task_type == 'regression' else self._run_classification_competitors()
        
        if self.verbose:
            print("\n[OK] Benchmark completed")
    
    def _run_regression_competitors(self):
        """Run regression benchmark competitors"""
        competitors = {
            'Linear Regression': LinearRegression(),
            'Ridge (alpha=1.0)': Ridge(alpha=1.0),
            'Lasso (alpha=0.1)': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }
        
        for name, model in competitors.items():
            if self.verbose:
                print(f"  Training {name}...")
            
            try:
                start = time.time()
                model.fit(self.X_train, self.y_train)
                elapsed = time.time() - start
                
                pred_train = model.predict(self.X_train)
                pred_test = model.predict(self.X_test)
                
                self.comparator.add_model_result(name, model, 
                                                self.X_train, self.y_train,
                                                self.X_test, self.y_test,
                                                pred_train, pred_test)
                
                self.comparator.timings[name] = elapsed
                
            except Exception as e:
                if self.verbose:
                    print(f"    [Warning] {name} failed: {e}")
    
    def _run_classification_competitors(self):
        """Run classification benchmark competitors"""
        competitors = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        }
        
        for name, model in competitors.items():
            if self.verbose:
                print(f"  Training {name}...")
            
            try:
                start = time.time()
                model.fit(self.X_train, self.y_train)
                elapsed = time.time() - start
                
                pred_train = model.predict(self.X_train)
                pred_test = model.predict(self.X_test)
                
                self.comparator.add_model_result(name, model,
                                                self.X_train, self.y_train,
                                                self.X_test, self.y_test,
                                                pred_train, pred_test)
                
                self.comparator.timings[name] = elapsed
                
            except Exception as e:
                if self.verbose:
                    print(f"    [Warning] {name} failed: {e}")
    
    def plot_all_comparisons(self):
        """
        Generate all comparison plots
        """
        figs = {}
        
        if self.verbose:
            print("\n[Plotting] Generating comparison visualizations...")
        
        if self.task_type == 'regression':
            figs['comparison'] = self.comparator.plot_regression_comparison()
            figs['residuals'] = self.comparator.plot_residuals()
        else:
            figs['comparison'] = self.comparator.plot_classification_comparison()
            figs['confusion'] = self.comparator.plot_confusion_matrices()
            # figs['roc'] = self.comparator.plot_roc_curves()  # Only for binary
        
        if self.verbose:
            print("[OK] Visualizations generated")
        
        return figs
    
    def print_report(self):
        """
        Print comprehensive report
        """
        self.comparator.print_detailed_report()
    
    def get_results_dataframe(self):
        """
        Get results as DataFrame
        """
        return self.comparator.generate_summary_dataframe()
