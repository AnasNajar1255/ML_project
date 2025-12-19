"""
Competitors Module - Méthodes concurrentes pour benchmarking RLT

Implémentations spécialisées des méthodes concurrentes requises 
pour l'évaluation académique complète selon spécifications Zhu et al.:

1. RF-log(p) & RF-√p: Random Forests adaptés avec sélection de variables
2. BART Alternative: Bayesian Additive Regression Trees implementation
3. Extensions pour méthodes standards
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import warnings

warnings.filterwarnings('ignore')


class AdaptedRandomForest(BaseEstimator):
    """
    RF-log(p) & RF-√p: Random Forests Adaptés
    
    Ces méthodes sélectionnent initialement les log(p) ou √p variables 
    les plus importantes de chaque modèle RF et réajustent ensuite 
    le modèle en utilisant uniquement ces variables.
    """
    
    def __init__(self, adaptation_type='log_p', n_estimators=100, task_type='regression', 
                 random_state=42, **rf_params):
        """
        Initialize Adapted Random Forest
        
        Parameters:
        -----------
        adaptation_type : str
            'log_p' for log(p) variables or 'sqrt_p' for √p variables
        n_estimators : int
            Number of trees in forest
        task_type : str  
            'regression' or 'classification'
        random_state : int
            Random state for reproducibility
        **rf_params : dict
            Additional RandomForest parameters
        """
        self.adaptation_type = adaptation_type
        self.n_estimators = n_estimators
        self.task_type = task_type
        self.random_state = random_state
        self.rf_params = rf_params
        
        # Models
        self.initial_model = None
        self.adapted_model = None
        self.selected_features = None
        self.n_features_ = None
        
    def _calculate_n_selected(self, p_features):
        """Calculate number of features to select based on adaptation type"""
        if self.adaptation_type == 'log_p':
            return max(1, int(np.log(p_features)))
        elif self.adaptation_type == 'sqrt_p':
            return max(1, int(np.sqrt(p_features)))
        else:
            raise ValueError(f"Unknown adaptation_type: {self.adaptation_type}")
    
    def fit(self, X, y):
        """
        Fit adapted Random Forest with feature selection
        
        Steps:
        1. Train initial RF on all features
        2. Select top log(p) or √p most important features  
        3. Retrain RF using only selected features
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
        y : array-like, (n_samples,)
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        self.n_features_ = X.shape[1]
        n_selected = self._calculate_n_selected(self.n_features_)
        
        print(f"\n🔧 ADAPTED RANDOM FOREST - {self.adaptation_type.upper()}")
        print(f"Original features: {self.n_features_}")
        print(f"Selected features: {n_selected}")
        
        # Step 1: Train initial RF on all features
        if self.task_type == 'regression':
            self.initial_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                **self.rf_params
            )
        else:
            self.initial_model = RandomForestClassifier(
                n_estimators=self.n_estimators, 
                random_state=self.random_state,
                **self.rf_params
            )
        
        self.initial_model.fit(X, y)
        
        # Step 2: Select most important features
        feature_importances = self.initial_model.feature_importances_
        self.selected_features = np.argsort(feature_importances)[-n_selected:]
        
        print(f"Top {n_selected} feature indices: {sorted(self.selected_features)}")
        
        # Step 3: Retrain on selected features only
        X_selected = X[:, self.selected_features]
        
        if self.task_type == 'regression':
            self.adapted_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state + 1,  # Different seed for diversity
                **self.rf_params
            )
        else:
            self.adapted_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state + 1,
                **self.rf_params
            )
        
        self.adapted_model.fit(X_selected, y)
        
        print(f"✅ Adapted model trained successfully")
        
        return self
    
    def predict(self, X):
        """
        Predict using adapted Random Forest
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
        
        Returns:
        --------
        predictions : array, (n_samples,)
        """
        if self.adapted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Use only selected features
        X_selected = X[:, self.selected_features]
        
        return self.adapted_model.predict(X_selected)
    
    def score(self, X, y):
        """Calculate score using adapted model"""
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            return np.mean(predictions == y)
        else:
            # R² score
            y_mean = np.mean(y)
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


class SimpleBART(BaseEstimator):
    """
    BART Alternative: Simplified Bayesian Additive Regression Trees
    
    Approximation of BART using ensemble of shallow trees with Bayesian-inspired
    regularization since full BART implementation is complex.
    
    Key BART concepts approximated:
    - Sum of trees model: y = sum(T_i) + ε
    - Shallow trees (stumps or 2-level trees)
    - Shrinkage/regularization of tree contributions
    - Ensemble averaging
    """
    
    def __init__(self, n_trees=200, max_depth=2, shrinkage=0.1, 
                 task_type='regression', random_state=42):
        """
        Initialize Simple BART alternative
        
        Parameters:
        -----------
        n_trees : int
            Number of trees in sum (BART uses ~200)
        max_depth : int  
            Maximum depth of each tree (BART uses shallow trees)
        shrinkage : float
            Shrinkage parameter (regularization)
        task_type : str
            'regression' or 'classification'
        random_state : int
            Random seed
        """
        self.n_trees = min(n_trees, 30)  # Cap at 30 trees for speed
        self.max_depth = max_depth
        self.shrinkage = shrinkage
        self.task_type = task_type
        self.random_state = random_state
        
        # Model components
        self.trees = []
        self.residuals_history = []
        self.tree_contributions = []
        self.base_prediction = None
        
    def fit(self, X, y):
        """
        Fit Simple BART using iterative residual fitting
        
        BART Algorithm Approximation:
        1. Initialize with base prediction
        2. For each tree: fit to current residuals
        3. Add shrunken prediction to ensemble
        4. Update residuals
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
        y : array-like, (n_samples,)
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        np.random.seed(self.random_state)
        
        print(f"\n🌲 SIMPLE BART TRAINING")
        print(f"Trees: {self.n_trees}, Max depth: {self.max_depth}")
        print(f"Shrinkage: {self.shrinkage}, Task: {self.task_type}")
        
        # Step 1: Initialize with base prediction
        if self.task_type == 'regression':
            self.base_prediction = np.mean(y)
            current_pred = np.full(len(y), self.base_prediction)
            residuals = y - current_pred
        else:
            # For classification, use logit transformation
            p_pos = np.mean(y)
            self.base_prediction = np.log(p_pos / (1 - p_pos + 1e-8))
            current_pred = np.full(len(y), self.base_prediction)
            residuals = y - self._sigmoid(current_pred)
        
        # Step 2: Iteratively fit trees to residuals
        for tree_idx in range(self.n_trees):
            # Create shallow tree
            if self.task_type == 'regression':
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=max(5, len(y) // 20),
                    random_state=self.random_state + tree_idx
                )
            else:
                tree = DecisionTreeRegressor(  # Use regressor even for classification
                    max_depth=self.max_depth,
                    min_samples_split=max(5, len(y) // 20),
                    random_state=self.random_state + tree_idx
                )
            
            # Fit to current residuals
            tree.fit(X, residuals)
            tree_pred = tree.predict(X)
            
            # Apply shrinkage (key BART component)
            shrunken_pred = self.shrinkage * tree_pred
            self.tree_contributions.append(shrunken_pred)
            
            # Update ensemble prediction
            current_pred += shrunken_pred
            
            # Update residuals
            if self.task_type == 'regression':
                residuals = y - current_pred
            else:
                residuals = y - self._sigmoid(current_pred)
            
            # Store tree and residuals
            self.trees.append(tree)
            self.residuals_history.append(np.mean(np.abs(residuals)))
            
            if (tree_idx + 1) % 50 == 0:
                print(f"  Tree {tree_idx + 1}/{self.n_trees}, Avg residual: {self.residuals_history[-1]:.4f}")
        
        print(f"✅ Simple BART training completed")
        
        return self
    
    def _sigmoid(self, x):
        """Sigmoid function for classification"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def predict(self, X):
        """
        Predict using Simple BART ensemble
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
        
        Returns:
        --------
        predictions : array, (n_samples,)
        """
        if not self.trees:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy array  
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Start with base prediction
        ensemble_pred = np.full(X.shape[0], self.base_prediction)
        
        # Add contributions from all trees
        for tree in self.trees:
            tree_pred = tree.predict(X)
            ensemble_pred += self.shrinkage * tree_pred
        
        if self.task_type == 'regression':
            return ensemble_pred
        else:
            # Convert back to probabilities and classes
            probabilities = self._sigmoid(ensemble_pred)
            return (probabilities > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities for classification"""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get logit predictions
        ensemble_pred = np.full(X.shape[0], self.base_prediction)
        for tree in self.trees:
            tree_pred = tree.predict(X)
            ensemble_pred += self.shrinkage * tree_pred
        
        # Convert to probabilities
        prob_pos = self._sigmoid(ensemble_pred)
        prob_neg = 1 - prob_pos
        
        return np.column_stack([prob_neg, prob_pos])
    
    def score(self, X, y):
        """Calculate score"""
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            return np.mean(predictions == y)
        else:
            # R² score
            y_mean = np.mean(y)
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


class CompetitorsBenchmark:
    """
    Centralized benchmark for all competitor methods
    
    Provides unified interface for running RLT vs all competitors
    according to academic specifications
    """
    
    def __init__(self, task_type='regression', random_state=42):
        """
        Initialize competitors benchmark
        
        Parameters:
        -----------
        task_type : str
            'regression' or 'classification'
        random_state : int
            Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.competitors = {}
        self.results = {}
        
    def initialize_competitors(self, n_features):
        """
        Initialize all competitor methods for given number of features
        
        Parameters:
        -----------
        n_features : int
            Number of features in dataset
        """
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
        from sklearn.linear_model import Lasso, LogisticRegression
        
        print(f"\n🏁 INITIALIZING COMPETITORS BENCHMARK")
        print(f"Task: {self.task_type}, Features: {n_features}")
        
        # RLT models are handled separately in benchmark - not included here
        # This prevents import conflicts and allows external RLT management
        
        self.competitors = {}
        
        # Skip RLT models - they are managed externally in the benchmark
        # This prevents the "RLT models not available" warning
        
        # 2. Random Forests variants
        if self.task_type == 'regression':
            self.competitors['Random_Forest'] = {
                'model': RandomForestRegressor,
                'params': {'n_estimators': 50, 'random_state': self.random_state}
            }
        else:
            self.competitors['Random_Forest'] = {
                'model': RandomForestClassifier,
                'params': {'n_estimators': 50, 'random_state': self.random_state}
            }
        
        # 4. RF-log(p) and RF-√p (optimized for speed)
        self.competitors['RF-log(p)'] = {
            'model': AdaptedRandomForest,
            'params': {
                'adaptation_type': 'log_p',
                'n_estimators': 50,
                'task_type': self.task_type,
                'random_state': self.random_state
            }
        }
        
        self.competitors['RF-√p'] = {
            'model': AdaptedRandomForest,
            'params': {
                'adaptation_type': 'sqrt_p', 
                'n_estimators': 50,
                'task_type': self.task_type,
                'random_state': self.random_state
            }
        }
        
        # 5. Gradient Boosting (optimized for speed)
        if self.task_type == 'regression':
            self.competitors['GradientBoosting'] = {
                'model': GradientBoostingRegressor,
                'params': {'n_estimators': 50, 'random_state': self.random_state}
            }
        else:
            self.competitors['GradientBoosting'] = {
                'model': GradientBoostingClassifier,  
                'params': {'n_estimators': 50, 'random_state': self.random_state}
            }
        
        # 6. Extremely Randomized Trees (optimized for speed)
        if self.task_type == 'regression':
            self.competitors['ExtraTrees'] = {
                'model': ExtraTreesRegressor,
                'params': {'n_estimators': 50, 'random_state': self.random_state}
            }
        else:
            self.competitors['ExtraTrees'] = {
                'model': ExtraTreesClassifier,
                'params': {'n_estimators': 50, 'random_state': self.random_state}
            }
        
        # 7. Lasso
        if self.task_type == 'regression':
            self.competitors['Lasso'] = {
                'model': Lasso,
                'params': {'alpha': 1.0, 'random_state': self.random_state}
            }
        else:
            self.competitors['Lasso'] = {
                'model': LogisticRegression,
                'params': {
                    'penalty': 'l1', 'solver': 'liblinear',
                    'C': 1.0, 'random_state': self.random_state
                }
            }
        
        # 8. BART Alternative (optimized for speed)
        self.competitors['BART'] = {
            'model': SimpleBART,
            'params': {
                'n_trees': 20,  # Further reduced for benchmark speed
                'task_type': self.task_type,
                'random_state': self.random_state
            }
        }
        
        print(f"✅ {len(self.competitors)} competitors initialized")
        
        return list(self.competitors.keys())
    
    def run_benchmark(self, X_train, y_train, X_test, y_test, verbose=True):
        """
        Run benchmark on all competitors
        
        Parameters:
        -----------
        X_train, y_train : arrays
            Training data
        X_test, y_test : arrays
            Test data
        verbose : bool
            Print progress
        
        Returns:
        --------
        results : dict
            {method_name: {'score': float, 'predictions': array}}
        """
        if not self.competitors:
            self.initialize_competitors(X_train.shape[1])
        
        self.results = {}
        
        if verbose:
            print(f"\n🏁 RUNNING BENCHMARK - {len(self.competitors)} methods")
        
        for method_name, config in self.competitors.items():
            if verbose:
                print(f"  🔄 {method_name}...")
            
            try:
                # All competitors use standard sklearn interface
                # RLT models are handled externally in the benchmark
                model = config['model'](**config['params'])
                model.fit(X_train, y_train)
                
                # Get predictions and score
                predictions = model.predict(X_test)
                score = model.score(X_test, y_test)
                
                self.results[method_name] = {
                    'score': score,
                    'predictions': predictions,
                    'model': model
                }
                
                if verbose:
                    print(f"    ✅ Score: {score:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"    ❌ Error: {str(e)[:50]}...")
                self.results[method_name] = {
                    'score': np.nan,
                    'error': str(e)
                }
        
        if verbose:
            print(f"✅ Benchmark completed")
        
        return self.results
    
    def get_ranking(self):
        """Get methods ranked by performance"""
        valid_results = {name: res['score'] for name, res in self.results.items() 
                        if 'score' in res and not np.isnan(res['score'])}
        
        # Sort by score (higher is better for both accuracy and R²)
        ranked = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def print_results(self):
        """Print formatted results"""
        print(f"\n📊 BENCHMARK RESULTS - {self.task_type.upper()}")
        print("="*60)
        
        ranking = self.get_ranking()
        
        for i, (method, score) in enumerate(ranking, 1):
            print(f"{i:2d}. {method:<20} {score:.4f}")
        
        # Show errors
        errors = [(name, res.get('error', '')) for name, res in self.results.items() 
                 if 'error' in res]
        
        if errors:
            print(f"\n❌ ERRORS:")
            for method, error in errors:
                print(f"    {method}: {error[:50]}...")
        
        print("="*60)