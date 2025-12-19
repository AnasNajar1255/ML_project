"""
Simulation Scenarios Module

Four comprehensive simulation scenarios for evaluating RLT and competitors:
1. Classification with independent covariances
2. Non-linear model with independent covariances
3. Checkerboard-like model with strong correlation
4. Linear model with multicollinearity

Each scenario tested at p = 200, 500, 1000 dimensions
200 repetitions, 1000 independent test samples per repetition
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings

warnings.filterwarnings('ignore')


class SimulationScenario:
    """Base class for simulation scenarios"""
    
    def __init__(self, scenario_name, random_state=42):
        self.scenario_name = scenario_name
        self.random_state = random_state
        self.np_random = np.random.RandomState(random_state)
    
    def generate_data(self, n_samples, p_features):
        """
        Generate data for the scenario
        Returns: X, y, task_type
        """
        raise NotImplementedError
    
    def run_repetitions(self, n_repetitions=200, p_values=[200, 500, 1000], 
                       n_train=500, n_test=1000, verbose=True):
        """
        Run multiple repetitions of the scenario with RLT and competitors
        
        Parameters:
        -----------
        n_repetitions : int
            Number of times to repeat the simulation
        p_values : list
            Feature dimensions to test
        n_train : int
            Training samples per repetition
        n_test : int
            Test samples per repetition
        verbose : bool
            Print progress
        
        Returns:
        --------
        results : dict
            Results dictionary: results[p][algorithm] = {'mean': X, 'std': Y, 'scores': [...]}
        """
        from sklearn.linear_model import Ridge, Lasso, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
        import sys
        import os
        
        # Add workspace to path for RLT import
        WORKSPACE = r'c:\Users\ayoub\Desktop\PROJET_ZHU_F'
        if WORKSPACE not in sys.path:
            sys.path.insert(0, WORKSPACE)
        
        try:
            from modeling import RLTModel
        except ImportError:
            print("⚠️ Warning: Could not import RLTModel - RLT will be skipped")
            RLTModel = None
        
        results = {}
        
        for p in p_values:
            if verbose:
                print(f"\n  Processing p={p}...")
            
            # Initialize algorithm results storage
            algorithm_scores = {
                'RLT': [],
                'Lasso': [],
                'RandomForest': [], 
                'GradientBoosting': []
            }
            
            for rep in range(n_repetitions):
                # Generate training data
                X_train, y_train, task_type = self.generate_data(n_train, p)
                
                # Generate test data  
                X_test, y_test, _ = self.generate_data(n_test, p)
                
                # Convert to DataFrames for RLT compatibility
                import pandas as pd
                X_train_df = pd.DataFrame(X_train)
                X_test_df = pd.DataFrame(X_test)
                y_train_series = pd.Series(y_train)
                y_test_series = pd.Series(y_test)
                
                # Test each algorithm
                rep_scores = {}
                
                # 1. RLT
                if RLTModel is not None:
                    try:
                        # Paramètres adaptés selon le scénario
                        if 'Classification' in self.scenario_name:
                            M, muting_rate = 30, 0.2  # Optimisé pour classification
                        elif 'linéaire locale' in self.scenario_name:
                            M, muting_rate = 80, 0.3  # Plus d'arbres pour non-linéarités
                        elif 'Interactions' in self.scenario_name:
                            M, muting_rate = 120, 0.4  # Encore plus pour interactions
                        else:
                            M, muting_rate = 15, 0.1  # Conservateur pour linéaire
                        
                        rlt_model = RLTModel(
                            X_train=X_train_df, y_train=y_train_series,
                            X_test=X_test_df, y_test=y_test_series,
                            M=M, muting_rate=muting_rate
                        )
                        rlt_model.train()
                        
                        if task_type == 'regression':
                            pred_test = rlt_model.predictions_test
                            score = mean_squared_error(y_test, pred_test)
                        else:
                            # CORRECTION SPÉCIALE POUR CLASSIFICATION
                            pred_test = rlt_model.predictions_test
                            
                            # Conversion robuste des prédictions object vers numérique
                            if hasattr(pred_test, 'dtype') and pred_test.dtype == 'object':
                                # Extraire valeurs numériques des objects
                                pred_numeric = np.array([float(p) if isinstance(p, (int, float, np.number)) 
                                                       else (1.0 if p == 1 or p == '1' or p == True
                                                            else 0.0) for p in pred_test])
                            else:
                                pred_numeric = np.asarray(pred_test, dtype=np.float64)
                            
                            # Vérifier et nettoyer les NaN
                            if np.any(np.isnan(pred_numeric)):
                                # Remplacer NaN par prédiction aléatoire
                                nan_mask = np.isnan(pred_numeric)
                                pred_numeric[nan_mask] = self.np_random.choice([0, 1], size=np.sum(nan_mask))
                            
                            # Conversion binaire sûre
                            pred_test_binary = np.round(np.clip(pred_numeric, 0, 1)).astype(int)
                            score = 1 - accuracy_score(y_test, pred_test_binary)
                        
                        rep_scores['RLT'] = score
                    except Exception as e:
                        if verbose and rep == 0:
                            print(f"    ⚠️ RLT failed: {str(e)[:50]}...")
                        rep_scores['RLT'] = np.nan
                else:
                    rep_scores['RLT'] = np.nan
                
                # 2. Lasso
                try:
                    if task_type == 'regression':
                        model = Lasso(alpha=0.1, max_iter=1000)
                        model.fit(X_train, y_train)
                        pred_test = model.predict(X_test)
                        score = mean_squared_error(y_test, pred_test)
                    else:
                        model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000)
                        model.fit(X_train, y_train)
                        pred_test = model.predict(X_test)
                        score = 1 - accuracy_score(y_test, pred_test)
                    
                    rep_scores['Lasso'] = score
                except Exception:
                    rep_scores['Lasso'] = np.nan
                
                # 3. Random Forest
                try:
                    if task_type == 'regression':
                        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                        model.fit(X_train, y_train)
                        pred_test = model.predict(X_test)
                        score = mean_squared_error(y_test, pred_test)
                    else:
                        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                        model.fit(X_train, y_train)
                        pred_test = model.predict(X_test)
                        score = 1 - accuracy_score(y_test, pred_test)
                    
                    rep_scores['RandomForest'] = score
                except Exception:
                    rep_scores['RandomForest'] = np.nan
                
                # 4. Gradient Boosting
                try:
                    if task_type == 'regression':
                        model = GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42)
                        model.fit(X_train, y_train)
                        pred_test = model.predict(X_test)
                        score = mean_squared_error(y_test, pred_test)
                    else:
                        model = GradientBoostingClassifier(n_estimators=50, max_depth=6, random_state=42)
                        model.fit(X_train, y_train)
                        pred_test = model.predict(X_test)
                        score = 1 - accuracy_score(y_test, pred_test)
                    
                    rep_scores['GradientBoosting'] = score
                except Exception:
                    rep_scores['GradientBoosting'] = np.nan
                
                # Store scores
                for alg, score in rep_scores.items():
                    algorithm_scores[alg].append(score)
                
                if verbose and (rep + 1) % 20 == 0:
                    print(f"    Completed {rep + 1}/{n_repetitions} repetitions")
            
            # Calculate statistics for this dimension
            results[p] = {}
            for alg, scores in algorithm_scores.items():
                valid_scores = [s for s in scores if not np.isnan(s)]
                if valid_scores:
                    results[p][alg] = {
                        'mean': np.mean(valid_scores),
                        'std': np.std(valid_scores),
                        'scores': valid_scores,
                        'success_rate': len(valid_scores) / len(scores)
                    }
                else:
                    results[p][alg] = {
                        'mean': np.nan,
                        'std': np.nan,
                        'scores': [],
                        'success_rate': 0.0
                    }
        
        return results


class Scenario1ClassificationIndependent(SimulationScenario):
    """
    Scenario 1: Classification non linéaire, variables indépendantes
    
    X ~ Uniforme[0,1]^p
    Y ~ Bernoulli(μ(X))
    μ(X) = Φ(10(X₁-1) + 20|X₂-0.5|)
    
    Seulement 2 variables sont réellement utiles, les autres sont du bruit.
    Ce que RLT doit prouver: RLT > RF sur effets non linéaires
    """
    
    def __init__(self, random_state=42):
        super().__init__('Scenario1: Classification non-linéaire indépendante', random_state)
    
    def generate_data(self, n_samples, p_features):
        """Generate classification data selon spécifications exactes"""
        from scipy.stats import norm
        
        # Generate independent uniform features [0,1]
        X = self.np_random.uniform(0, 1, size=(n_samples, p_features))
        
        # True non-linear model: μ(X) = Φ(10(X₁-1) + 20|X₂-0.5|)
        linear_combination = 10 * (X[:, 0] - 1) + 20 * np.abs(X[:, 1] - 0.5)
        prob = norm.cdf(linear_combination)  # Φ(z) = norm.cdf(z)
        
        # Generate binary labels
        y = (self.np_random.rand(n_samples) < prob).astype(int)
        
        return X, y, 'classification'


class Scenario2NonlinearIndependent(SimulationScenario):
    """
    Scenario 2: Régression non linéaire locale
    
    Y = 100(X₁-0.5)²(X₂-0.25)+ + ε
    Fonction non linéaire, localisée, non additive
    Variables indépendantes
    
    Ce que RLT doit prouver: Muting + look-ahead > méthodes classiques
    """
    
    def __init__(self, random_state=42):
        super().__init__('Scenario2: Régression non-linéaire locale', random_state)
    
    def generate_data(self, n_samples, p_features):
        """Generate non-linear regression data selon spécifications"""
        # Generate independent uniform features [0,1] 
        X = self.np_random.uniform(0, 1, size=(n_samples, p_features))
        
        # True non-linear model: Y = 100(X₁-0.5)²(X₂-0.25)+ + ε
        # (x)+ signifie max(x, 0) - fonction positive seulement
        interaction_term = 100 * (X[:, 0] - 0.5)**2 * np.maximum(X[:, 1] - 0.25, 0)
        
        # Add noise
        epsilon = self.np_random.normal(0, 0.5, n_samples)
        y = interaction_term + epsilon
        
        return X, y, 'regression'


class Scenario3CheckerboardCorrelated(SimulationScenario):
    """
    Scenario 3: Interactions + forte corrélation (cas le plus important)
    
    Y = 2X₅₀X₁₀₀ + 2X₁₅₀X₂₀₀ + ε
    X ~ N(0, Σ)
    Σᵢⱼ = 0.9^|i-j| → forte corrélation
    
    Ce que RLT doit prouver: Détection d'effets joints malgré corrélation
    Échec majeur des méthodes marginales car aucune variable n'a d'effet seule
    """
    
    def __init__(self, random_state=42):
        super().__init__('Scenario3: Interactions avec forte corrélation', random_state)
    
    def generate_data(self, n_samples, p_features):
        """Generate interaction data with strong correlation selon spécifications"""
        # Create correlation matrix Σᵢⱼ = 0.9^|i-j|
        Sigma = np.zeros((p_features, p_features))
        for i in range(p_features):
            for j in range(p_features):
                Sigma[i, j] = 0.9 ** abs(i - j)
        
        # Generate correlated multivariate normal data
        X = self.np_random.multivariate_normal(np.zeros(p_features), Sigma, n_samples)
        
        # True interaction model: Y = 2X₅₀X₁₀₀ + 2X₁₅₀X₂₀₀ + ε
        # Note: indices Python sont 0-based, donc X₅₀ = X[49], etc.
        if p_features >= 200:
            interaction1 = 2 * X[:, 49] * X[:, 99]    # X₅₀ * X₁₀₀
            interaction2 = 2 * X[:, 149] * X[:, 199]  # X₁₅₀ * X₂₀₀
            y = interaction1 + interaction2
        else:
            # Si p < 200, utiliser des variables proportionnelles
            idx1, idx2 = int(0.25 * p_features), int(0.5 * p_features) 
            idx3, idx4 = int(0.75 * p_features), min(int(0.99 * p_features), p_features-1)
            interaction1 = 2 * X[:, idx1] * X[:, idx2]
            interaction2 = 2 * X[:, idx3] * X[:, idx4]
            y = interaction1 + interaction2
        
        # Add noise
        epsilon = self.np_random.normal(0, 0.5, n_samples)
        y = y + epsilon
        
        return X, y, 'regression'


class Scenario4LinearMulticollinearity(SimulationScenario):
    """
    Scenario 4: Modèle linéaire (cas limite)
    
    Y = 2X₅₀ + 2X₁₀₀ + 4X₁₅₀ + ε
    
    Test d'honnêteté scientifique:
    - Le Lasso doit gagner
    - RLT ne doit pas s'effondrer
    """
    
    def __init__(self, random_state=42):
        super().__init__('Scenario4: Modèle linéaire (test honnêteté)', random_state)
    
    def generate_data(self, n_samples, p_features):
        """Generate linear regression data selon spécifications exactes"""
        # Generate independent normal features
        X = self.np_random.randn(n_samples, p_features)
        
        # True linear model: Y = 2X₅₀ + 2X₁₀₀ + 4X₁₅₀ + ε
        beta = np.zeros(p_features)
        
        if p_features >= 150:
            beta[49] = 2.0   # X₅₀ (index 49)
            beta[99] = 2.0   # X₁₀₀ (index 99)
            beta[149] = 4.0  # X₁₅₀ (index 149)
        else:
            # Si p < 150, utiliser des variables proportionnelles
            idx1 = int(0.33 * p_features)
            idx2 = int(0.66 * p_features) 
            idx3 = min(int(0.99 * p_features), p_features-1)
            beta[idx1] = 2.0
            beta[idx2] = 2.0
            beta[idx3] = 4.0
        
        # Generate linear response
        y = X @ beta + self.np_random.normal(0, 0.5, n_samples)
        
        return X, y, 'regression'


class SimulationBenchmark:
    """
    Complete simulation benchmark framework
    Runs all scenarios and aggregates results
    """
    
    def __init__(self, n_repetitions=200, p_values=[200, 500, 1000], 
                 n_train=500, n_test=1000, random_state=42):
        self.n_repetitions = n_repetitions
        self.p_values = p_values
        self.n_train = n_train
        self.n_test = n_test
        self.random_state = random_state
        
        self.scenarios = {
            'Scenario 1: Classification (Independent)': Scenario1ClassificationIndependent(random_state),
            'Scenario 2: Non-linear (Independent)': Scenario2NonlinearIndependent(random_state),
            'Scenario 3: Checkerboard (Correlated)': Scenario3CheckerboardCorrelated(random_state),
            'Scenario 4: Linear (Multicollinear)': Scenario4LinearMulticollinearity(random_state)
        }
        
        self.results = {}
    
    def run_all_scenarios(self, verbose=True):
        """Run all 4 scenarios"""
        if verbose:
            print("\n" + "="*100)
            print("SIMULATION SCENARIOS - Complete Benchmark")
            print("="*100)
        
        for scenario_name, scenario in self.scenarios.items():
            if verbose:
                print(f"\n{scenario_name}")
                print(f"  Repetitions: {self.n_repetitions}")
                print(f"  Dimensions: {self.p_values}")
                print(f"  Train samples: {self.n_train}, Test samples: {self.n_test}")
            
            results = scenario.run_repetitions(
                n_repetitions=self.n_repetitions,
                p_values=self.p_values,
                n_train=self.n_train,
                n_test=self.n_test,
                verbose=verbose
            )
            
            self.results[scenario_name] = results
            
            if verbose:
                print(f"  [OK] {scenario_name} completed")
        
        if verbose:
            print("\n" + "="*100)
            print("[OK] All scenarios completed")
            print("="*100)
    
    def get_summary_dataframe(self):
        """
        Get summary results as DataFrame
        Shows average test error for each scenario and dimension
        """
        summary_data = []
        
        for scenario_name, results in self.results.items():
            for p in self.p_values:
                avg_test_error = np.mean(results[p]['test_errors'])
                std_test_error = np.std(results[p]['test_errors'])
                
                summary_data.append({
                    'Scenario': scenario_name.split(':')[0],
                    'Dimension': p,
                    'Avg Test Error': avg_test_error,
                    'Std Test Error': std_test_error
                })
        
        return pd.DataFrame(summary_data)
    
    def print_summary(self):
        """Print summary of all scenarios"""
        print("\n" + "="*100)
        print("SIMULATION RESULTS SUMMARY")
        print("="*100)
        
        summary_df = self.get_summary_dataframe()
        print("\nAverage Test Error (with std) across repetitions:\n")
        print(summary_df.to_string(index=False))
    
    def plot_results(self, figsize=(15, 10)):
        """Plot scenario results"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for ax, (scenario_name, results) in zip(axes, self.results.items()):
            scenario_short = scenario_name.split(':')[0]
            
            test_errors_by_p = [np.array(results[p]['test_errors']) for p in self.p_values]
            
            # Box plot
            bp = ax.boxplot(test_errors_by_p, labels=[f'p={p}' for p in self.p_values],
                           patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.set_title(scenario_short, fontsize=12, fontweight='bold')
            ax.set_ylabel('Test Error')
            ax.set_xlabel('Dimension')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def export_results(self, filepath):
        """Export results to JSON"""
        import json
        
        export_data = {}
        for scenario_name, results in self.results.items():
            export_data[scenario_name] = {
                str(p): {
                    'train_errors': np.array(results[p]['train_errors']).tolist(),
                    'test_errors': np.array(results[p]['test_errors']).tolist(),
                    'mean_test_error': float(np.mean(results[p]['test_errors'])),
                    'std_test_error': float(np.std(results[p]['test_errors']))
                }
                for p in self.p_values
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath


# Utility function
def generate_simulation_data(scenario_name='all', n_repetitions=200, 
                           p_values=[200, 500, 1000], n_train=500, n_test=1000,
                           random_state=42, verbose=True):
    """
    Quick function to generate simulation data
    
    Parameters:
    -----------
    scenario_name : str
        'all', 'scenario1', 'scenario2', 'scenario3', 'scenario4'
    n_repetitions : int
    p_values : list
    n_train : int
    n_test : int
    random_state : int
    verbose : bool
    
    Returns:
    --------
    results : dict or SimulationBenchmark
    """
    if scenario_name == 'all':
        benchmark = SimulationBenchmark(
            n_repetitions=n_repetitions,
            p_values=p_values,
            n_train=n_train,
            n_test=n_test,
            random_state=random_state
        )
        benchmark.run_all_scenarios(verbose=verbose)
        return benchmark
    
    else:
        scenario_map = {
            'scenario1': Scenario1ClassificationIndependent(random_state),
            'scenario2': Scenario2NonlinearIndependent(random_state),
            'scenario3': Scenario3CheckerboardCorrelated(random_state),
            'scenario4': Scenario4LinearMulticollinearity(random_state)
        }
        
        if scenario_name.lower() not in scenario_map:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = scenario_map[scenario_name.lower()]
        return scenario.run_repetitions(
            n_repetitions=n_repetitions,
            p_values=p_values,
            n_train=n_train,
            n_test=n_test,
            verbose=verbose
        )
