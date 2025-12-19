"""
Explainability (XAI) Module: LIME and SHAP Integration

Provides model-agnostic explanations for RLT and other models
- LIME: Local Interpretable Model-agnostic Explanations
- SHAP: SHapley Additive exPlanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not installed. Install with: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


class LIMEExplainer:
    """
    LIME Explainer for model-agnostic local explanations
    """
    
    def __init__(self, X_train, task_type='regression', feature_names=None, verbose=True):
        """
        Initialize LIME Explainer
        
        Parameters:
        -----------
        X_train : array-like
            Training data for reference
        task_type : str
            'regression' or 'classification'
        feature_names : list
            Feature names (auto-generated if None)
        verbose : bool
            Print progress
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME not installed. Install with: pip install lime")
        
        self.X_train = np.asarray(X_train)
        self.task_type = task_type
        self.verbose = verbose
        
        # Set feature names
        if feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(self.X_train.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Initialize LIME explainer
        if task_type == 'regression':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train,
                feature_names=self.feature_names,
                mode='regression',
                verbose=verbose
            )
        else:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train,
                feature_names=self.feature_names,
                mode='classification',
                class_names=['Class_0', 'Class_1'],
                verbose=verbose
            )
    
    def explain_prediction(self, model, X_instance, prediction=None, num_features=10):
        """
        Explain a single prediction using LIME
        
        Parameters:
        -----------
        model : object
            Predict function or model object
        X_instance : array
            Single instance to explain
        prediction : float
            Prediction for this instance (optional)
        num_features : int
            Number of features to show in explanation
            
        Returns:
        --------
        explanation : LimeExplanation object
            LIME explanation
        """
        X_instance = np.asarray(X_instance).reshape(1, -1)
        
        # Get prediction if not provided
        if prediction is None:
            if hasattr(model, 'predict'):
                prediction = model.predict(X_instance)[0]
            else:
                prediction = model(X_instance)[0]
        
        if self.verbose:
            print(f"[LIME] Explaining instance with prediction: {prediction:.4f}")
        
        # Get explanation
        explanation = self.explainer.explain_instance(
            X_instance[0],
            predict_fn=lambda x: np.array([model.predict(x.reshape(1, -1))[0] 
                                          if hasattr(model, 'predict') 
                                          else model(x.reshape(1, -1))[0] for x in x]),
            num_features=num_features
        )
        
        return explanation
    
    def get_explanation_df(self, explanation):
        """
        Convert LIME explanation to DataFrame
        
        Parameters:
        -----------
        explanation : LimeExplanation object
            LIME explanation
            
        Returns:
        --------
        df : DataFrame
            Feature importance from explanation
        """
        exp_list = explanation.as_list()
        
        features = []
        weights = []
        
        for feat_name, weight in exp_list:
            features.append(feat_name)
            weights.append(weight)
        
        df = pd.DataFrame({
            'Feature': features,
            'Weight': weights
        })
        
        return df
    
    def plot_explanation(self, explanation, figsize=(10, 6)):
        """
        Plot LIME explanation
        
        Parameters:
        -----------
        explanation : LimeExplanation object
            LIME explanation
        figsize : tuple
            Figure size
        """
        fig = plt.figure(figsize=figsize)
        
        # Get explanation data
        exp_list = explanation.as_list()
        features = [x[0][:30] for x in exp_list]  # Truncate long names
        weights = [x[1] for x in exp_list]
        
        # Create bar plot
        colors = ['green' if w > 0 else 'red' for w in weights]
        plt.barh(features, weights, color=colors, alpha=0.7)
        plt.xlabel('Weight')
        plt.title('LIME Explanation - Feature Contributions')
        plt.tight_layout()
        
        return fig


class SHAPExplainer:
    """
    SHAP Explainer for model-agnostic explanations using Shapley values
    """
    
    def __init__(self, X_train, task_type='regression', feature_names=None, 
                 method='kernel', verbose=True):
        """
        Initialize SHAP Explainer
        
        Parameters:
        -----------
        X_train : array-like
            Training data for reference
        task_type : str
            'regression' or 'classification'
        feature_names : list
            Feature names (auto-generated if None)
        method : str
            'kernel', 'sampling', or 'tree'
        verbose : bool
            Print progress
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        self.X_train = np.asarray(X_train)
        self.task_type = task_type
        self.method = method
        self.verbose = verbose
        
        # Set feature names
        if feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(self.X_train.shape[1])]
        else:
            self.feature_names = feature_names
        
        self.explainer = None
        self.shap_values = None
    
    def fit(self, model, X_background=None, sample_size=100):
        """
        Fit SHAP explainer
        
        Parameters:
        -----------
        model : object
            Model or prediction function
        X_background : array
            Background data for SHAP
        sample_size : int
            Sample size for background data
        """
        if self.verbose:
            print(f"[SHAP] Initializing {self.method} explainer...")
        
        # Use subset of training data as background
        if X_background is None:
            if len(self.X_train) > sample_size:
                indices = np.random.choice(len(self.X_train), sample_size, replace=False)
                X_background = self.X_train[indices]
            else:
                X_background = self.X_train
        
        # Create explainer
        if self.method == 'kernel':
            self.explainer = shap.KernelExplainer(
                model=lambda x: np.array([model.predict(x.reshape(1, -1))[0] 
                                         if hasattr(model, 'predict') 
                                         else model(x.reshape(1, -1))[0] for x in x]),
                data=shap.sample(X_background, min(100, len(X_background)))
            )
        elif self.method == 'sampling':
            self.explainer = shap.SamplingExplainer(
                model=lambda x: np.array([model.predict(x.reshape(1, -1))[0] 
                                         if hasattr(model, 'predict') 
                                         else model(x.reshape(1, -1))[0] for x in x]),
                data=X_background
            )
        else:
            raise ValueError(f"Method {self.method} not supported. Use 'kernel' or 'sampling'")
    
    def explain_instances(self, X_instances, model=None):
        """
        Explain multiple instances using SHAP
        
        Parameters:
        -----------
        X_instances : array
            Instances to explain
        model : object
            Model for prediction (if not already fitted)
            
        Returns:
        --------
        shap_values : array
            SHAP values for each instance
        """
        X_instances = np.asarray(X_instances)
        
        if self.explainer is None:
            if model is None:
                raise ValueError("Explainer not fitted. Provide model or call fit() first")
            self.fit(model)
        
        if self.verbose:
            print(f"[SHAP] Computing SHAP values for {len(X_instances)} instances...")
        
        self.shap_values = self.explainer.shap_values(X_instances)
        
        return self.shap_values
    
    def get_feature_importance(self, mean_abs=True):
        """
        Get feature importance from SHAP values
        
        Parameters:
        -----------
        mean_abs : bool
            Use mean absolute SHAP values
            
        Returns:
        --------
        importance_df : DataFrame
            Feature importance ranking
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Call explain_instances() first")
        
        if isinstance(self.shap_values, list):
            # Classification case
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        if mean_abs:
            importances = np.mean(np.abs(shap_vals), axis=0)
        else:
            importances = np.mean(shap_vals, axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': importances
        }).sort_values('SHAP_Importance', ascending=False)
        
        return importance_df
    
    def plot_summary(self, figsize=(10, 8)):
        """
        Plot SHAP summary plot
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Call explain_instances() first")
        
        fig = plt.figure(figsize=figsize)
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        # Create summary plot
        feature_importances = np.mean(np.abs(shap_vals), axis=0)
        indices = np.argsort(feature_importances)[::-1][:15]  # Top 15
        
        plt.barh([self.feature_names[i] for i in indices], 
                [feature_importances[i] for i in indices],
                color='steelblue', alpha=0.7)
        plt.xlabel('Mean |SHAP value|')
        plt.title('SHAP Summary Plot - Feature Importance')
        plt.tight_layout()
        
        return fig
    
    def plot_force(self, instance_idx=0, figsize=(14, 4)):
        """
        Plot SHAP force plot for single instance
        
        Parameters:
        -----------
        instance_idx : int
            Index of instance to plot
        figsize : tuple
            Figure size
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Call explain_instances() first")
        
        fig = plt.figure(figsize=figsize)
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0][instance_idx]
        else:
            shap_vals = self.shap_values[instance_idx]
        
        # Create force-like plot
        positive_features = []
        positive_values = []
        negative_features = []
        negative_values = []
        
        for i, val in enumerate(shap_vals):
            if val > 0:
                positive_features.append(self.feature_names[i][:20])
                positive_values.append(val)
            else:
                negative_features.append(self.feature_names[i][:20])
                negative_values.append(-val)
        
        y_pos_pos = np.arange(len(positive_features))
        y_pos_neg = np.arange(len(negative_features))
        
        plt.subplot(1, 2, 1)
        plt.barh(y_pos_pos, positive_values, color='green', alpha=0.7)
        plt.yticks(y_pos_pos, positive_features)
        plt.xlabel('SHAP Value')
        plt.title('Positive Contributions (Push prediction up)')
        
        plt.subplot(1, 2, 2)
        plt.barh(y_pos_neg, negative_values, color='red', alpha=0.7)
        plt.yticks(y_pos_neg, negative_features)
        plt.xlabel('SHAP Value')
        plt.title('Negative Contributions (Push prediction down)')
        
        plt.tight_layout()
        
        return fig


class XAIAnalyzer:
    """
    Combined LIME and SHAP analyzer for comprehensive model explainability
    """
    
    def __init__(self, model, X_train, y_train, X_test=None, y_test=None,
                 task_type='regression', feature_names=None, verbose=True):
        """
        Initialize XAI Analyzer
        
        Parameters:
        -----------
        model : object
            Trained model
        X_train, y_train : arrays
            Training data
        X_test, y_test : arrays
            Test data (optional)
        task_type : str
            'regression' or 'classification'
        feature_names : list
            Feature names
        verbose : bool
            Print progress
        """
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train).flatten()
        self.X_test = np.asarray(X_test) if X_test is not None else None
        self.y_test = np.asarray(y_test).flatten() if y_test is not None else None
        self.task_type = task_type
        self.verbose = verbose
        
        # Set feature names
        if feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(self.X_train.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Initialize explainers
        self.lime_explainer = None
        self.shap_explainer = None
        
        if LIME_AVAILABLE:
            self.lime_explainer = LIMEExplainer(
                X_train, task_type=task_type, 
                feature_names=self.feature_names,
                verbose=verbose
            )
        
        if SHAP_AVAILABLE:
            self.shap_explainer = SHAPExplainer(
                X_train, task_type=task_type,
                feature_names=self.feature_names,
                method='kernel',
                verbose=verbose
            )
            self.shap_explainer.fit(model, sample_size=min(100, len(X_train)))
    
    def explain_predictions(self, X_instances, num_features=10):
        """
        Explain predictions using both LIME and SHAP
        
        Parameters:
        -----------
        X_instances : array
            Instances to explain
        num_features : int
            Number of features to display
            
        Returns:
        --------
        explanations : dict
            LIME and SHAP explanations
        """
        X_instances = np.asarray(X_instances)
        
        if self.verbose:
            print(f"[XAI] Explaining {len(X_instances)} instances...")
        
        explanations = {
            'lime': None,
            'shap': None
        }
        
        # LIME explanations
        if self.lime_explainer is not None:
            if self.verbose:
                print("  - LIME explanations...")
            # For first instance only
            explanations['lime'] = self.lime_explainer.explain_prediction(
                self.model, X_instances[0], num_features=num_features
            )
        
        # SHAP explanations
        if self.shap_explainer is not None:
            if self.verbose:
                print("  - SHAP explanations...")
            explanations['shap'] = self.shap_explainer.explain_instances(X_instances)
        
        return explanations
    
    def get_global_feature_importance(self, method='shap'):
        """
        Get global feature importance using SHAP
        
        Parameters:
        -----------
        method : str
            'shap' or 'lime'
            
        Returns:
        --------
        importance_df : DataFrame
            Feature importance ranking
        """
        if method == 'shap' and self.shap_explainer is not None:
            if self.X_test is not None:
                self.shap_explainer.explain_instances(self.X_test, model=self.model)
            else:
                self.shap_explainer.explain_instances(self.X_train, model=self.model)
            return self.shap_explainer.get_feature_importance()
        else:
            raise ValueError(f"Method {method} not available or not initialized")
    
    def plot_all_explanations(self, instance_idx=0):
        """
        Plot comprehensive explanations for a single instance
        
        Parameters:
        -----------
        instance_idx : int
            Index of instance to explain
        """
        figs = {}
        
        X_inst = self.X_test[instance_idx:instance_idx+1] if self.X_test is not None else self.X_train[instance_idx:instance_idx+1]
        
        # Get prediction
        if hasattr(self.model, 'predict'):
            pred = self.model.predict(X_inst)[0]
        else:
            pred = self.model(X_inst)[0]
        
        if self.verbose:
            print(f"[XAI] Plotting explanations for instance {instance_idx} (prediction: {pred:.4f})")
        
        # SHAP plots
        if self.shap_explainer is not None:
            try:
                figs['shap_summary'] = self.shap_explainer.plot_summary()
                figs['shap_force'] = self.shap_explainer.plot_force(instance_idx=instance_idx)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not plot SHAP visualizations: {e}")
        
        return figs


def compare_xai_methods(model, X_instance, y_true, 
                       lime_explainer=None, shap_explainer=None,
                       task_type='regression', verbose=True):
    """
    Compare LIME and SHAP explanations for a single prediction
    
    Parameters:
    -----------
    model : object
        Trained model
    X_instance : array
        Single instance to explain
    y_true : float
        True value
    lime_explainer : LIMEExplainer
    shap_explainer : SHAPExplainer
    task_type : str
        'regression' or 'classification'
    verbose : bool
        Print results
        
    Returns:
    --------
    comparison : dict
        Comparison of LIME and SHAP
    """
    X_instance = np.asarray(X_instance).reshape(1, -1)
    
    # Get prediction
    if hasattr(model, 'predict'):
        prediction = model.predict(X_instance)[0]
    else:
        prediction = model(X_instance)[0]
    
    comparison = {
        'instance': X_instance[0],
        'prediction': prediction,
        'y_true': y_true,
        'error': abs(prediction - y_true) if task_type == 'regression' else int(prediction != y_true),
        'lime_explanation': None,
        'shap_explanation': None
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print("XAI COMPARISON - LIME vs SHAP")
        print("=" * 80)
        print(f"Instance: {X_instance[0][:5]}... (showing first 5 features)")
        print(f"Prediction: {prediction:.4f} | True Value: {y_true:.4f}" if task_type == 'regression' else f"Prediction: {prediction} | True: {y_true}")
        print(f"Error: {comparison['error']:.4f}")
    
    # Get LIME explanation
    if lime_explainer is not None:
        try:
            lime_exp = lime_explainer.explain_prediction(model, X_instance[0], prediction, num_features=5)
            lime_df = lime_explainer.get_explanation_df(lime_exp)
            comparison['lime_explanation'] = lime_df
            
            if verbose:
                print("\nLIME Top Features:")
                print(lime_df.to_string(index=False))
        except Exception as e:
            if verbose:
                print(f"LIME explanation failed: {e}")
    
    # Get SHAP explanation
    if shap_explainer is not None:
        try:
            shap_vals = shap_explainer.explain_instances(X_instance, model=model)
            comparison['shap_explanation'] = shap_vals
            
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            
            if verbose:
                print("\nSHAP Top Features:")
                top_idx = np.argsort(np.abs(shap_vals[0]))[::-1][:5]
                for idx in top_idx:
                    print(f"  Feature_{idx}: {shap_vals[0][idx]:.4f}")
        except Exception as e:
            if verbose:
                print(f"SHAP explanation failed: {e}")
    
    if verbose:
        print("=" * 80)
    
    return comparison
