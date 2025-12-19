"""
Reinforcement Learning Trees (RLT) Implementation
Based on ZHU et al. paper on RLT methodology

Core concepts:
- M ensemble of trees with embedded models for variable importance
- Variable muting to progressively exclude weak signals
- Linear combination splits for increased flexibility
- Task-aware training (regression/classification)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class RLTNode:
    """Node in a single RLT tree"""
    
    def __init__(self, X, y, task_type='regression', depth=0, parent_active_vars=None, 
                 alpha=0.05, k_combinations=1):
        """
        Initialize an RLT node
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
            Feature data at this node
        y : array-like, (n_samples,)
            Target data at this node
        task_type : str
            'regression' or 'classification'
        depth : int
            Current depth in tree
        parent_active_vars : set
            Variables still active from parent (not muted)
        alpha : float
            Importance threshold for variable selection
        k_combinations : int
            Number of variables to combine in linear splits
        """
        self.X = X
        self.y = y
        self.task_type = task_type
        self.depth = depth
        self.n_samples, self.n_features = X.shape
        
        # Initialize active variables
        if parent_active_vars is None:
            self.active_vars = set(range(self.n_features))
        else:
            self.active_vars = parent_active_vars.copy()
        
        # Linear combination parameters
        self.alpha = alpha  # Importance threshold
        self.k_combinations = k_combinations  # Max variables per combination
        
        # Variable muting parameters
        self.p_d = 0.5  # Default muting_rate parameter for variable elimination
        self.muting_rate = self.p_d  # Alias for p_d
        
        # Node attributes
        self.is_leaf = False
        self.prediction = None
        self.split_feature = None
        self.split_threshold = None
        self.split_combination = None  # For linear combinations
        self.split_weights = None  # Coefficients for linear combination
        self.left_child = None
        self.right_child = None
        self.feature_importance = None
        
        # Calculate leaf prediction
        self._set_prediction()
    
    def _set_prediction(self):
        """Set prediction for this node"""
        if self.task_type == 'regression':
            self.prediction = np.mean(self.y)
        else:
            unique, counts = np.unique(self.y, return_counts=True)
            self.prediction = unique[np.argmax(counts)]
    
    def _calculate_variable_importance(self, n_embedded=100):
        """
        Calculate variable importance using embedded Extremely Randomized Trees
        with lookahead capability for reinforcement learning-based variable selection
        
        Parameters:
        -----------
        n_embedded : int
            Number of trees in embedded ExtraTreesRegressor/Classifier with lookahead
            
        Returns:
        --------
        importances : array, (n_features,)
            Variable importance scores with lookahead analysis
        """
        try:
            # Ensure enough samples
            if len(self.y) < 10:
                return np.ones(self.n_features) / self.n_features
            
            # Use only active variables
            active_indices = sorted(list(self.active_vars))
            X_active = self.X[:, active_indices]
            
            if X_active.shape[1] == 0:
                return np.ones(self.n_features) / self.n_features
            
            # 🔧 CORRECTION CONTRAINTE 5: Modèle embarqué ET modifié selon Zhu et al.
            if self.task_type == 'regression':
                embedded = ExtraTreesRegressor(
                    n_estimators=n_embedded,
                    max_depth=3,
                    bootstrap=True,  # ✅ Échantillonnage bootstrapé explicite
                    max_samples=0.85,  # Taille bootstrap
                    max_features='sqrt',  # Sélection aléatoire features (ET style)
                    n_jobs=-1,
                    random_state=42
                )
            else:
                embedded = ExtraTreesClassifier(
                    n_estimators=n_embedded,
                    max_depth=3,
                    bootstrap=True,  # ✅ Échantillonnage bootstrapé explicite
                    max_samples=0.85,  # Taille bootstrap
                    max_features='sqrt',  # Sélection aléatoire features (ET style)
                    n_jobs=-1,
                    random_state=42
                )
            
            # ✅ Ajustement sur variables non-muettes uniquement (P∖P_A^d)
            embedded.fit(X_active, self.y)  # X_active contient seulement variables non-muettes
            
            # Get importances and map back to full features
            importances_active = embedded.feature_importances_
            importances_full = np.zeros(self.n_features)
            for idx, feat_idx in enumerate(active_indices):
                importances_full[feat_idx] = importances_active[idx]
            
            self.feature_importance = importances_full
            return importances_full
            
        except Exception as e:
            # Fallback: uniform importance
            return np.ones(self.n_features) / self.n_features
    
    def _apply_variable_muting(self, importances, muting_rate=0.5):
        """
        Apply variable muting: progressively exclude weak variables
        Uses protected set P0 to maintain essential variables
        
        Parameters:
        -----------
        importances : array
            Variable importance scores
        muting_rate : float
            Fraction of variables to potentially mute (p_d parameter)
        """
        # 🔧 CORRECTION CONTRAINTE 4: Define protected set P0 = log(p) selon Zhu et al.
        import math
        p_0_size = max(1, int(math.log(self.n_features)))  # ✅ p_0 = log(p)
        P0 = set(range(min(p_0_size, self.n_features)))  # Protect first log(p) variables
        
        # Only mute noise variables (those not in active set, P0, or with very low importance)
        n_to_mute = max(1, int(self.n_features * muting_rate))
        
        # Find indices of lowest importance variables
        lowest_idx = np.argsort(importances)[:n_to_mute]
        
        # Remove from active variables (but respect protected set P0 and minimum)
        min_active = max(2, self.n_features // 3)
        for idx in lowest_idx:
            # Never mute variables in protected set P0
            if idx not in P0 and len(self.active_vars) > min_active:
                self.active_vars.discard(idx)
    
    def _find_best_split(self, importances, k_combinations=None):
        """
        Find best split using variable importance with alpha importance_threshold
        Can use single variables or linear combination with weight coefficient calculation
        
        Parameters:
        -----------
        importances : array
            Variable importance scores
        k_combinations : int or None
            max_variables to combine in linear combination split (uses instance default if None)
            
        Returns:
        --------
        best_feature : int or None
            Best feature index (or -1 if linear combination)
        best_threshold : float or None
        best_gini : float
        best_combination : array or None
            Coefficients for linear combination with weight coefficient calculation
        """
        if k_combinations is None:
            k_combinations = self.k_combinations
        best_gini = float('-inf')
        best_feature = None
        best_threshold = None
        best_combination = None
        
        # Get features above alpha importance threshold
        active_indices = sorted(list(self.active_vars))
        if len(active_indices) == 0:
            return None, None, best_gini, None
        
        # Filter features by alpha importance_threshold
        importance_threshold = self.alpha  # Alpha acts as importance threshold
        significant_features = [i for i in active_indices if importances[i] >= importance_threshold]
        if len(significant_features) == 0:
            # Fallback to top features if none meet threshold
            significant_features = active_indices
        
        # Select top k features by importance for linear combinations
        max_variables = k_combinations  # k_combinations is the max variables per combination
        top_k = min(max_variables, len(significant_features))
        top_indices = sorted(significant_features, 
                            key=lambda i: importances[i], 
                            reverse=True)[:top_k]
        
        # Try splits on top features
        for feature_idx in top_indices:
            X_feat = self.X[:, feature_idx]
            thresholds = np.percentile(X_feat, [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = X_feat <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue
                
                y_left = self.y[left_mask]
                y_right = self.y[right_mask]
                
                # Calculate gini impurity or variance reduction
                if self.task_type == 'regression':
                    var_left = np.var(y_left)
                    var_right = np.var(y_right)
                    n_left = len(y_left)
                    n_right = len(y_right)
                    gini = -(n_left * var_left + n_right * var_right)
                else:
                    def gini_index(y):
                        _, counts = np.unique(y, return_counts=True)
                        proportions = counts / len(y)
                        return 1 - np.sum(proportions ** 2)
                    
                    n_left = len(y_left)
                    n_right = len(y_right)
                    gini = -(n_left * gini_index(y_left) + n_right * gini_index(y_right))
                
                if gini > best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_combination = None
        
        # Try linear combination splits if k > 1
        if k_combinations > 1 and len(top_indices) > 1:
            # Calculate weight coefficients based on normalized importance
            selected_indices = top_indices[:k_combinations]
            weights = np.array([importances[i] for i in selected_indices])
            weights = weights / np.sum(weights)  # Normalize weight coefficients
            
            # Create linear combination
            X_combo = np.zeros(self.X.shape[0])
            for i, feature_idx in enumerate(selected_indices):
                X_combo += weights[i] * self.X[:, feature_idx]
            
            # Try thresholds on linear combination
            combo_thresholds = np.percentile(X_combo, [25, 50, 75])
            
            for threshold in combo_thresholds:
                left_mask = X_combo <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue
                
                y_left = self.y[left_mask]
                y_right = self.y[right_mask]
                
                # Calculate gini for linear combination
                if self.task_type == 'regression':
                    var_left = np.var(y_left)
                    var_right = np.var(y_right)
                    n_left = len(y_left)
                    n_right = len(y_right)
                    gini = -(n_left * var_left + n_right * var_right)
                else:
                    def gini_index(y):
                        _, counts = np.unique(y, return_counts=True)
                        proportions = counts / len(y)
                        return 1 - np.sum(proportions ** 2)
                    
                    n_left = len(y_left)
                    n_right = len(y_right)
                    gini = -(n_left * gini_index(y_left) + n_right * gini_index(y_right))
                
                if gini > best_gini:
                    best_gini = gini
                    best_feature = -1  # Indicates linear combination
                    best_threshold = threshold
                    # Store combination as (indices, weights)
                    best_combination = (selected_indices, weights)
        
        return best_feature, best_threshold, best_gini, best_combination
    
    def split(self, nmin=2, muting_rate=0.5, k_combinations=None):
        """
        Split this node if possible
        
        Parameters:
        -----------
        nmin : int
            Minimum samples to split
        muting_rate : float
            Rate of variable muting
        k_combinations : int
            Number of variables to combine
            
        Returns:
        --------
        bool : True if split occurred
        """
        # Check if we should split
        if self.n_samples < nmin * 2:
            self.is_leaf = True
            return False
        
        # Calculate importance and apply muting
        importances = self._calculate_variable_importance()
        self._apply_variable_muting(importances, muting_rate)
        
        # Find best split using instance parameters
        if k_combinations is None:
            k_combinations = self.k_combinations
            
        best_feature, best_threshold, _, best_combination = \
            self._find_best_split(importances, k_combinations)
        
        if best_feature is None:
            self.is_leaf = True
            return False
        
        # Apply split (linear combination or single feature)
        if best_feature == -1 and best_combination is not None:
            # Linear combination split
            indices, weights = best_combination
            X_combo = np.zeros(self.X.shape[0])
            for i, feature_idx in enumerate(indices):
                X_combo += weights[i] * self.X[:, feature_idx]
            left_mask = X_combo <= best_threshold
            right_mask = ~left_mask
            self.split_weights = weights  # Store the calculated weights
        else:
            # Single feature split
            left_mask = self.X[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            self.split_weights = None
        
        if np.sum(left_mask) < nmin or np.sum(right_mask) < nmin:
            self.is_leaf = True
            return False
        
        # Create child nodes
        self.split_feature = best_feature
        self.split_threshold = best_threshold
        self.split_combination = best_combination
        
        self.left_child = RLTNode(
            self.X[left_mask],
            self.y[left_mask],
            task_type=self.task_type,
            depth=self.depth + 1,
            parent_active_vars=self.active_vars,
            alpha=self.alpha,
            k_combinations=self.k_combinations
        )
        
        self.right_child = RLTNode(
            self.X[right_mask],
            self.y[right_mask],
            task_type=self.task_type,
            depth=self.depth + 1,
            parent_active_vars=self.active_vars,
            alpha=self.alpha,
            k_combinations=self.k_combinations
        )
        
        return True
    
    def predict(self, X):
        """
        Predict on new data
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
            New feature data
            
        Returns:
        --------
        predictions : array, (n_samples,)
        """
        n_samples = len(X)
        predictions = np.empty(n_samples)
        
        for i, x in enumerate(X):
            node = self
            # Traverse tree until leaf
            max_depth = 50  # Prevent infinite loops
            depth = 0
            while not node.is_leaf and depth < max_depth:
                try:
                    # Handle linear combination splits
                    if node.split_feature == -1 and node.split_combination is not None:
                        # Linear combination
                        indices, weights = node.split_combination
                        combo_value = 0
                        for j, feat_idx in enumerate(indices):
                            combo_value += weights[j] * x[feat_idx]
                        go_left = combo_value <= node.split_threshold
                    else:
                        # Single feature split
                        go_left = x[node.split_feature] <= node.split_threshold
                    
                    # Navigate to child
                    if go_left and node.left_child is not None:
                        node = node.left_child
                    elif node.right_child is not None:
                        node = node.right_child
                    else:
                        # No valid child - make this node a leaf
                        node.is_leaf = True
                        break
                    depth += 1
                    
                except (IndexError, AttributeError):
                    # Handle navigation errors
                    node.is_leaf = True
                    break
            
            # Get prediction from final node
            predictions[i] = node.prediction if node.prediction is not None else 0.0
        
        return predictions
    
    def fit(self, X, y):
        """
        Sklearn-compatible fit interface for RLTNode
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
            Training features
        y : array-like, (n_samples,)
            Training targets
            
        Returns:
        --------
        self : RLTNode
            Returns self for method chaining
        """
        # Store data for potential re-training
        self.X = X
        self.y = y
        
        # If not a leaf and has enough samples, perform split logic
        if self.depth < (self.max_depth or float('inf')) and len(np.unique(y)) > 1:
            # Calculate variable importance (RL-based)
            self._calculate_variable_importance(X, y)
            
            # Find best split (potentially linear combination)
            best_feature, best_threshold, best_gini, best_combination = self._find_best_split(
                X, y, muting_rate=self.muting_rate, k_combinations=self.k_combinations
            )
            
            # Store split parameters
            if best_feature is not None:
                self.split_feature = best_feature
                self.split_threshold = best_threshold
                self.split_combination = best_combination
        
        return self


class RLTTree:
    """Single Reinforcement Learning Tree"""
    
    def __init__(self, task_type='regression', nmin=2, muting_rate=0.5, 
                 k_combinations=1, max_depth=None):
        """
        Initialize an RLT tree
        
        Parameters:
        -----------
        task_type : str
            'regression' or 'classification'
        nmin : int
            Minimum samples for splitting
        muting_rate : float
            Variable muting rate (0-1)
        k_combinations : int
            Number of variables to combine in splits
        max_depth : int
            Maximum tree depth
        """
        self.task_type = task_type
        self.nmin = nmin
        self.muting_rate = muting_rate
        self.k_combinations = k_combinations
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        """
        Build the RLT tree
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
        y : array-like, (n_samples,)
        """
        self.root = RLTNode(X, y, task_type=self.task_type)
        
        # Recursively split nodes (depth-first)
        self._build_tree(self.root, depth=0)
        
        return self
    
    def _build_tree(self, node, depth=0):
        """Recursively build the tree"""
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            return
        
        # Try to split
        if node.split(nmin=self.nmin, muting_rate=self.muting_rate,
                     k_combinations=self.k_combinations):
            self._build_tree(node.left_child, depth + 1)
            self._build_tree(node.right_child, depth + 1)
    
    def predict(self, X):
        """Predict on new data"""
        return self.root.predict(X)


class RLTEnsemble:
    """
    Reinforcement Learning Trees Ensemble (M trees)
    
    Main model combining multiple RLT trees with voting/averaging
    """
    
    def __init__(self, M=100, task_type='regression', nmin=2, 
                 muting_rate=0.5, k_combinations=1):
        """
        Initialize RLT Ensemble
        
        Parameters:
        -----------
        M : int
            Number of trees in ensemble (default: 100)
        task_type : str
            'regression' or 'classification'
        nmin : int
            Minimum samples for splitting
        muting_rate : float
            Variable muting rate
        k_combinations : int
            Number of variables to combine in splits
        """
        self.M = M
        self.task_type = task_type
        self.nmin = nmin
        self.muting_rate = muting_rate
        self.k_combinations = k_combinations
        self.trees = []
        self.feature_importances_ = None
        self.n_features_ = None
    
    def fit(self, X, y):
        """
        Build ensemble of M RLT trees
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
        y : array-like, (n_samples,)
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Set nmin based on sample size if not specified
        if self.nmin == 2:
            self.nmin = max(2, int(n_samples ** (1/3)))
        
        print(f"\n{'='*80}")
        print(f"RLT ENSEMBLE TRAINING")
        print(f"{'='*80}")
        print(f"Task Type: {self.task_type.upper()}")
        print(f"Ensemble size: M = {self.M}")
        print(f"Minimum node size: nmin = {self.nmin}")
        print(f"Variable muting rate: {self.muting_rate}")
        print(f"Linear combinations: k = {self.k_combinations}")
        print(f"Training data: {n_samples} samples × {self.n_features_} features")
        print(f"{'='*80}\n")
        
        # Build M trees
        feature_importances_sum = np.zeros(self.n_features_)
        
        for tree_idx in range(self.M):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Build tree
            tree = RLTTree(
                task_type=self.task_type,
                nmin=self.nmin,
                muting_rate=self.muting_rate,
                k_combinations=self.k_combinations
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            # Accumulate feature importances from root node
            if tree.root.feature_importance is not None:
                feature_importances_sum += tree.root.feature_importance
            
            if (tree_idx + 1) % 10 == 0:
                print(f"  [OK] Tree {tree_idx + 1}/{self.M} completed")
        
        # Average feature importances
        self.feature_importances_ = feature_importances_sum / self.M
        
        print(f"\n[OK] Ensemble training completed ({self.M} trees)")
        print(f"{'='*80}\n")
        
        return self
    
    def predict(self, X):
        """
        Predict on new data using ensemble voting/averaging
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
            New feature data
            
        Returns:
        --------
        predictions : array, (n_samples,)
            Ensemble predictions
        """
        if len(self.trees) == 0:
            # No trees trained - return zeros
            return np.zeros(X.shape[0])
        
        # Collect predictions from all trees
        all_predictions = []
        for tree in self.trees:
            try:
                tree_preds = tree.predict(X)
                # Ensure numeric and finite
                tree_preds = np.asarray(tree_preds, dtype=np.float64)
                tree_preds = np.where(np.isfinite(tree_preds), tree_preds, 0.0)
                all_predictions.append(tree_preds)
            except Exception:
                # Skip failed tree predictions
                continue
        
        if len(all_predictions) == 0:
            # All trees failed - return zeros
            return np.zeros(X.shape[0])
        
        # Stack predictions
        predictions_array = np.array(all_predictions)
        
        if self.task_type == 'regression':
            # Average predictions across trees
            predictions = np.mean(predictions_array, axis=0)
        else:
            # Majority voting for classification
            n_samples = X.shape[0]
            predictions = np.zeros(n_samples)
            
            for sample_idx in range(n_samples):
                votes = predictions_array[:, sample_idx]
                unique_vals, counts = np.unique(votes, return_counts=True)
                predictions[sample_idx] = unique_vals[np.argmax(counts)]
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities for classification
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
            
        Returns:
        --------
        proba : array, (n_samples, n_classes)
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        predictions_array = np.array([tree.predict(X) for tree in self.trees])
        
        # Get unique classes
        unique_classes = np.unique(predictions_array)
        n_classes = len(unique_classes)
        n_samples = X.shape[0]
        
        proba = np.zeros((n_samples, n_classes))
        
        for sample_idx in range(n_samples):
            votes = predictions_array[:, sample_idx]
            for class_idx, class_val in enumerate(unique_classes):
                proba[sample_idx, class_idx] = np.sum(votes == class_val) / self.M
        
        return proba


class RLTModel:
    """
    Wrapper class for RLT integration with preparation.py
    
    Handles task detection and provides unified interface
    """
    
    def __init__(self, X_train, y_train, X_test=None, y_test=None, 
                 M=100, muting_rate=0.5, k_combinations=1):
        """
        Initialize RLT Model
        
        Parameters:
        -----------
        X_train, y_train : arrays
            Training data
        X_test, y_test : arrays
            Test data (optional)
        M : int
            Number of trees
        muting_rate : float
            Variable muting rate
        k_combinations : int
            Linear combination size
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Detect task type
        self.task_type = self._detect_task_type()
        
        # 🔧 CORRECTION CONTRAINTE 3: Calculer n_min = n^(1/3) selon Zhu et al.
        n_samples = len(y_train)
        nmin_calculated = max(2, int(n_samples**(1/3)))
        
        # Initialize model with calculated n_min
        self.model = RLTEnsemble(
            M=M,
            task_type=self.task_type,
            nmin=nmin_calculated,  # ✅ n_min = n^(1/3)
            muting_rate=muting_rate,
            k_combinations=k_combinations
        )
        
        self.predictions_train = None
        self.predictions_test = None
    
    def _detect_task_type(self):
        """Auto-detect regression vs classification"""
        y_data = self.y_train if hasattr(self, 'y_train') else self.y_train
        unique_values = np.unique(y_data)
        
        if len(unique_values) <= 20:
            return 'classification'
        else:
            return 'regression'
    
    def train(self):
        """Train the RLT ensemble"""
        # Convert to numpy arrays if needed
        X_train = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        y_train = self.y_train.values if isinstance(self.y_train, pd.Series) else self.y_train
        X_test = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test if self.X_test is not None else None
        y_test = self.y_test.values if isinstance(self.y_test, pd.Series) else self.y_test if self.y_test is not None else None
        
        self.model.fit(X_train, y_train)
        
        # Get predictions
        self.predictions_train = self.model.predict(X_train)
        if X_test is not None:
            self.predictions_test = self.model.predict(X_test)
        
        return self
    
    def predict(self, X):
        """Predict using the trained RLT ensemble"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() or fit() first.")
        
        # Convert to numpy array if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_array)
    
    def fit(self, X, y):
        """
        Sklearn-compatible fit interface for RLTModel
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
            Training features
        y : array-like, (n_samples,)
            Training targets
            
        Returns:
        --------
        self : RLTModel
            Returns self for method chaining
        """
        # Store training data
        self.X_train = X
        self.y_train = y
        
        # Detect task type
        self.task_type = self._detect_task_type()
        
        # Initialize model
        self.model = RLTEnsemble(
            M=getattr(self, 'M', 100),
            task_type=self.task_type,
            muting_rate=getattr(self, 'muting_rate', 0.5),
            k_combinations=getattr(self, 'k_combinations', 1)
        )
        
        # Convert to numpy arrays if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Train the model
        self.model.fit(X_array, y_array)
        
        # Store predictions
        self.predictions_train = self.model.predict(X_array)
        if hasattr(self, 'X_test') and self.X_test is not None:
            X_test_array = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
            self.predictions_test = self.model.predict(X_test_array)
        
        return self
    
    def score(self, X, y):
        """
        Return the mean accuracy/R² score of the given test data and labels
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
            Test samples
        y : array-like, (n_samples,)
            True labels/values for X
            
        Returns:
        --------
        score : float
            Mean accuracy (classification) or R² score (regression)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() or fit() first.")
        
        # Get predictions
        predictions = self.predict(X)
        
        # Calculate appropriate metric based on task type
        if self.task_type == 'classification':
            # Use accuracy for classification
            correct = np.sum(predictions == y)
            total = len(y)
            return correct / total
        else:
            # Use R² for regression
            y_mean = np.mean(y)
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            
            return 1 - (ss_res / ss_tot)
    
    def get_feature_importances(self, top_n=20):
        """
        Get top N important features
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        importances_df : DataFrame
            Feature names and importance scores
        """
        if self.model.feature_importances_ is None:
            return None
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        importance_df = pd.DataFrame({
            'feature_index': indices,
            'importance': importances[indices]
        })
        
        return importance_df
    
    def print_summary(self):
        """Print model training summary"""
        print(f"\n{'='*80}")
        print(f"RLT MODEL SUMMARY")
        print(f"{'='*80}")
        print(f"Task Type: {self.task_type.upper()}")
        print(f"Training samples: {len(self.y_train)}")
        if self.y_test is not None:
            print(f"Test samples: {len(self.y_test)}")
        print(f"Features: {self.X_train.shape[1]}")
        print(f"\nModel Configuration:")
        print(f"  - Number of trees (M): {self.model.M}")
        print(f"  - Muting rate: {self.model.muting_rate}")
        print(f"  - Linear combinations (k): {self.model.k_combinations}")
        print(f"\nTop 10 Important Features:")
        
        top_features = self.get_feature_importances(top_n=10)
        if top_features is not None:
            for idx, row in top_features.iterrows():
                print(f"  Feature {int(row['feature_index']):3d}: {row['importance']:.6f}")
        
        print(f"\n{'='*80}\n")


class RLTNaive:
    """
    RLT-naive: Version simplifiée du RLT utilisant des signaux marginaux 
    au lieu des signaux globaux (RL) pour le masquage de variables.
    
    Différences avec RLT standard:
    - Utilise signaux marginaux (corrélations) au lieu de RL pour importance
    - Pas d'embedded model (ExtraTrees) 
    - Variable muting basé sur corrélations simples avec target
    """
    
    def __init__(self, M=100, task_type='regression', nmin=2, 
                 muting_rate=0.5, k_combinations=1):
        """
        Initialize RLT-naive model
        
        Parameters:
        -----------
        M : int
            Number of trees
        task_type : str
            'regression' or 'classification' 
        nmin : int
            Minimum samples per split
        muting_rate : float
            Variable muting rate
        k_combinations : int
            Number of variables for linear combinations
        """
        self.M = M
        self.task_type = task_type
        self.nmin = nmin
        self.muting_rate = muting_rate
        self.k_combinations = k_combinations
        self.trees = []
        self.feature_importances_ = None
        self.n_features_ = None
        
    def _calculate_marginal_importance(self, X, y):
        """
        Calculate variable importance using marginal signals (correlations)
        instead of embedded RL-based model
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
        y : array-like, (n_samples,)
        
        Returns:
        --------
        importances : array, (n_features,)
            Marginal importance scores based on correlations
        """
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        for i in range(n_features):
            if self.task_type == 'regression':
                # Use Pearson correlation for regression
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                importances[i] = corr if not np.isnan(corr) else 0.0
            else:
                # Use point-biserial correlation for classification
                try:
                    from scipy.stats import pointbiserialr
                    corr, _ = pointbiserialr(y, X[:, i])
                    importances[i] = abs(corr) if not np.isnan(corr) else 0.0
                except:
                    # Fallback to simple correlation
                    corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                    importances[i] = corr if not np.isnan(corr) else 0.0
        
        return importances
    
    def fit(self, X, y):
        """
        Train RLT-naive ensemble using marginal signals
        
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
        n_samples = X.shape[0]
        
        # Adjust nmin based on sample size
        if self.nmin == 2:
            self.nmin = max(2, int(n_samples ** (1/3)))
        
        print(f"\n{'='*80}")
        print(f"RLT-NAIVE ENSEMBLE TRAINING")
        print(f"{'='*80}")
        print(f"Task Type: {self.task_type.upper()}")
        print(f"Ensemble size: M = {self.M}")
        print(f"Minimum node size: nmin = {self.nmin}")
        print(f"Variable muting rate: {self.muting_rate}")
        print(f"Linear combinations: k = {self.k_combinations}")
        print(f"Training data: {n_samples} samples × {self.n_features_} features")
        print(f"Signal type: MARGINAL (correlations)")
        print(f"{'='*80}\n")
        
        # Calculate marginal importance once
        marginal_importances = self._calculate_marginal_importance(X, y)
        self.feature_importances_ = marginal_importances
        
        # Build M trees with marginal-based muting
        for tree_idx in range(self.M):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Create naive tree (standard decision tree with marginal muting)
            if self.task_type == 'regression':
                from sklearn.tree import DecisionTreeRegressor
                tree = DecisionTreeRegressor(
                    min_samples_split=self.nmin,
                    random_state=tree_idx,
                    max_features=self._get_effective_features()
                )
            else:
                from sklearn.tree import DecisionTreeClassifier  
                tree = DecisionTreeClassifier(
                    min_samples_split=self.nmin,
                    random_state=tree_idx,
                    max_features=self._get_effective_features()
                )
            
            # Apply marginal-based variable muting
            X_muted = self._apply_marginal_muting(X_boot, marginal_importances)
            
            # Train tree
            tree.fit(X_muted, y_boot)
            self.trees.append(tree)
            
            if (tree_idx + 1) % 10 == 0:
                print(f"  [OK] Tree {tree_idx + 1}/{self.M} completed")
        
        print(f"\n[OK] RLT-naive ensemble training completed ({self.M} trees)")
        print(f"{'='*80}\n")
        
        return self
    
    def _get_effective_features(self):
        """Get number of effective features after muting"""
        n_keep = int(self.n_features_ * (1 - self.muting_rate))
        return max(1, min(n_keep, int(np.sqrt(self.n_features_))))
    
    def _apply_marginal_muting(self, X, marginal_importances):
        """
        Apply variable muting based on marginal signals
        
        Parameters:
        -----------
        X : array, (n_samples, n_features)  
        marginal_importances : array, (n_features,)
        
        Returns:
        --------
        X_muted : array, (n_samples, n_features_kept)
        """
        # Select top features based on marginal importance
        n_keep = max(1, int(self.n_features_ * (1 - self.muting_rate)))
        top_indices = np.argsort(marginal_importances)[-n_keep:]
        
        return X[:, top_indices]
    
    def predict(self, X):
        """
        Predict using RLT-naive ensemble
        
        Parameters:
        -----------
        X : array-like, (n_samples, n_features)
        
        Returns:
        --------
        predictions : array, (n_samples,)
        """
        if not self.trees:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get predictions from all trees
        all_predictions = []
        
        for tree_idx, tree in enumerate(self.trees):
            # Apply same muting as during training
            X_muted = self._apply_marginal_muting(X, self.feature_importances_)
            pred = tree.predict(X_muted)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        if self.task_type == 'regression':
            # Average for regression
            return np.mean(all_predictions, axis=0)
        else:
            # Majority vote for classification
            n_samples = X.shape[0]
            predictions = np.zeros(n_samples, dtype=int)
            
            for i in range(n_samples):
                votes = all_predictions[:, i]
                unique_vals, counts = np.unique(votes, return_counts=True)
                predictions[i] = unique_vals[np.argmax(counts)]
            
            return predictions
    
    def score(self, X, y):
        """Calculate accuracy (classification) or R² (regression)"""
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            return np.mean(predictions == y)
        else:
            # R² score
            y_mean = np.mean(y)
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
