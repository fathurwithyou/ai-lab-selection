from collections import Counter

import numpy as np

from ..base_estimator import BaseEstimator
from .classifier import ClassifierMixin
from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier(ClassifierMixin, BaseEstimator):
    """
    Random Forest Classifier using ensemble of decision trees.

    Combines multiple decision trees trained on bootstrap samples
    with random feature subsets to reduce overfitting and improve
    generalization.
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        """
        Initialize Random Forest Classifier.

        Parameters:
        n_estimators (int): Number of trees in the forest
        criterion (str): Splitting criterion ('gini' or 'entropy')
        max_depth (int): Maximum depth of the trees
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required at a leaf node
        min_impurity_decrease (float): Minimum impurity decrease for a split
        max_features (str or int): Number of features to consider when looking for best split
                                 - 'sqrt': sqrt(n_features)
                                 - 'log2': log2(n_features)
                                 - int: exact number of features
                                 - None: all features
        bootstrap (bool): Whether to bootstrap samples when building trees
        random_state (int): Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []
        self.feature_indices = []
        self.classes = None
        self.n_classes = None
        self.n_features = None

        if random_state is not None:
            np.random.seed(random_state)

    def _get_max_features(self, n_features):
        """Calculate the number of features to use for each split."""
        if self.max_features is None:
            return n_features
        elif self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample of the data."""
        n_samples = X.shape[0]
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[bootstrap_indices], y[bootstrap_indices]

    def _random_feature_subset(self, n_features):
        """Select a random subset of features."""
        max_features = self._get_max_features(n_features)
        return np.random.choice(n_features, size=max_features, replace=False)

    def fit(self, X, y):
        """
        Fit the random forest classifier.

        Parameters:
        X (array-like): Training features of shape (n_samples, n_features)
        y (array-like): Target labels of shape (n_samples,)

        Returns:
        self: Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        self.trees = []
        self.feature_indices = []

        for i in range(self.n_estimators):
            tree_random_state = None
            if self.random_state is not None:
                tree_random_state = self.random_state + i

            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            feature_subset = self._random_feature_subset(self.n_features)
            self.feature_indices.append(feature_subset)

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=tree_random_state,
            )

            X_subset = X_sample[:, feature_subset]
            tree.fit(X_subset, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """
        Predict class labels using majority voting.

        Parameters:
        X (array-like): Features of shape (n_samples, n_features)

        Returns:
        predictions (array): Predicted class labels
        """
        if not self.trees:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        n_samples = X.shape[0]

        tree_predictions = []
        for tree, feature_indices in zip(
            self.trees, self.feature_indices, strict=False
        ):
            X_subset = X[:, feature_indices]
            predictions = tree.predict(X_subset)
            tree_predictions.append(predictions)

        tree_predictions = np.array(tree_predictions)

        final_predictions = []
        for i in range(n_samples):
            sample_predictions = tree_predictions[:, i]

            vote_counts = Counter(sample_predictions)
            majority_vote = vote_counts.most_common(1)[0][0]
            final_predictions.append(majority_vote)

        return np.array(final_predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities by averaging tree probabilities.

        Parameters:
        X (array-like): Features of shape (n_samples, n_features)

        Returns:
        probabilities (array): Probabilities of shape (n_samples, n_classes)
        """
        if not self.trees:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        n_samples = X.shape[0]

        total_probabilities = np.zeros((n_samples, self.n_classes))

        for tree, feature_indices in zip(
            self.trees, self.feature_indices, strict=False
        ):
            X_subset = X[:, feature_indices]
            tree_probabilities = tree.predict_proba(X_subset)
            total_probabilities += tree_probabilities

        average_probabilities = total_probabilities / self.n_estimators

        return average_probabilities

    def feature_importances(self):
        """
        Calculate feature importances based on impurity decrease.

        Returns:
        importances (array): Feature importances of shape (n_features,)
        """
        if not self.trees:
            raise ValueError("Model must be fitted before calculating importances")

        importances = np.zeros(self.n_features)

        for tree, feature_indices in zip(
            self.trees, self.feature_indices, strict=False
        ):
            tree_importances = self._calculate_tree_importance(
                tree.root, feature_indices
            )

            for i, feature_idx in enumerate(feature_indices):
                if i < len(tree_importances):
                    importances[feature_idx] += tree_importances[i]

        importances = importances / self.n_estimators

        total_importance = np.sum(importances)
        if total_importance > 0:
            importances = importances / total_importance

        return importances

    def _calculate_tree_importance(self, node, feature_indices):
        """
        Calculate feature importances for a single tree.

        Parameters:
        node: Tree node
        feature_indices: Feature indices used by this tree

        Returns:
        importances (array): Feature importances for this tree
        """
        importances = np.zeros(len(feature_indices))

        def _traverse_tree(node, total_samples):
            if node is None or node.is_leaf():
                return

            if node.feature is not None:
                try:
                    feature_pos = np.where(feature_indices == node.feature)[0][0]

                    if (
                        hasattr(node, "left")
                        and hasattr(node, "right")
                        and node.left
                        and node.right
                    ):
                        left_samples = (
                            node.left.n_samples
                            if hasattr(node.left, "n_samples")
                            else 0
                        )
                        right_samples = (
                            node.right.n_samples
                            if hasattr(node.right, "n_samples")
                            else 0
                        )

                        if total_samples > 0:
                            importance = (node.n_samples / total_samples) * (
                                node.impurity
                                - (left_samples / node.n_samples)
                                * (
                                    node.left.impurity
                                    if hasattr(node.left, "impurity")
                                    else 0
                                )
                                - (right_samples / node.n_samples)
                                * (
                                    node.right.impurity
                                    if hasattr(node.right, "impurity")
                                    else 0
                                )
                            )
                            importances[feature_pos] += importance
                except (IndexError, AttributeError):
                    pass

            if hasattr(node, "left") and node.left:
                _traverse_tree(node.left, total_samples)
            if hasattr(node, "right") and node.right:
                _traverse_tree(node.right, total_samples)

        total_samples = node.n_samples if hasattr(node, "n_samples") else 1
        _traverse_tree(node, total_samples)

        return importances

    def get_params(self):
        """Get parameters for this estimator."""
        return {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self
