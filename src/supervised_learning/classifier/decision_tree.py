import numpy as np

from ..base_estimator import BaseEstimator
from .classifier import ClassifierMixin


class TreeNode:
    """
    A node in the decision tree.
    """

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        impurity=None,
        n_samples=None,
        class_counts=None,
    ):
        """
        Initialize tree node.

        Parameters:
        feature (int): Feature index for splitting
        threshold (float): Threshold value for splitting
        left (TreeNode): Left child node
        right (TreeNode): Right child node
        value: Predicted class (for leaf nodes)
        impurity (float): Node impurity
        n_samples (int): Number of samples in this node
        class_counts (dict): Count of each class in this node
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity = impurity
        self.n_samples = n_samples
        self.class_counts = class_counts

    def is_leaf(self):
        """Check if this is a leaf node."""
        return self.value is not None


class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """
    CART (Classification and Regression Tree) for classification.

    Uses Gini impurity as the splitting criterion and implements
    binary splits for continuous and categorical features.
    """

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        random_state=None,
    ):
        """
        Initialize Decision Tree Classifier.

        Parameters:
        criterion (str): Splitting criterion ('gini' or 'entropy')
        max_depth (int): Maximum depth of the tree
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required at a leaf node
        min_impurity_decrease (float): Minimum impurity decrease for a split
        random_state (int): Random seed for reproducibility
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

        self.root = None
        self.classes = None
        self.n_classes = None
        self.n_features = None

        if random_state is not None:
            np.random.seed(random_state)

    def _gini_impurity(self, y):
        """
        Calculate Gini impurity.

        Parameters:
        y (array): Class labels

        Returns:
        gini (float): Gini impurity
        """
        if len(y) == 0:
            return 0

        counts = np.bincount(y)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _entropy(self, y):
        """
        Calculate entropy.

        Parameters:
        y (array): Class labels

        Returns:
        entropy (float): Entropy
        """
        if len(y) == 0:
            return 0

        counts = np.bincount(y)
        probabilities = counts[counts > 0] / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _calculate_impurity(self, y):
        """Calculate impurity based on chosen criterion."""
        if self.criterion == "gini":
            return self._gini_impurity(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _get_class_counts(self, y):
        """Get class counts as a dictionary."""
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts, strict=False))

    def _find_best_split(self, X, y):
        """
        Find the best split for the current node.

        Parameters:
        X (array): Feature matrix
        y (array): Target labels

        Returns:
        best_feature (int): Best feature index
        best_threshold (float): Best threshold value
        best_impurity_decrease (float): Impurity decrease from best split
        """
        best_impurity_decrease = 0
        best_feature = None
        best_threshold = None
        current_impurity = self._calculate_impurity(y)

        n_samples, n_features = X.shape

        for feature in range(n_features):
            feature_values = np.unique(X[:, feature])

            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2

                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                weighted_impurity = (
                    n_left * left_impurity + n_right * right_impurity
                ) / n_samples
                impurity_decrease = current_impurity - weighted_impurity

                if impurity_decrease > best_impurity_decrease:
                    best_impurity_decrease = impurity_decrease
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_impurity_decrease

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Parameters:
        X (array): Feature matrix
        y (array): Target labels
        depth (int): Current depth

        Returns:
        node (TreeNode): Root node of the subtree
        """
        n_samples = len(y)
        class_counts = self._get_class_counts(y)
        impurity = self._calculate_impurity(y)

        most_common_class = max(class_counts, key=class_counts.get)

        if (
            depth == self.max_depth
            or n_samples < self.min_samples_split
            or impurity == 0
            or len(np.unique(y)) == 1
        ):
            return TreeNode(
                value=most_common_class,
                impurity=impurity,
                n_samples=n_samples,
                class_counts=class_counts,
            )

        best_feature, best_threshold, best_impurity_decrease = self._find_best_split(
            X, y
        )

        if best_feature is None or best_impurity_decrease < self.min_impurity_decrease:
            return TreeNode(
                value=most_common_class,
                impurity=impurity,
                n_samples=n_samples,
                class_counts=class_counts,
            )

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            impurity=impurity,
            n_samples=n_samples,
            class_counts=class_counts,
        )

    def fit(self, X, y):
        """
        Fit the decision tree classifier.

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

        if y.dtype.kind in {"U", "S", "O"}:
            label_to_int = {label: i for i, label in enumerate(self.classes)}
            y = np.array([label_to_int[label] for label in y])

        self.root = self._build_tree(X, y)

        return self

    def _predict_sample(self, x, node):
        """
        Predict a single sample by traversing the tree.

        Parameters:
        x (array): Single sample
        node (TreeNode): Current node

        Returns:
        prediction: Predicted class
        """
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        """
        Predict class labels.

        Parameters:
        X (array-like): Features of shape (n_samples, n_features)

        Returns:
        predictions (array): Predicted class labels
        """
        if self.root is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        predictions = []

        for x in X:
            prediction = self._predict_sample(x, self.root)
            predictions.append(prediction)

        return np.array(predictions)

    def _predict_proba_sample(self, x, node):
        """
        Predict probabilities for a single sample.

        Parameters:
        x (array): Single sample
        node (TreeNode): Current node

        Returns:
        probabilities (dict): Class probabilities
        """
        if node.is_leaf():
            total_samples = sum(node.class_counts.values())
            proba = {}
            for class_label in self.classes:
                count = node.class_counts.get(class_label, 0)
                proba[class_label] = count / total_samples
            return proba

        if x[node.feature] <= node.threshold:
            return self._predict_proba_sample(x, node.left)
        else:
            return self._predict_proba_sample(x, node.right)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters:
        X (array-like): Features of shape (n_samples, n_features)

        Returns:
        probabilities (array): Probabilities of shape (n_samples, n_classes)
        """
        if self.root is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        probabilities = []

        for x in X:
            sample_proba = self._predict_proba_sample(x, self.root)

            proba_array = [
                sample_proba.get(class_label, 0) for class_label in self.classes
            ]
            probabilities.append(proba_array)

        return np.array(probabilities)

    def get_depth(self):
        """Get the depth of the tree."""

        def _get_depth(node):
            if node is None or node.is_leaf():
                return 0
            return 1 + max(_get_depth(node.left), _get_depth(node.right))

        return _get_depth(self.root)

    def get_n_leaves(self):
        """Get the number of leaves in the tree."""

        def _count_leaves(node):
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)

        return _count_leaves(self.root)
