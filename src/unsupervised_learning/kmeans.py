import numpy as np

import sys
sys.path.append('..')
from base_estimator import BaseEstimator


class KMeans(BaseEstimator):
    def __init__(self, n_clusters=8, max_iter=300, init="random", random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _init_centroids(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        if self.init == "random":
            centroids = np.random.uniform(
                X.min(axis=0), X.max(axis=0), (self.n_clusters, n_features)
            )
        elif self.init == "k-means++":
            centroids = self._kmeans_plus_plus_init(X)
        else:
            raise ValueError(f"Unsupported initialization method: {self.init}")

        return centroids

    def _kmeans_plus_plus_init(self, X):
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        centroids[0] = X[np.random.randint(n_samples)]

        for c_id in range(1, self.n_clusters):
            dist_sq = np.array(
                [min([np.sum((x - c) ** 2) for c in centroids[:c_id]]) for x in X]
            )
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break

            centroids[c_id] = X[i]

        return centroids

    def _assign_clusters(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)
        return centroids

    def _compute_inertia(self, X, labels, centroids):
        inertia = 0.0
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                cluster_points = X[labels == k]
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia

    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape

        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} must be <= n_samples={n_samples}"
            )

        centroids = self._init_centroids(X)

        for _i in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels, centroids)
        self.n_iter_ = _i + 1

        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        return self._assign_clusters(X, self.cluster_centers_)

    def fit_predict(self, X):
        return self.fit(X).predict(X)
