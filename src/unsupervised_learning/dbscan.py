import numpy as np

from ..base_estimator import BaseEstimator


class DBSCAN(BaseEstimator):
    def __init__(self, eps=0.5, min_samples=5, metric="euclidean", p=2):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.p = p
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None

    def _distance(self, x1, x2):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        elif self.metric == "minkowski":
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1.0 / self.p)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _region_query(self, X, point_idx):
        neighbors = []
        for i, point in enumerate(X):
            if self._distance(X[point_idx], point) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, labels, visited):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                neighbor_neighbors = self._region_query(X, neighbor_idx)

                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors.extend(neighbor_neighbors)

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            i += 1

    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        labels = np.full(n_samples, -1, dtype=int)
        visited = set()
        cluster_id = 0
        core_samples = []

        for point_idx in range(n_samples):
            if point_idx in visited:
                continue

            visited.add(point_idx)
            neighbors = self._region_query(X, point_idx)

            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                core_samples.append(point_idx)
                self._expand_cluster(
                    X, point_idx, neighbors, cluster_id, labels, visited
                )
                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_samples)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        if len(unique_labels) > 0:
            self.components_ = X[self.core_sample_indices_]
        else:
            self.components_ = np.array([]).reshape(0, X.shape[1])

        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
