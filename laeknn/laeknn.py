"""Locally Adaptive Evidential K‑Nearest Neighbours classifier.

This module implements the LAE‑KNN algorithm.  It combines local
Mahalanobis distance estimation, density‑tempered kernel weights and
Dirichlet evidential aggregation to produce probability estimates and
uncertainty for each prediction.

Example:

>>> import numpy as np
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> from laeknn import LAEKNN
>>> X, y = load_iris(return_X_y=True)
>>> Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
>>> clf = LAEKNN().fit(Xtr, ytr)
>>> preds = clf.predict(Xte)
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class LAEKNN:
    """Locally Adaptive Evidential K‑Nearest Neighbours classifier.

    Parameters
    ----------
    per_class_centers : int, default=5
        Number of representative coreset centres to select per class.

    m : int, default=10
        Number of nearest coreset points used to estimate the local
        covariance matrix.

    k_density : int, default=10
        Number of neighbours used to compute the k‑nearest neighbour
        distance proxy for density tempering.

    beta : float, default=0.5
        Exponent controlling the strength of density tempering.  A value of
        0 corresponds to a standard Gaussian kernel.

    tau : float, default=1.0
        Bandwidth parameter for the Gaussian kernel in the weight
        computation.

    lam : float, default=1e-2
        Regularisation parameter added to the diagonal of the covariance
        matrix to ensure invertibility.
    """

    def __init__(self,
                 per_class_centers: int = 5,
                 m: int = 10,
                 k_density: int = 10,
                 beta: float = 0.5,
                 tau: float = 1.0,
                 lam: float = 1e-2) -> None:
        self.per_class_centers = per_class_centers
        self.m = m
        self.k_density = k_density
        self.beta = beta
        self.tau = tau
        self.lam = lam

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LAEKNN":
        """Fit the LAE‑KNN model.

        This method computes a small coreset by selecting a fixed number of
        centres per class via k‑means clustering and precomputes nearest
        neighbour structures for distance and density computations.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        LAEKNN
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        centres = []
        labels = []
        # per‑class k‑means to build a coreset
        for cls in self.classes_:
            X_c = X[y == cls]
            n_clusters = min(self.per_class_centers, len(X_c))
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            km.fit(X_c)
            centres.append(km.cluster_centers_)
            labels.append(np.full(n_clusters, cls))
        self.S = np.vstack(centres)
        self.Sy = np.concatenate(labels)
        # nearest neighbours for local covariance and density
        self.nn_S = NearestNeighbors(n_neighbors=min(self.m, len(self.S)))
        self.nn_S.fit(self.S)
        self.nn_den = NearestNeighbors(n_neighbors=min(self.k_density, len(self.S)))
        self.nn_den.fit(self.S)
        # precompute k‑th nearest neighbour distances for density tempering
        self.rS = self._k_radius(self.S)
        return self

    def _k_radius(self, Z: np.ndarray) -> np.ndarray:
        """Compute the k‑th nearest neighbour distance for each point in Z."""
        dists, _ = self.nn_den.kneighbors(Z)
        # select the last column (kth distance) and add epsilon for stability
        return dists[:, -1] + 1e-12

    def _local_cov(self, x: np.ndarray) -> np.ndarray:
        """Estimate a local covariance matrix around a query point."""
        distances, indices = self.nn_S.kneighbors(x.reshape(1, -1))
        local_points = self.S[indices[0]]
        mu = local_points.mean(axis=0, keepdims=True)
        centred = local_points - mu
        cov = (centred.T @ centred) / len(local_points)
        cov += self.lam * np.eye(centred.shape[1])
        return cov

    def _weights(self, x: np.ndarray) -> np.ndarray:
        """Compute kernel weights from a query point to all coreset points."""
        cov = self._local_cov(x)
        inv_cov = np.linalg.inv(cov)
        diff = self.S - x
        # Mahalanobis squared distance for all coreset points
        d2 = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        # local density proxies
        r_x = self._k_radius(x.reshape(1, -1))
        t = (r_x * self.rS) ** self.beta
        return np.exp(-d2 / (self.tau * t))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities for query points.

        Parameters
        ----------
        X : ndarray of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        ndarray of shape (n_queries, n_classes)
            Class probability estimates.
        """
        X = np.asarray(X)
        C = len(self.classes_)
        probs = np.zeros((len(X), C))
        for i, x in enumerate(X):
            w = self._weights(x)
            evidence = np.zeros(C)
            for idx, cls in enumerate(self.classes_):
                evidence[idx] = w[self.Sy == cls].sum()
            # Dirichlet parameters with unit prior
            alpha = evidence + 1.0
            probs[i] = alpha / alpha.sum()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for query points."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]