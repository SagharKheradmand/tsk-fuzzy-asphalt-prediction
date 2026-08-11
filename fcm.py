import numpy as np
from dataclasses import dataclass


@dataclass
class FCMResult:
    centers: np.ndarray
    U: np.ndarray
    sigmas: np.ndarray


def fuzzy_c_means(X, n_clusters, m=2.0, max_iter=200, tol=1e-5, seed=42):
    """
    Fuzzy C-Means clustering from scratch.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)

    U = rng.random((n_clusters, n))
    U /= U.sum(axis=0, keepdims=True)

    for _ in range(max_iter):
        Um = U**m
        centers = (Um @ X) / (Um.sum(axis=1, keepdims=True) + 1e-12)

        dist2 = np.zeros((n_clusters, n))
        for i in range(n_clusters):
            diff = X - centers[i]
            dist2[i] = np.sum(diff**2, axis=1)

        dist2 = np.where(dist2 < 1e-12, 1e-12, dist2)
        power = 1.0 / (m - 1.0)

        ratio = (dist2[:, None, :] / dist2[None, :, :]) ** power
        U_new = 1.0 / ratio.sum(axis=1)

        if np.max(np.abs(U_new - U)) < tol:
            break

        U = U_new

    sigmas = np.zeros((n_clusters, d))
    Um = U**m
    for i in range(n_clusters):
        diff = X - centers[i]
        sigmas[i] = np.sqrt(
            np.sum(Um[i][:, None] * diff**2, axis=0) / (Um[i].sum() + 1e-12)
        )
    sigmas = np.where(sigmas < 1e-6, 1e-6, sigmas)

    return FCMResult(centers=centers, U=U, sigmas=sigmas)
