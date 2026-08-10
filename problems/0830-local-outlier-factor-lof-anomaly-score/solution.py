import numpy as np

def local_outlier_factor(X, k):
    """
    Compute the Local Outlier Factor (LOF) score for each point in X.

    Args:
        X: array-like of shape (n, d)
        k: int, neighborhood size

    Returns:
        list of length n with the LOF score of each point
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # Pairwise Euclidean distances.
    distances = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)

    # Find k nearest neighbors for each point, excluding the point itself.
    neighbors = []
    k_distances = np.empty(n, dtype=float)

    for i in range(n):
        order = np.argsort(distances[i], kind="stable")
        order = order[order != i]
        nbrs = order[:k]

        neighbors.append(nbrs)
        k_distances[i] = distances[i, nbrs[-1]]

    # Compute local reachability density (LRD).
    lrd = np.empty(n, dtype=float)

    for i in range(n):
        nbrs = neighbors[i]

        reach_dists = np.maximum(
            k_distances[nbrs],
            distances[i, nbrs]
        )

        lrd[i] = 1.0 / np.mean(reach_dists)

    # Compute LOF.
    lof = np.empty(n, dtype=float)

    for i in range(n):
        nbrs = neighbors[i]
        lof[i] = np.mean(lrd[nbrs] / lrd[i])

    return lof.tolist()
