"""
Environmental distance metric — Contribution 2 of the paper.

Quantifies how different two physical scenes are, using channel-statistic features.
Lets us plot: generalization accuracy vs environmental distance.
"""
import numpy as np


def scene_descriptor(X):
    """
    Compute a scene descriptor from a dataset of sub-6 GHz features.

    Features chosen so they reflect the underlying channel structure, not
    the specific users:
      - mean & std of feature magnitudes (energy distribution)
      - per-dimension std (angular spread proxy)
      - sparsity (number of effective paths)
      - effective rank (via singular values)
    """
    mags = np.abs(X)
    mean_energy = float(mags.mean())
    std_energy = float(mags.std())
    per_dim_std = float(mags.std(axis=0).mean())

    # Sparsity: fraction of feature values below 10% of max
    threshold = 0.1 * mags.max()
    sparsity = float((mags < threshold).mean())

    # Effective rank: entropy of normalized singular values
    # Use a subsample to keep this fast
    n = min(1000, X.shape[0])
    sub = X[np.random.choice(X.shape[0], n, replace=False)]
    _, s, _ = np.linalg.svd(sub, full_matrices=False)
    s_norm = s / (s.sum() + 1e-12)
    entropy = -float((s_norm * np.log(s_norm + 1e-12)).sum())

    return np.array([mean_energy, std_energy, per_dim_std, sparsity, entropy])


def environmental_distance(X_a, X_b):
    """Euclidean distance between normalized scene descriptors."""
    da = scene_descriptor(X_a)
    db = scene_descriptor(X_b)
    # Normalize each descriptor component to unit scale using the pair
    denom = np.maximum(np.abs(da) + np.abs(db), 1e-6)
    return float(np.linalg.norm((da - db) / denom))


def pairwise_env_distances(datasets):
    """
    datasets: dict {scene_name: X}
    Returns: dict of pairwise distances (as a dict keyed by frozenset)
    """
    names = list(datasets.keys())
    dists = {}
    for i, a in enumerate(names):
        for b in names[i:]:
            if a == b:
                dists[(a, b)] = 0.0
            else:
                dists[(a, b)] = environmental_distance(datasets[a], datasets[b])
                dists[(b, a)] = dists[(a, b)]
    return dists
