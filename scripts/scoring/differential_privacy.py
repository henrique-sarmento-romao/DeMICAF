from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_feature_histograms(
    latent: NDArray[np.floating],
    bin_edges: list[NDArray[np.floating]],
) -> list[NDArray[np.int64]]:
    """
    Compute one histogram for every latent feature.

    latent:
        Shape (n_records, d).

    bin_edges:
        A list of length d. bin_edges[j] contains the public edges
        for latent feature j.
    """
    latent = np.asarray(latent, dtype=float)

    if latent.ndim != 2:
        raise ValueError("latent must have shape (n_records, d).")

    n_records, d = latent.shape

    if len(bin_edges) != d:
        raise ValueError(f"Expected {d} sets of bin edges, received {len(bin_edges)}.")

    histograms: list[NDArray[np.int64]] = []

    for j in range(d):
        edges = np.asarray(bin_edges[j], dtype=float)

        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(f"Invalid edges for feature {j}.")

        if not np.all(np.diff(edges) > 0):
            raise ValueError(f"Bin edges for feature {j} must be strictly increasing.")

        # Map every value into exactly one public bin.
        # The first and last bins effectively absorb out-of-range values.
        upper = np.nextafter(edges[-1], edges[0])
        values = np.clip(latent[:, j], edges[0], upper)

        counts, _ = np.histogram(values, bins=edges)
        histograms.append(counts.astype(np.int64))

    return histograms


def concatenate_histograms(
    histograms: list[NDArray[np.integer]],
) -> NDArray[np.float64]:
    """Concatenate all feature histograms into one DP query vector."""
    if not histograms:
        raise ValueError("At least one histogram is required.")

    return np.concatenate(
        [np.asarray(histogram, dtype=float) for histogram in histograms]
    )


def dp_feature_histograms(
    client_histograms: list[list[NDArray[np.integer]]],
    epsilon: float,
    delta: float,
    seed: int | None = None,
    replacement_adjacency: bool = False,
) -> tuple[list[NDArray[np.float64]], float]:
    """
    Aggregate d feature histograms and apply one Gaussian mechanism.

    Assumption:
        Each protected unit contributes exactly one latent vector and,
        consequently, one count to every feature histogram.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1).")

    if not client_histograms:
        raise ValueError("At least one client is required.")

    d = len(client_histograms[0])

    if d == 0:
        raise ValueError("At least one latent feature is required.")

    for client in client_histograms:
        if len(client) != d:
            raise ValueError("Every client must provide one histogram per feature.")

    bin_counts = [len(client_histograms[0][j]) for j in range(d)]

    for client in client_histograms:
        for j in range(d):
            if len(client[j]) != bin_counts[j]:
                raise ValueError(f"Inconsistent histogram size for feature {j}.")

    global_histograms = [
        np.sum(
            [np.asarray(client[j], dtype=float) for client in client_histograms],
            axis=0,
        )
        for j in range(d)
    ]

    # Each record contributes one count to each of d histograms.
    l2_sensitivity = np.sqrt(2.0 * d) if replacement_adjacency else np.sqrt(float(d))

    sigma = l2_sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon

    rng = np.random.default_rng(seed)

    noisy_histograms = [
        histogram
        + rng.normal(
            loc=0.0,
            scale=sigma,
            size=histogram.shape,
        )
        for histogram in global_histograms
    ]

    # Valid DP post-processing.
    noisy_histograms = [np.maximum(histogram, 0.0) for histogram in noisy_histograms]

    return noisy_histograms, sigma
