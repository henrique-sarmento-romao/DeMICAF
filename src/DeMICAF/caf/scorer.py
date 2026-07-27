"""Compliance scoring functions: Mahalanobis distance (MD) and cosine-similarity k-NN (CS)."""

from collections.abc import Sequence

import numpy as np
import torch
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf


def reference_vector(ref_data: np.ndarray, regularization: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Return the (mean, regularized covariance) of a reference feature set."""
    ref_data = np.asarray(ref_data, dtype=np.float64)
    if ref_data.ndim != 2:
        raise ValueError("ref_data must be a 2D array")

    n_samples, n_features = ref_data.shape

    mean = np.mean(ref_data, axis=0)
    cov = LedoitWolf().fit(ref_data).covariance_ if n_samples >= 2 else np.eye(n_features, dtype=np.float64)

    if not np.all(np.isfinite(cov)):
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    cov = cov + regularization * np.eye(n_features)
    return mean, cov


def mahalanobis_distance(
    points: np.ndarray,
    mean_vectors: Sequence[np.ndarray],
    covmatrixs: Sequence[np.ndarray],
    chi: bool = False,
    aggregation_method: str = "min",
) -> np.ndarray:
    """Squared Mahalanobis distance of each point to the closest cluster reference."""
    # initialize with inf so skipped clusters don't produce zero minima
    inv_covmatrixs = [np.linalg.pinv(cov) for cov in covmatrixs]
    mahal_distances = np.full((len(points), len(mean_vectors)), np.inf)
    for i, (mean_vec, inv_covmat) in enumerate(zip(mean_vectors, inv_covmatrixs, strict=False)):
        points_mu = points - mean_vec
        left = np.dot(points_mu, inv_covmat)
        mahal_cluster = np.sum(left * points_mu, axis=1)
        mahal_distances[:, i] = mahal_cluster

    if aggregation_method == "min":
        mahal = np.min(mahal_distances, axis=1)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

    return chi2.sf(mahal, df=points.shape[1]) if chi else mahal


def mahalanobis(
    ref_data: np.ndarray,
    points: np.ndarray,
    k: int = 1,
    chi: bool = False,
    regularization: float = 1e-5,
    aggregation_method: str = "min",
    size_weighting: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Score points against a reference set, optionally k-means clustered (MD ranking function)."""
    # Compute reference vectors and covariance matrices for each cluster.
    if k == 1:
        mean_vec, covmat = reference_vector(ref_data, regularization=regularization)
        mean_vectors = [mean_vec]
        covmatrixs = [covmat]
    else:
        clusterer = KMeans(n_clusters=k, init="k-means++", n_init=10).fit(ref_data)
        centroids = clusterer.cluster_centers_
        labels = clusterer.labels_
        unique_labels = np.arange(k)

        if verbose:
            cluster_sizes = np.bincount(labels)
            print("Cluster sizes:", cluster_sizes)

        mean_vectors = []
        covmatrixs = []
        for i, cluster_mean in enumerate(centroids):
            cluster_data = ref_data[labels == unique_labels[i]]
            if size_weighting:
                adjusted_regularization = regularization * np.sqrt(max(cluster_data.shape[0], 1))
            else:
                adjusted_regularization = regularization

            if cluster_data.shape[0] == 0:
                cluster_data = cluster_mean[None, :]

            _, cluster_cov = reference_vector(cluster_data, regularization=adjusted_regularization)

            mean_vectors.append(cluster_mean)
            covmatrixs.append(cluster_cov)

    # Compute Mahalanobis distances to each cluster
    mahal = mahalanobis_distance(points, mean_vectors, covmatrixs, chi=chi, aggregation_method=aggregation_method)
    return mahal, mean_vectors


def cosine_sim(
    ref_data: np.ndarray | torch.Tensor,
    data: np.ndarray | torch.Tensor,
    k: int = 1,
    chunk_size: int = 1000,
    verbose: bool = False,
) -> torch.Tensor:
    """Arc-cosine distance to the k nearest reference neighbours (CS ranking function)."""
    num_samples = data.shape[0]
    # Select device: prefer mps (Apple Silicon), then CUDA, then CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Normalize both ref_data and data to lie on the unit sphere
    ref_tensor = torch.as_tensor(ref_data).to(device)
    ref_norms = ref_tensor.norm(dim=1, keepdim=True) + 1e-12
    ref_tensor = ref_tensor / ref_norms

    data_tensor = torch.as_tensor(data).to(device)
    data_norms = data_tensor.norm(dim=1, keepdim=True) + 1e-12
    data_tensor = data_tensor / data_norms

    # Initialize a tensor to store k-NN distances
    knn_distances = torch.zeros(num_samples, dtype=torch.float32, device=device)

    elements = k  # Retrieve k elements from ref_data
    for start_idx in range(0, num_samples, chunk_size):
        if verbose:
            print(f"Processing chunk starting at index {start_idx}")
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk_data = data_tensor[start_idx:end_idx, :]

        # Compute cosine similarities between chunk_data and ref_data
        cosine_sim_chunk = torch.mm(chunk_data, ref_tensor.t())

        # Check if ref_data and data are the same, meaning we have self-similarity to handle
        if ref_tensor.shape == data_tensor.shape and torch.equal(ref_tensor, data_tensor):
            # Identify self-similarity within this chunk
            for i in range(start_idx, end_idx):
                cosine_sim_chunk[i - start_idx, i] = -2.0  # Exclude self-similarity by setting it to a low value

        # Find the top k nearest neighbors for each row in the chunk
        top_k_values, _ = torch.topk(cosine_sim_chunk, k=elements, dim=1, largest=True)

        # Ensure values are within the valid range for arccos [-1, 1]
        top_k_values = torch.clamp(top_k_values, -1.0, 1.0)

        # Calculate distance to k-th nearest neighbor (average for top-k)
        knn_distances[start_idx:end_idx] = torch.arccos(top_k_values.mean(dim=1))

    return knn_distances.cpu()
