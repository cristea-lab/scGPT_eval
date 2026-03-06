from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _majority_vote(labels: np.ndarray) -> str:
    counts = Counter(labels.tolist())
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def run_knn_label_transfer(
    ref_adata,
    query_adata,
    *,
    embedding_key: str,
    label_key: str,
    k: int,
) -> pd.DataFrame:
    if embedding_key not in ref_adata.obsm:
        raise KeyError(f"{embedding_key!r} not found in reference embeddings")
    if embedding_key not in query_adata.obsm:
        raise KeyError(f"{embedding_key!r} not found in query embeddings")
    if label_key not in ref_adata.obs or label_key not in query_adata.obs:
        raise KeyError(f"{label_key!r} must exist in both reference and query metadata")

    ref_embeddings = np.asarray(ref_adata.obsm[embedding_key])
    query_embeddings = np.asarray(query_adata.obsm[embedding_key])
    effective_k = min(k, ref_embeddings.shape[0])
    if effective_k < 1:
        raise ValueError("Reference set is empty after splitting by sample id.")

    knn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
    knn.fit(ref_embeddings)
    distances, indices = knn.kneighbors(query_embeddings)

    ref_labels = ref_adata.obs[label_key].to_numpy()
    predicted = [_majority_vote(ref_labels[idx]) for idx in indices]
    truth = query_adata.obs[label_key].astype(str).to_numpy()

    rows = []
    for obs_name, true_label, pred_label, dist, idx in zip(
        query_adata.obs_names,
        truth,
        predicted,
        distances,
        indices,
    ):
        rows.append(
            {
                "cell_id": obs_name,
                "truth": true_label,
                "prediction": pred_label,
                "neighbor_distances": ";".join(f"{value:.6g}" for value in dist),
                "neighbor_indices": ";".join(str(int(value)) for value in idx),
            }
        )
    return pd.DataFrame(rows)
