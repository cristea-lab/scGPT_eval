from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

DEFAULT_BIO_LABELS = [
    "celltype_infercnv",
    "Level 1 Annotation",
    "Level 2 Annotation",
    "Level 3 Annotation",
]


def _normalized_weights(weights: Sequence[float] | None) -> np.ndarray:
    arr = np.asarray(weights if weights is not None else (1.0, 1.0, 1.0), dtype=float)
    if arr.shape != (3,):
        raise ValueError("weights must contain exactly three values: (ASW, NMI, ARI).")
    total = arr.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")
    return arr / total


def calculate_avgbio_table(
    adata,
    use_key: str,
    true_label_keys: Sequence[str],
    *,
    metric: str = "euclidean",
    weights: Sequence[float] | None = None,
    random_state: int = 0,
    cluster_key: str = "louvain",
    neighbors_kwargs: dict | None = None,
    cluster_kwargs: dict | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Compute avgBIO and keep ASW/NMI/ARI for each label.

    The implementation follows the zero-shot notebook as the source of truth:
    1. build neighbors on `use_key`
    2. run Louvain clustering
    3. compare predicted clusters against each label
    """

    if use_key not in adata.obsm:
        raise KeyError(f"{use_key!r} not found in adata.obsm")

    work_adata = adata.copy() if copy else adata
    rep = work_adata.obsm[use_key]
    weights_arr = _normalized_weights(weights)

    neighbor_args = dict(neighbors_kwargs or {})
    neighbor_args.setdefault("use_rep", use_key)
    neighbor_args.setdefault("random_state", random_state)
    sc.pp.neighbors(work_adata, **neighbor_args)

    cluster_args = dict(cluster_kwargs or {})
    cluster_args.setdefault("random_state", random_state)
    sc.tl.louvain(work_adata, key_added=cluster_key, **cluster_args)

    pred = work_adata.obs[cluster_key].to_numpy()
    rows: list[dict] = []
    for label_key in true_label_keys:
        if label_key not in work_adata.obs:
            raise KeyError(f"{label_key!r} not found in adata.obs")

        y_true = work_adata.obs[label_key].to_numpy()
        asw = silhouette_score(rep, y_true, metric=metric)
        nmi = normalized_mutual_info_score(y_true, pred)
        ari = adjusted_rand_score(y_true, pred)
        scaled_asw = (asw + 1.0) / 2.0
        scaled_ari = (ari + 1.0) / 2.0
        avgbio = float(np.dot(weights_arr, np.array([scaled_asw, nmi, scaled_ari])))

        rows.append(
            {
                "representation_key": use_key,
                "label_key": label_key,
                "cluster_key": cluster_key,
                "asw": float(asw),
                "nmi": float(nmi),
                "ari": float(ari),
                "scaled_asw": float(scaled_asw),
                "scaled_ari": float(scaled_ari),
                "avgbio": avgbio,
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table.attrs["avgbio_mean"] = float(table["avgbio"].mean())
        table.attrs["weights"] = weights_arr.tolist()
        table.attrs["cluster_key"] = cluster_key
        table.attrs["representation_key"] = use_key
    return table


def calculate_avgbio(
    adata,
    use_key: str,
    true_label_keys: Sequence[str],
    *,
    metric: str = "euclidean",
    weights: Sequence[float] | None = None,
    random_state: int = 0,
    cluster_key: str = "louvain",
    neighbors_kwargs: dict | None = None,
    cluster_kwargs: dict | None = None,
    copy: bool = True,
) -> dict:
    table = calculate_avgbio_table(
        adata,
        use_key,
        true_label_keys,
        metric=metric,
        weights=weights,
        random_state=random_state,
        cluster_key=cluster_key,
        neighbors_kwargs=neighbors_kwargs,
        cluster_kwargs=cluster_kwargs,
        copy=copy,
    )
    return {
        "representation_key": use_key,
        "cluster_key": cluster_key,
        "weights": table.attrs.get("weights"),
        "avgbio_mean": table.attrs.get("avgbio_mean"),
        "per_label": table.to_dict(orient="records"),
        "table": table,
    }


def _json_ready(data: dict) -> dict:
    out = {}
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            out[key] = value.to_dict(orient="records")
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def save_avgbio_results(
    results: pd.DataFrame | dict,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, pd.DataFrame):
        table = results
        payload = {
            "representation_key": table.attrs.get("representation_key"),
            "cluster_key": table.attrs.get("cluster_key"),
            "weights": table.attrs.get("weights"),
            "avgbio_mean": table.attrs.get("avgbio_mean"),
            "per_label": table.to_dict(orient="records"),
        }
    else:
        table = results["table"] if isinstance(results.get("table"), pd.DataFrame) else pd.DataFrame(results["per_label"])
        payload = _json_ready(results)
        payload.pop("table", None)

    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return csv_path, json_path
