from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from .config import DATA_DIR, DEFAULT_SAMPLE_KEY, SAVE_DIR


def normalize_sample_id(value: Any) -> str:
    text = str(value).strip().upper()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return str(int(text))
    return text


def load_python_inputs(
    *,
    data_dir: Path = DATA_DIR,
    save_dir: Path = SAVE_DIR,
) -> dict[str, Any]:
    adata = ad.read_h5ad(data_dir / "data.h5ad")
    raw = ad.read_h5ad(data_dir / "raw.h5ad")
    meta = pd.read_csv(data_dir / "meta.csv", index_col=0, low_memory=False)
    with (save_dir / "novalue_embed.pickle").open("rb") as handle:
        novalue_embed = pickle.load(handle)

    adata.obs = meta.loc[adata.obs_names].copy()
    raw.obs = meta.loc[raw.obs_names].copy()
    adata.obs["_sampleid_normalized"] = adata.obs[DEFAULT_SAMPLE_KEY].map(normalize_sample_id)
    raw.obs["_sampleid_normalized"] = raw.obs[DEFAULT_SAMPLE_KEY].map(normalize_sample_id)
    return {
        "adata": adata,
        "raw": raw,
        "meta": meta,
        "novalue_embed": novalue_embed,
    }


def attach_embedding(adata, raw, novalue_embed, method_spec: dict):
    work = adata.copy()
    source = method_spec["source"]
    embedding_key = method_spec["embedding_key"]

    if source == "raw_h5ad":
        if embedding_key not in raw.obsm:
            raise KeyError(f"{embedding_key!r} not found in raw.obsm")
        work.obsm[embedding_key] = raw[work.obs_names].obsm[embedding_key]
        return work, embedding_key

    if source == "novalue_pickle":
        idx = method_spec["pickle_index"]
        work.obsm[embedding_key] = novalue_embed[idx]
        return work, embedding_key

    raise ValueError(f"Unsupported embedding source: {source}")


def split_reference_query(
    adata,
    *,
    reference_sample_ids: tuple[str, ...],
    sample_key: str = DEFAULT_SAMPLE_KEY,
):
    if "_sampleid_normalized" in adata.obs:
        normalized = adata.obs["_sampleid_normalized"]
    else:
        normalized = adata.obs[sample_key].map(normalize_sample_id)

    ref_mask = normalized.isin(reference_sample_ids)
    ref = adata[ref_mask].copy()
    query = adata[~ref_mask].copy()
    return ref, query
