from __future__ import annotations

import pickle
from typing import Iterable

import numpy as np
import scanpy as sc
import torch
from scipy import sparse

from .config import PipelineConfig


def load_subraw_adata(config: PipelineConfig):
    """
    Load the subset AnnData object and attach the precomputed binned counts layer.

    The `subraw_X_binned` file is intentionally loaded directly here because its
    generation currently lives in a separate preprocessing workflow. Keeping this
    as a single well-documented dependency makes it easy to replace later.
    """

    adata = sc.read(config.data_dir / config.subraw_filename)
    adata.var[config.model.gene_col] = adata.var.index

    with open(config.binned_counts_path, "rb") as handle:
        adata.layers[config.model.input_layer_key] = pickle.load(handle)

    return adata


def filter_genes_to_vocab(adata, vocab, gene_column: str):
    """Restrict the AnnData object to genes present in the scGPT vocabulary."""

    adata = adata.copy()
    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var[gene_column]]
    keep_mask = adata.var["id_in_vocab"].to_numpy() >= 0
    return adata[:, keep_mask].copy()


def sample_cells_per_annotation(
    adata,
    annotation_column: str,
    max_cells_per_type: int,
    seed: int,
    excluded_cell_types: Iterable[str] = (),
):
    """
    Sample up to `max_cells_per_type` cells for each annotation category.

    This mirrors the legacy notebooks' acceleration strategy, but the behavior is
    centralized here so pretrained and random runs use the same logic.
    """

    sampled = adata
    if excluded_cell_types:
        sampled = sampled[~sampled.obs[annotation_column].isin(list(excluded_cell_types))].copy()

    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []

    for cell_type in sampled.obs[annotation_column].astype(str).unique():
        cell_indices = np.flatnonzero(sampled.obs[annotation_column].astype(str).to_numpy() == cell_type)
        if len(cell_indices) > max_cells_per_type:
            cell_indices = rng.choice(cell_indices, size=max_cells_per_type, replace=False)
        sampled_indices.extend(cell_indices.tolist())

    sampled_indices.sort()
    return sampled[sampled_indices].copy()


def _max_nonzero_genes(matrix) -> int:
    """Compute the maximum number of non-zero genes per cell for dense or sparse matrices."""

    if sparse.issparse(matrix):
        counts = matrix.getnnz(axis=1)
        return int(np.asarray(counts).max())

    counts = np.count_nonzero(np.asarray(matrix), axis=1)
    return int(np.max(counts))


def adata_to_model_inputs(adata, vocab, config: PipelineConfig) -> dict[str, torch.Tensor | list[str]]:
    """
    Convert AnnData into the token/value tensors expected by scGPT.

    The legacy code performed this conversion inline inside the notebook. Moving
    it here makes the assumptions explicit and reduces duplication.
    """

    from scgpt.tokenizer import tokenize_and_pad_batch

    layer_key = config.model.input_layer_key
    all_counts = adata.layers[layer_key]
    genes = adata.var[config.model.gene_col].tolist()
    gene_ids = np.asarray(vocab(genes), dtype=int)

    if config.model.include_zero_gene:
        max_seq_len = adata.var.shape[0] + 1
    else:
        max_seq_len = _max_nonzero_genes(all_counts) + 1

    tokenized = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=config.model.pad_token,
        pad_value=config.model.pad_value,
        append_cls=True,
        include_zero_gene=config.model.include_zero_gene,
    )

    return {
        "gene_ids": tokenized["genes"],
        "values": torch.tensor(tokenized["values"], dtype=torch.float32),
        "labels": adata.obs[config.annotation_column].astype(str).tolist(),
    }
