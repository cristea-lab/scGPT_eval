from __future__ import annotations

from collections.abc import Sequence

import scanpy as sc

DEFAULT_UMAP_LABELS = [
    "Level 1 Annotation",
    "Level 2 Annotation",
    "Level 3 Annotation",
]


def compute_representation_umap(
    adata,
    *,
    use_rep: str,
    neighbors_kwargs: dict | None = None,
    umap_kwargs: dict | None = None,
    copy: bool = True,
):
    if use_rep not in adata.obsm:
        raise KeyError(f"{use_rep!r} not found in adata.obsm")

    work_adata = adata.copy() if copy else adata
    neighbor_args = dict(neighbors_kwargs or {})
    neighbor_args.setdefault("use_rep", use_rep)
    sc.pp.neighbors(work_adata, **neighbor_args)
    sc.tl.umap(work_adata, **(umap_kwargs or {}))
    return work_adata


def plot_annotation_umaps(
    adata,
    *,
    labels: Sequence[str] = DEFAULT_UMAP_LABELS,
    titles: Sequence[str] | None = None,
    save: str | None = None,
    **plot_kwargs,
):
    missing = [label for label in labels if label not in adata.obs]
    if missing:
        raise KeyError(f"Missing annotation columns: {missing}")

    plot_args = dict(plot_kwargs)
    plot_args.setdefault("frameon", False)
    plot_args.setdefault("wspace", 0.35)
    plot_args.setdefault("legend_loc", None)
    if titles is not None:
        plot_args["title"] = list(titles)
    return sc.pl.umap(adata, color=list(labels), save=save, **plot_args)
