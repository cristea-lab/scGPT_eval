from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from einops import rearrange
from scipy import sparse
from tqdm.auto import tqdm

from .metrics import DEFAULT_BIO_LABELS, calculate_avgbio_table, save_avgbio_results
from .plotting import DEFAULT_UMAP_LABELS, compute_representation_umap

DEFAULT_SCGPT_PARAMS = {
    "gene_col": "gene",
    "label_col": "treatmentStatus",
    "data_is_raw": False,
    "filter_gene_by_counts": False,
    "input_layer_key": "X_binned",
    "max_seq_len": 3001,
    "batch_size": 32,
    "domain_spec_batchnorm": "batchnorm",
    "input_emb_style": "continuous",
    "cell_emb_style": "cls",
    "n_input_bins": 51,
    "mvc_decoder_style": "inner product",
    "ecs_threshold": 0.0,
    "explicit_zero_prob": False,
    "use_fast_transformer": True,
    "fast_transformer_backend": "flash",
    "pre_norm": False,
    "n_layers_cls": 3,
    "nhead": 8,
    "embsize": 512,
    "d_hid": 512,
    "nlayers": 12,
    "dropout": 0.2,
    "ADV": False,
    "MLM": False,
    "MVC": False,
    "CLS": False,
    "DAB": False,
    "CCE": False,
    "ECS": False,
    "do_sample_in_train": False,
    "INPUT_BATCH_LABELS": False,
    "num_batch_labels": None,
    "epochs": 10,
    "n_bins": 51,
    "include_zero_gene": False,
    "pad_token": "<pad>",
    "mask_value": -1,
    "pad_value": -2,
}


def _lazy_import_scgpt():
    import scgpt as scg
    from scgpt.model import TransformerModel
    from scgpt.preprocess import Preprocessor
    from scgpt.tokenizer import tokenize_and_pad_batch
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    return {
        "scg": scg,
        "TransformerModel": TransformerModel,
        "Preprocessor": Preprocessor,
        "GeneVocab": GeneVocab,
        "tokenize_and_pad_batch": tokenize_and_pad_batch,
    }


def build_hvg_subset(adata, n_hvg: int, *, gene_col: str = "gene"):
    if "variances_norm" not in adata.var:
        raise KeyError("'variances_norm' not found in adata.var")
    subset = adata[:, adata.var.sort_values(by="variances_norm", ascending=False).head(n_hvg).index].copy()
    if gene_col not in subset.var:
        subset.var[gene_col] = subset.var.index
    return subset


def embed_adata_with_scgpt(
    adata,
    model_dir: str | Path,
    *,
    gene_col: str = "gene",
    cell_type_key: str = "celltype_infercnv",
    batch_size: int = 128,
    max_length: int | None = None,
):
    modules = _lazy_import_scgpt()
    scg = modules["scg"]
    work_adata = adata.copy()
    if gene_col not in work_adata.var:
        work_adata.var[gene_col] = work_adata.var.index
    kwargs = {
        "gene_col": gene_col,
        "cell_type_key": cell_type_key,
        "batch_size": batch_size,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    return scg.tasks.embed_data(work_adata, Path(model_dir), **kwargs)


def preprocess_for_scgpt(
    adata,
    *,
    params: dict | None = None,
    use_key: str = "X",
    result_binned_key: str = "X_binned",
):
    modules = _lazy_import_scgpt()
    Preprocessor = modules["Preprocessor"]
    merged = dict(DEFAULT_SCGPT_PARAMS)
    if params:
        merged.update(params)

    processor = Preprocessor(
        use_key=use_key,
        filter_gene_by_counts=merged["filter_gene_by_counts"],
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=merged["data_is_raw"],
        subset_hvg=False,
        binning=merged["n_bins"],
        result_binned_key=result_binned_key,
    )
    work_adata = adata.copy()
    processor(work_adata)
    return work_adata


def _filter_to_vocab(adata, vocab, *, gene_col: str) -> Any:
    work_adata = adata.copy()
    work_adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in work_adata.var[gene_col]]
    return work_adata[:, work_adata.var["id_in_vocab"] >= 0].copy()


def tokenize_adata(
    adata,
    vocab,
    *,
    params: dict | None = None,
    gene_col: str = "gene",
    layer_key: str = "X_binned",
    include_zero: bool = False,
):
    modules = _lazy_import_scgpt()
    tokenize_and_pad_batch = modules["tokenize_and_pad_batch"]
    merged = dict(DEFAULT_SCGPT_PARAMS)
    if params:
        merged.update(params)

    all_counts = adata.layers[layer_key]
    genes = adata.var[gene_col].tolist()
    gene_ids = np.asarray(vocab(genes), dtype=int)
    if include_zero:
        max_seq_len = adata.var.shape[0] + 1
    else:
        nonzero = np.sum(all_counts > 0, axis=1)
        max_seq_len = int(np.asarray(nonzero).max()) + 1

    tokenized = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=merged["pad_token"],
        pad_value=merged["pad_value"],
        append_cls=True,
        include_zero_gene=include_zero,
    )
    return {
        "gene_ids": tokenized["genes"],
        "values": torch.tensor(tokenized["values"], dtype=torch.float),
    }


def load_pretrained_model(
    model_dir: str | Path,
    *,
    params: dict | None = None,
    device: str | torch.device | None = None,
):
    modules = _lazy_import_scgpt()
    GeneVocab = modules["GeneVocab"]
    TransformerModel = modules["TransformerModel"]
    merged = dict(DEFAULT_SCGPT_PARAMS)
    if params:
        merged.update(params)

    model_dir = Path(model_dir)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    model = TransformerModel(
        len(vocab),
        merged["embsize"],
        merged["nhead"],
        merged["d_hid"],
        merged["nlayers"],
        vocab=vocab,
        pad_token=merged["pad_token"],
        pad_value=merged["pad_value"],
        domain_spec_batchnorm=merged["domain_spec_batchnorm"],
        n_input_bins=merged["n_input_bins"],
        use_fast_transformer=merged["use_fast_transformer"],
        pre_norm=merged["pre_norm"],
    )
    model.domain_spec_batchnorm = False

    pretrained = torch.load(model_dir / "best_model.pt", map_location="cpu")
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in pretrained.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model_state.update(compatible)
    model.load_state_dict(model_state)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, vocab, merged


def _run_attn(layer, src, key_padding_mask):
    qkv = layer.self_attn.Wqkv(src)
    qkv_before = qkv.detach().cpu().numpy()
    qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=layer.self_attn.num_heads)
    qkv_after = qkv.detach().cpu().numpy()
    context, attn_weights = layer.self_attn.inner_attn(
        qkv,
        key_padding_mask=key_padding_mask,
        need_weights=False,
        causal=layer.self_attn.causal,
    )
    projected = layer.self_attn.out_proj(rearrange(context, "b s h d -> b s (h d)"))
    return projected, qkv_before, qkv_after, context.detach().cpu().numpy()


def _run_layer_with_attn(layer, total_embs, src_key_padding_mask):
    if src_key_padding_mask.dtype != torch.bool:
        src_key_padding_mask = src_key_padding_mask.bool()
    flash_mask = ~src_key_padding_mask

    if layer.norm_scheme != "post":
        raise NotImplementedError("Only post-norm layers are supported.")

    src2, qkv_before, qkv_after, context = _run_attn(layer, total_embs, key_padding_mask=flash_mask)
    src = total_embs + layer.dropout1(src2)
    src = layer.norm1(src)
    src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
    src = src + layer.dropout2(src2)
    src = layer.norm2(src)
    return src, qkv_before, qkv_after, context


def extract_layerwise_scgpt_embeddings(
    model,
    data_pt: dict,
    *,
    layers: Sequence[int] = tuple(range(12)),
    batch_size: int = 64,
    pad_value: int = -2,
    use_batch_norm: bool = True,
    capture_before_transformer: bool = True,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    layer_list = list(layers)
    all_gene_ids = data_pt["gene_ids"]
    all_values = data_pt["values"]
    src_key_padding_mask = all_values.eq(pad_value)

    if len(layer_list) == 0:
        raise ValueError("layers must contain at least one layer index")

    n_cells = all_gene_ids.size(0)
    embsize = model.encoder.embedding_dim
    n_head = model.transformer_encoder.layers[0].self_attn.num_heads
    head_dim = embsize // n_head

    outputs = np.zeros((len(layer_list), n_cells, embsize), dtype=np.float32)
    qkv_before_out = np.zeros((len(layer_list), n_cells, 3 * embsize), dtype=np.float32)
    qkv_after_out = np.zeros((len(layer_list), n_cells, 3, n_head, head_dim), dtype=np.float32)
    context_out = np.zeros((len(layer_list), n_cells, n_head, head_dim), dtype=np.float32)
    before_transformer = np.zeros((n_cells, embsize), dtype=np.float32) if capture_before_transformer else None

    model.eval()
    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enabled):
        for start in tqdm(range(0, n_cells, batch_size), desc="Extract scGPT embeddings"):
            stop = min(start + batch_size, n_cells)
            batch_gene_ids = all_gene_ids[start:stop].to(device)
            batch_values = all_values[start:stop].to(device)
            batch_mask = src_key_padding_mask[start:stop].to(device)

            src_embs = model.encoder(batch_gene_ids)
            val_embs = model.value_encoder(batch_values)
            total_embs = src_embs + val_embs
            if use_batch_norm:
                total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

            if before_transformer is not None:
                before_transformer[start:stop] = total_embs[:, 0, :].detach().cpu().numpy()

            last_layer = max(layer_list)
            for layer_idx, layer in enumerate(model.transformer_encoder.layers[: last_layer + 1]):
                total_embs, qkv_before, qkv_after, context = _run_layer_with_attn(layer, total_embs, batch_mask)
                if layer_idx in layer_list:
                    target_idx = layer_list.index(layer_idx)
                    outputs[target_idx, start:stop] = total_embs[:, 0, :].detach().cpu().numpy()
                    qkv_before_out[target_idx, start:stop] = qkv_before[:, 0]
                    qkv_after_out[target_idx, start:stop] = qkv_after[:, 0]
                    context_out[target_idx, start:stop] = context[:, 0]

    result = {
        "layers": outputs,
        "qkv_before": qkv_before_out,
        "qkv_after": qkv_after_out,
        "context": context_out,
        "layer_indices": np.asarray(layer_list, dtype=np.int64),
    }
    if before_transformer is not None:
        result["before_transformer"] = before_transformer
    return result


def align_embeddings_to_adata(
    source_obs_names: Sequence[str],
    target_adata,
    embeddings: np.ndarray,
) -> np.ndarray:
    indexer = pd.Index(source_obs_names).get_indexer(target_adata.obs_names)
    if (indexer < 0).any():
        missing = target_adata.obs_names[indexer < 0].tolist()
        raise ValueError(f"{len(missing)} cells from target_adata are missing in source embeddings, e.g. {missing[:5]}")
    return embeddings[indexer]


def extract_raw_embeddings_for_subset(
    raw_adata,
    subset_adata,
    model_dir: str | Path,
    *,
    hvg_n: int = 2000,
    params: dict | None = None,
    batch_size: int | None = None,
    output_prefix: str | Path | None = None,
) -> dict[str, Any]:
    model, vocab, merged = load_pretrained_model(model_dir, params=params)
    if batch_size is not None:
        merged["batch_size"] = batch_size

    raw_hvg = build_hvg_subset(raw_adata, hvg_n, gene_col=merged["gene_col"])
    subset_hvg = build_hvg_subset(subset_adata, hvg_n, gene_col=merged["gene_col"])

    processed_raw = preprocess_for_scgpt(raw_hvg, params=merged)
    processed_raw = _filter_to_vocab(processed_raw, vocab, gene_col=merged["gene_col"])
    tokenized = tokenize_adata(
        processed_raw,
        vocab,
        params=merged,
        gene_col=merged["gene_col"],
        layer_key=merged["input_layer_key"],
        include_zero=merged["include_zero_gene"],
    )
    bundle = extract_layerwise_scgpt_embeddings(
        model,
        tokenized,
        layers=tuple(range(merged["nlayers"])),
        batch_size=merged["batch_size"],
        pad_value=merged["pad_value"],
        use_batch_norm=True,
        capture_before_transformer=True,
    )

    aligned = {
        "raw_obs_names": np.asarray(processed_raw.obs_names),
        "subset_obs_names": np.asarray(subset_hvg.obs_names),
        "layer_indices": bundle["layer_indices"],
        "subset_layers": align_embeddings_to_adata(processed_raw.obs_names, subset_hvg, bundle["layers"]),
        "subset_before_transformer": align_embeddings_to_adata(
            processed_raw.obs_names, subset_hvg, bundle["before_transformer"]
        ),
        "subset_qkv_before": bundle["qkv_before"][:, pd.Index(processed_raw.obs_names).get_indexer(subset_hvg.obs_names)],
        "subset_qkv_after": bundle["qkv_after"][:, pd.Index(processed_raw.obs_names).get_indexer(subset_hvg.obs_names)],
        "subset_context": bundle["context"][:, pd.Index(processed_raw.obs_names).get_indexer(subset_hvg.obs_names)],
        "raw_bundle": bundle,
    }
    if output_prefix is not None:
        save_embedding_bundle(aligned, output_prefix)
    return aligned


def save_embedding_bundle(bundle: dict[str, Any], output_prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_payload = {}
    meta_payload = {}
    for key, value in bundle.items():
        if isinstance(value, np.ndarray):
            npz_payload[key] = value
        elif isinstance(value, dict):
            nested_prefix = prefix.with_name(prefix.name + f"_{key}")
            save_embedding_bundle(value, nested_prefix)
        else:
            meta_payload[key] = value.tolist() if isinstance(value, np.ndarray) else value

    npz_path = prefix.with_suffix(".npz")
    meta_path = prefix.with_suffix(".json")
    if npz_payload:
        np.savez_compressed(npz_path, **npz_payload)
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False, default=str))
    return npz_path, meta_path


def compute_method_umaps_and_metrics(
    adata,
    methods: dict[str, str],
    *,
    label_keys: Sequence[str] = DEFAULT_BIO_LABELS,
    plot_label_keys: Sequence[str] = DEFAULT_UMAP_LABELS,
    output_dir: str | Path | None = None,
    neighbors_kwargs: dict | None = None,
    umap_kwargs: dict | None = None,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    output_dir_path = Path(output_dir) if output_dir is not None else None
    if output_dir_path is not None:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    for method_name, rep_key in methods.items():
        work_adata = compute_representation_umap(
            adata,
            use_rep=rep_key,
            neighbors_kwargs=neighbors_kwargs,
            umap_kwargs=umap_kwargs,
            copy=True,
        )
        metric_table = calculate_avgbio_table(work_adata, "X_umap", label_keys)
        results[method_name] = {
            "adata": work_adata,
            "metrics": metric_table,
            "plot_labels": list(plot_label_keys),
        }
        if output_dir_path is not None:
            work_adata.write_h5ad(output_dir_path / f"{method_name}_umap.h5ad")
            save_avgbio_results(metric_table, output_dir_path / f"{method_name}_avgbio")

    return results


def compute_zero_shot_scgpt_grid(
    adata,
    model_specs: dict[str, str | Path],
    *,
    hvg_sizes: Sequence[int] = (1000, 2000, 3000),
    label_keys: Sequence[str] = DEFAULT_BIO_LABELS,
    output_dir: str | Path | None = None,
    batch_size: int = 128,
    gene_col: str = "gene",
    cell_type_key: str = "celltype_infercnv",
) -> dict[str, dict[str, Any]]:
    """Run the legacy zero-shot grid from Python instead of notebooks.

    `model_specs` can include multiple checkpoints, for example:
    {
        "scgpt_pretrained": "/path/to/whole_human",
        "scgpt_random": "/path/to/randomized",
    }
    """

    output_root = Path(output_dir) if output_dir is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for hvg_n in hvg_sizes:
        subset = build_hvg_subset(adata, hvg_n, gene_col=gene_col)
        for model_name, model_dir in model_specs.items():
            run_name = f"{model_name}_{hvg_n}hvg"
            embedded = embed_adata_with_scgpt(
                subset,
                model_dir,
                gene_col=gene_col,
                cell_type_key=cell_type_key,
                batch_size=batch_size,
            )
            run_dir = output_root / run_name if output_root is not None else None
            method_result = compute_method_umaps_and_metrics(
                embedded,
                methods={run_name: "X_scGPT"},
                label_keys=label_keys,
                output_dir=run_dir,
            )
            results[run_name] = method_result[run_name]
    return results


def compute_pca_harmony_baselines(
    adata,
    *,
    label_keys: Sequence[str] = DEFAULT_BIO_LABELS,
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    methods = {}
    for method_name, rep_key in (("pca", "X_pca"), ("harmony", "X_harmony")):
        if rep_key in adata.obsm:
            methods[method_name] = rep_key
    if not methods:
        raise ValueError("None of the baseline representations ('X_pca', 'X_harmony') were found.")
    return compute_method_umaps_and_metrics(
        adata,
        methods=methods,
        label_keys=label_keys,
        output_dir=output_dir,
    )
