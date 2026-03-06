from __future__ import annotations

import gc
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import PipelineConfig


def _resolve_vocab_token_lookup(vocab, token_id: int) -> str:
    """Convert a token id into a readable token string if the vocab exposes one."""

    if hasattr(vocab, "lookup_token"):
        return str(vocab.lookup_token(int(token_id)))
    if hasattr(vocab, "itos"):
        return str(vocab.itos[int(token_id)])
    if hasattr(vocab, "idx_to_token"):
        return str(vocab.idx_to_token[int(token_id)])
    return str(int(token_id))


def extract_multihead_attention(
    model,
    data_pt: dict[str, torch.Tensor | list[str]],
    layers: list[int] | tuple[int, ...],
    config: PipelineConfig,
    device: str,
):
    """
    Extract head-specific attention matrices in a format compatible with the
    legacy downstream GSEA analysis.

    The returned structure matches the notebooks:
    `results[layer][head] -> list[(max_scores, tokens, label, values, matrix)]`.
    Keeping this layout makes the GSEA stage easy to validate against the legacy
    outputs even though the implementation is now modular.
    """

    model.eval()
    model.use_fast_transformer = False
    model.fast_transformer = False
    for layer in model.transformer_encoder.layers:
        if hasattr(layer, "fast_attn"):
            layer.fast_attn = False
    model.to(device)

    vocab = getattr(model, "vocab", None)
    if vocab is None:
        raise ValueError("The scGPT model must expose its vocabulary for token decoding.")

    try:
        pad_id = int(vocab[config.model.pad_token])
    except Exception as exc:
        raise ValueError("Could not resolve the pad token id from the vocabulary.") from exc

    all_gene_ids = data_pt["gene_ids"]
    all_values = data_pt["values"]
    all_labels = data_pt.get("labels")

    if not isinstance(all_gene_ids, torch.Tensor):
        all_gene_ids = torch.as_tensor(all_gene_ids, dtype=torch.long)
    else:
        all_gene_ids = all_gene_ids.detach().clone().long()

    if isinstance(all_values, torch.Tensor):
        all_values_np = all_values.detach().cpu().numpy()
    else:
        all_values_np = np.asarray(all_values)

    src_key_padding_mask_all = all_gene_ids.eq(pad_id)
    num_cells, seq_len = all_gene_ids.shape

    first_layer = model.transformer_encoder.layers[0]
    num_heads = int(getattr(first_layer.self_attn, "num_heads", getattr(model, "nhead", 8)))
    selected_layers = sorted({int(layer) for layer in layers})
    max_selected_layer = max(selected_layers)
    results = {layer: {head: [] for head in range(num_heads)} for layer in selected_layers}

    autocast_enabled = device.startswith("cuda") and torch.cuda.is_available()
    autocast_context = torch.amp.autocast(device_type="cuda", enabled=True) if autocast_enabled else nullcontext()

    with torch.no_grad(), autocast_context:
        for start in range(0, num_cells, config.attention_batch_size):
            stop = min(start + config.attention_batch_size, num_cells)
            batch_size = stop - start

            gene_ids = all_gene_ids[start:stop].to(device)
            values = torch.as_tensor(all_values_np[start:stop], dtype=torch.float32, device=device)
            padding_mask = src_key_padding_mask_all[start:stop].to(device)

            hidden = model.encoder(gene_ids) + model.value_encoder(values)
            if hasattr(model, "bn") and model.bn is not None:
                hidden = model.bn(hidden.permute(0, 2, 1)).permute(0, 2, 1)

            for layer_index, encoder_layer in enumerate(model.transformer_encoder.layers):
                if hasattr(encoder_layer.self_attn, "Wqkv") and hasattr(encoder_layer.self_attn.Wqkv, "weight"):
                    attention_dtype = encoder_layer.self_attn.Wqkv.weight.dtype
                elif hasattr(encoder_layer.self_attn, "in_proj_weight"):
                    attention_dtype = encoder_layer.self_attn.in_proj_weight.dtype
                else:
                    attention_dtype = hidden.dtype

                hidden_for_layer = hidden.to(attention_dtype)

                if layer_index in results:
                    qkv = _project_qkv(encoder_layer.self_attn, hidden_for_layer)
                    qkv = qkv.view(batch_size, seq_len, 3, num_heads, -1)

                    for head_index in range(num_heads):
                        query = qkv[:, :, 0, head_index, :]
                        key = qkv[:, :, 1, head_index, :]
                        attention_scores = torch.bmm(query, key.transpose(1, 2))

                        if config.rank_normalize_attention:
                            attention_scores = _rank_normalize_attention(attention_scores, seq_len)

                        matrix = attention_scores.transpose(1, 2)

                        for batch_offset in range(batch_size):
                            cell_index = start + batch_offset
                            token_ids = all_gene_ids[cell_index].cpu().numpy()
                            token_strings = [_resolve_vocab_token_lookup(vocab, token_id) for token_id in token_ids]
                            matrix_np = matrix[batch_offset].float().cpu().numpy()
                            max_scores = matrix[batch_offset].max(dim=1)[0].float().cpu().numpy()

                            if config.zero_pad_scores:
                                pad_mask = src_key_padding_mask_all[cell_index].cpu().numpy()
                                max_scores[pad_mask] = 0.0

                            label = None if all_labels is None else all_labels[cell_index]
                            results[layer_index][head_index].append(
                                (
                                    max_scores,
                                    token_strings,
                                    label,
                                    all_values_np[cell_index].copy(),
                                    matrix_np,
                                )
                            )

                        del attention_scores, matrix, query, key
                        if autocast_enabled:
                            torch.cuda.empty_cache()

                    del qkv

                hidden = encoder_layer(hidden_for_layer, src_key_padding_mask=padding_mask)
                if layer_index >= max_selected_layer:
                    break

            del gene_ids, values, padding_mask, hidden
            if autocast_enabled:
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    return results


def _project_qkv(self_attention_module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Support both flash-attention and standard PyTorch multi-head attention backends."""

    if hasattr(self_attention_module, "Wqkv"):
        return self_attention_module.Wqkv(hidden_states)
    if hasattr(self_attention_module, "in_proj_weight"):
        return F.linear(
            hidden_states,
            self_attention_module.in_proj_weight,
            self_attention_module.in_proj_bias,
        )
    raise RuntimeError("Unsupported attention implementation: missing QKV projection weights.")


def _rank_normalize_attention(attention_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Match the legacy notebook's per-query rank normalization.

    The normalization is intentionally kept, because the downstream ranking and
    GSEA results were built on these normalized values.
    """

    attention_scores = attention_scores.float()
    flat_scores = attention_scores.reshape(-1, seq_len)
    order = torch.argsort(flat_scores, dim=1)
    rank = torch.argsort(order, dim=1)
    return rank.reshape(attention_scores.shape).float() / float(seq_len)


def save_attention_results(results: dict[int, dict[int, list]], output_dir: Path):
    """Persist extracted attention results using the legacy file naming scheme."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for layer, layer_results in results.items():
        with open(output_dir / f"examples_scores_attention_layer{layer}.p", "wb") as handle:
            pickle.dump(layer_results, handle)
