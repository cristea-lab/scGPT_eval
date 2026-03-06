from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from attention.attention import extract_multihead_attention, save_attention_results
from attention.config import DEFAULT_PIPELINE_CONFIG
from attention.data import adata_to_model_inputs, filter_genes_to_vocab, load_subraw_adata, sample_cells_per_annotation
from attention.gsea import build_ranked_gene_lists, load_attention_results, run_preranked_gsea_for_head
from attention.modeling import build_model, load_pretrained_weights, load_vocab


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the manuscript pipeline."""

    parser = argparse.ArgumentParser(
        description="Extract scGPT attention, build CLS-centric ranked gene lists, and run preranked GSEA.",
    )
    parser.add_argument(
        "--model-init",
        choices=("pretrained", "random"),
        required=True,
        help="Choose whether to load the pretrained checkpoint or keep the model randomly initialized.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG.output_root,
        help="Directory that will contain both attention pickles and GSEA outputs.",
    )
    parser.add_argument(
        "--max-cells-per-type",
        type=int,
        default=DEFAULT_PIPELINE_CONFIG.max_cells_per_type,
        help="Maximum number of sampled cells for each Level 1 annotation.",
    )
    parser.add_argument(
        "--attention-batch-size",
        type=int,
        default=DEFAULT_PIPELINE_CONFIG.attention_batch_size,
        help="Batch size used during attention extraction.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=list(DEFAULT_PIPELINE_CONFIG.selected_heads),
        help="Attention heads to analyze. Defaults to all heads in layer 0.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_PIPELINE_CONFIG.attention_layers),
        help="Transformer layers to extract attention from.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for model inference.",
    )
    parser.add_argument(
        "--cls-token",
        default="60695",
        help="CLS token string expected in the saved attention output. The legacy notebooks used token id strings.",
    )
    parser.add_argument(
        "--pad-token-id",
        default="60694",
        help="PAD token string expected in the saved attention output. The legacy notebooks used token id strings.",
    )
    return parser.parse_args()


def main():
    """
    Run the complete manuscript pipeline for either pretrained or random scGPT.

    The flow is:
    1. Load and sample the manuscript subset data.
    2. Build the scGPT model.
    3. Extract head-resolved attention matrices and save them.
    4. Convert CLS-centric attention into ranked gene lists.
    5. Run preranked KEGG GSEA and save standardized CSV outputs.
    """

    args = parse_args()
    config = replace(
        DEFAULT_PIPELINE_CONFIG,
        output_root=args.output_root,
        max_cells_per_type=args.max_cells_per_type,
        attention_batch_size=args.attention_batch_size,
        selected_heads=tuple(args.heads),
        attention_layers=tuple(args.layers),
    )

    run_name = args.model_init
    attention_output_dir = config.output_root / run_name / "attention"
    ranked_output_dir = config.output_root / run_name / "ranked_gene_lists"
    gsea_output_dir = config.output_root / run_name / "gsea"

    vocab = load_vocab(config.model_dir / "vocab.json")
    adata = load_subraw_adata(config)
    adata = filter_genes_to_vocab(adata, vocab=vocab, gene_column=config.model.gene_col)
    sampled_adata = sample_cells_per_annotation(
        adata,
        annotation_column=config.annotation_column,
        max_cells_per_type=config.max_cells_per_type,
        seed=config.sample_seed,
        excluded_cell_types=config.excluded_cell_types,
    )
    data_pt = adata_to_model_inputs(sampled_adata, vocab=vocab, config=config)

    model = build_model(vocab=vocab, config=config)
    if args.model_init == "pretrained":
        model = load_pretrained_weights(model, config.model_dir / "best_model.pt")

    attention_results = extract_multihead_attention(
        model=model,
        data_pt=data_pt,
        layers=config.attention_layers,
        config=config,
        device=args.device,
    )
    save_attention_results(attention_results, attention_output_dir)

    for layer in config.attention_layers:
        layer_results = load_attention_results(attention_output_dir, layer=layer)
        ranked_paths = build_ranked_gene_lists(
            layer_results=layer_results,
            layer=layer,
            heads=config.selected_heads,
            config=config,
            output_dir=ranked_output_dir,
            cls_token=args.cls_token,
            pad_token=args.pad_token_id,
        )
        for head, head_rankings in ranked_paths.items():
            run_preranked_gsea_for_head(
                head=head,
                ranked_paths=head_rankings,
                vocab=vocab,
                config=config,
                output_dir=gsea_output_dir,
            )

    print(f"Finished pipeline for {run_name}.")
    print(f"Attention outputs: {attention_output_dir}")
    print(f"Ranked gene lists: {ranked_output_dir}")
    print(f"GSEA CSV outputs: {gsea_output_dir}")


if __name__ == "__main__":
    main()
