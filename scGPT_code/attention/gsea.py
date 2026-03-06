from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

from .config import PipelineConfig


def load_attention_results(attention_dir: Path, layer: int) -> dict[int, list]:
    """Load one layer of head-resolved attention results saved by the extraction stage."""

    with open(attention_dir / f"examples_scores_attention_layer{layer}.p", "rb") as handle:
        return pickle.load(handle)


def build_ranked_gene_lists(
    layer_results: dict[int, list],
    layer: int,
    heads: list[int] | tuple[int, ...],
    config: PipelineConfig,
    output_dir: Path,
    cls_token: str,
    pad_token: str,
):
    """
    Convert CLS-centric attention into ranked gene lists for each label/head pair.

    The attention matrices stored by the extraction stage use the legacy
    `targets_by_rows` convention. In that format, the CLS query lives in the
    CLS column, so selecting that column recovers CLS-to-gene attention.
    """

    ranked_paths: dict[int, dict[str, Path]] = {}

    for head in heads:
        examples = layer_results[head]
        label_gene_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        label_gene_cell_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for _, token_strings, label, _, matrix in examples:
            if label is None:
                continue

            token_strings = [str(token) for token in token_strings]
            cls_index = _find_special_token_index(token_strings, [cls_token, "<cls>"])
            if cls_index is None:
                continue

            gene_scores = _extract_cls_gene_scores(
                matrix=np.asarray(matrix),
                cls_index=cls_index,
                matrix_axes=config.matrix_axes,
            )

            selected_gene_items = _select_gene_scores_for_aggregation(
                tokens=token_strings,
                scores=gene_scores,
                cls_tokens={str(cls_token), "<cls>"},
                pad_tokens={str(pad_token), "<pad>"},
                method=config.aggregation_method,
                top_p=config.top_p,
            )

            seen_genes_in_cell: set[str] = set()
            for token, score in selected_gene_items:
                label_gene_sums[str(label)][token] += float(score)
                if token not in seen_genes_in_cell:
                    label_gene_cell_counts[str(label)][token] += 1
                    seen_genes_in_cell.add(token)

        head_dir = output_dir / f"layer{layer}" / f"h{head}"
        head_dir.mkdir(parents=True, exist_ok=True)
        ranked_paths[head] = {}

        for label, gene_scores in label_gene_sums.items():
            filtered_scores = {
                gene: score
                for gene, score in gene_scores.items()
                if label_gene_cell_counts[label][gene] >= config.min_cells_per_token
            }
            ranking = pd.DataFrame(
                sorted(filtered_scores.items(), key=lambda item: item[1], reverse=True),
                columns=["gene_identifier", "score"],
            )

            ranking_path = head_dir / f"{_sanitize_label(label)}.rnk"
            ranking.to_csv(ranking_path, sep="\t", header=False, index=False)
            ranked_paths[head][label] = ranking_path

    return ranked_paths


def _extract_cls_gene_scores(matrix: np.ndarray, cls_index: int, matrix_axes: str) -> np.ndarray:
    """Return the CLS-to-target gene scores from one saved attention matrix."""

    if matrix_axes == "targets_by_rows":
        return matrix[:, cls_index]
    if matrix_axes == "queries_by_rows":
        return matrix[cls_index, :]
    raise ValueError(f"Unsupported matrix_axes value: {matrix_axes}")


def _find_special_token_index(tokens: list[str], special_tokens: list[str]) -> int | None:
    """Support special tokens stored either as literal tokens or as numeric token ids."""

    for special_token in special_tokens:
        if special_token in tokens:
            return tokens.index(special_token)

    return None


def _select_gene_scores_for_aggregation(
    tokens: list[str],
    scores: np.ndarray,
    cls_tokens: set[str],
    pad_tokens: set[str],
    method: str,
    top_p: float,
):
    """
    Select the per-cell gene scores that contribute to the cross-cell ranking.

    The legacy utility exposed `method="top_p"` with `top_p=0.30`. This implementation
    preserves that behavior by keeping only the strongest fraction of genes per
    cell before aggregating scores by cell type.
    """

    valid_items = [
        (token, float(score))
        for token, score in zip(tokens, scores)
        if token not in cls_tokens and token not in pad_tokens
    ]

    if method == "all":
        return valid_items

    if method != "top_p":
        raise ValueError(f"Unsupported aggregation method: {method}")

    if not valid_items:
        return []

    valid_items = sorted(valid_items, key=lambda item: item[1], reverse=True)
    keep_count = max(1, int(np.ceil(len(valid_items) * top_p)))
    return valid_items[:keep_count]


def convert_ranked_list_to_symbols(ranking_path: Path, vocab) -> Path:
    """
    Convert a ranked list keyed by token identifier into a symbol-based ranked list.

    The legacy pipeline sometimes wrote numeric token ids and sometimes readable
    gene tokens, depending on how the scGPT vocabulary exposed reverse lookup.
    This helper accepts both forms so the downstream GSEA step stays stable.
    """

    ranking = pd.read_csv(
        ranking_path,
        sep=r"\s+|,|\t",
        header=None,
        engine="python",
        names=["gene_identifier", "score"],
    )

    symbols = [_token_identifier_to_symbol(value, vocab) for value in ranking["gene_identifier"].tolist()]
    symbol_ranking = pd.DataFrame(
        {
            "gene_symbol": symbols,
            "score": pd.to_numeric(ranking["score"], errors="coerce"),
        }
    ).dropna(subset=["gene_symbol", "score"])

    symbol_ranking = symbol_ranking.groupby("gene_symbol", as_index=False)["score"].max()
    symbol_path = ranking_path.with_name(f"{ranking_path.stem}_symbols.rnk")
    symbol_ranking.to_csv(symbol_path, sep="\t", header=False, index=False)
    return symbol_path


def _token_identifier_to_symbol(token_identifier, vocab) -> str | None:
    """Convert one ranked-list token identifier into a gene symbol."""

    value = str(token_identifier)
    if hasattr(vocab, "lookup_token"):
        try:
            return str(vocab.lookup_token(int(value)))
        except Exception:
            pass

    if value in {None, "", "nan"}:
        return None
    return value


def run_preranked_gsea_for_head(
    head: int,
    ranked_paths: dict[str, Path],
    vocab,
    config: PipelineConfig,
    output_dir: Path,
):
    """Run KEGG preranked GSEA for one head across all cell-type labels."""

    result_paths: dict[str, Path] = {}

    for label, ranking_path in ranked_paths.items():
        symbol_ranking_path = convert_ranked_list_to_symbols(ranking_path, vocab)
        label_dir = output_dir / f"h{head}" / _sanitize_label(str(label))
        label_dir.mkdir(parents=True, exist_ok=True)

        prerank_result = gp.prerank(
            rnk=str(symbol_ranking_path),
            gene_sets=config.gene_sets,
            threads=config.gsea_threads,
            permutation_num=config.gsea_permutations,
            min_size=config.gsea_min_size,
            max_size=config.gsea_max_size,
            seed=config.gsea_seed,
            outdir=str(label_dir),
            format="png",
        )

        result_table = normalize_gsea_result_table(prerank_result.res2d, head=head, label=label)
        result_path = label_dir / f"h{head}__{_sanitize_label(str(label))}__kegg_prerank_results.csv"
        result_table.to_csv(result_path, index=False)
        result_paths[label] = result_path

    return result_paths


def normalize_gsea_result_table(result_table: pd.DataFrame, head: int, label: str) -> pd.DataFrame:
    """Normalize GSEA output columns into a consistent manuscript-friendly schema."""

    result_table = result_table.copy()
    if "Term" not in result_table.columns:
        result_table = result_table.reset_index().rename(columns={"index": "Term"})

    rename_map = {
        "es": "ES",
        "nes": "NES",
        "P-value": "NOM p-val",
        "pval": "NOM p-val",
        "FDR": "FDR q-val",
        "FDR q-value": "FDR q-val",
        "fdr": "FDR q-val",
        "Lead Genes": "Lead_genes",
    }
    result_table = result_table.rename(columns={key: value for key, value in rename_map.items() if key in result_table.columns})

    required_columns = ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val", "Tag %", "Gene %", "Lead_genes"]
    for column in required_columns:
        if column not in result_table.columns:
            result_table[column] = np.nan

    if "Name" not in result_table.columns:
        result_table["Name"] = result_table["Term"]

    result_table.insert(0, "label", label)
    result_table.insert(0, "head", f"h{head}")

    ordered_columns = [
        "head",
        "label",
        "Name",
        "Term",
        "ES",
        "NES",
        "NOM p-val",
        "FDR q-val",
        "FWER p-val",
        "Tag %",
        "Gene %",
        "Lead_genes",
    ]
    return result_table[ordered_columns]


def _sanitize_label(label: str) -> str:
    """Create a filesystem-friendly label while keeping it human-readable."""

    return str(label).replace("/", "_")
