from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from label_transfer import (
    DEFAULT_K,
    DEFAULT_LABEL_KEY,
    DEFAULT_REFERENCE_SAMPLE_IDS,
    DEFAULT_RESULTS_DIR,
    PYTHON_METHODS,
    attach_embedding,
    load_python_inputs,
    run_knn_label_transfer,
    save_result_bundle,
    split_reference_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run kNN-based Level 2 label transfer for Harmony-PCA and scGPT embeddings."
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of nearest neighbors.")
    parser.add_argument(
        "--label-key",
        default=DEFAULT_LABEL_KEY,
        help="Annotation column to transfer.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory used to save method-specific result bundles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_python_inputs()
    summary_rows = []

    for method_name, method_spec in PYTHON_METHODS.items():
        adata, embedding_key = attach_embedding(
            inputs["adata"],
            inputs["raw"],
            inputs["novalue_embed"],
            method_spec,
        )
        ref, query = split_reference_query(
            adata,
            reference_sample_ids=DEFAULT_REFERENCE_SAMPLE_IDS,
        )
        predictions = run_knn_label_transfer(
            ref,
            query,
            embedding_key=embedding_key,
            label_key=args.label_key,
            k=args.k,
        )
        metadata = {
            "method_name": method_name,
            "display_name": method_spec["display_name"],
            "embedding_key": embedding_key,
            "label_key": args.label_key,
            "k": args.k,
            "reference_sample_ids": list(DEFAULT_REFERENCE_SAMPLE_IDS),
            "reference_n_cells": int(ref.n_obs),
            "query_n_cells": int(query.n_obs),
        }
        save_result_bundle(
            predictions=predictions,
            output_dir=results_dir / method_name,
            metadata=metadata,
        )

        metrics_row = json.loads((results_dir / method_name / "metrics.json").read_text())
        summary_rows.append(metrics_row)

    pd.DataFrame(summary_rows).to_csv(results_dir / "python_methods_summary.csv", index=False)


if __name__ == "__main__":
    main()
