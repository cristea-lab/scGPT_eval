from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    truth = predictions["truth"].astype(str)
    pred = predictions["prediction"].astype(str)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro")),
        "macro_precision": float(precision_score(truth, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, average="macro", zero_division=0)),
        "n_cells": int(len(predictions)),
    }


def build_confusion_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = predictions["truth"].astype(str)
    pred = predictions["prediction"].astype(str)
    labels = sorted(set(truth) | set(pred))
    counts = confusion_matrix(truth, pred, labels=labels)

    counts_df = pd.DataFrame(counts, index=labels, columns=labels)
    counts_df.index.name = "truth"
    counts_df.columns.name = "prediction"

    row_sums = counts_df.sum(axis=1).replace(0, 1)
    row_norm = counts_df.div(row_sums, axis=0)
    row_norm.index.name = "truth"
    row_norm.columns.name = "prediction"
    return counts_df, row_norm


def save_result_bundle(
    *,
    predictions: pd.DataFrame,
    output_dir: str | Path,
    metadata: dict,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(predictions)
    counts_df, row_norm_df = build_confusion_tables(predictions)

    payload = {
        **metadata,
        **metrics,
        "labels": counts_df.index.tolist(),
    }

    paths = {
        "metrics_json": out_dir / "metrics.json",
        "metrics_csv": out_dir / "metrics.csv",
        "predictions_csv": out_dir / "predictions.csv",
        "confusion_counts_csv": out_dir / "confusion_matrix_counts.csv",
        "confusion_row_norm_csv": out_dir / "confusion_matrix_row_normalized.csv",
    }

    paths["metrics_json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    pd.DataFrame([payload]).to_csv(paths["metrics_csv"], index=False)
    predictions.to_csv(paths["predictions_csv"], index=False)
    counts_df.to_csv(paths["confusion_counts_csv"])
    row_norm_df.to_csv(paths["confusion_row_norm_csv"])
    return paths
