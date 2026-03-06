library(Seurat)
library(dplyr)
library(jsonlite)
library(yardstick)

normalize_sample_id <- function(x) {
  text <- toupper(trimws(as.character(x)))
  text <- sub("\\.0$", "", text)
  if (grepl("^[0-9]+$", text)) {
    return(as.character(as.integer(text)))
  }
  text
}

build_confusion_tables <- function(truth, prediction) {
  labels <- sort(union(as.character(truth), as.character(prediction)))
  truth_f <- factor(as.character(truth), levels = labels)
  prediction_f <- factor(as.character(prediction), levels = labels)

  counts <- table(truth = truth_f, prediction = prediction_f)
  row_sums <- rowSums(counts)
  safe_row_sums <- ifelse(row_sums == 0, 1, row_sums)
  row_norm <- sweep(counts, 1, safe_row_sums, "/")

  list(
    counts = as.data.frame.matrix(counts),
    row_norm = as.data.frame.matrix(row_norm),
    labels = labels
  )
}

save_result_bundle <- function(predictions, out_dir, metadata) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  labels <- sort(union(as.character(predictions$truth), as.character(predictions$prediction)))
  truth_f <- factor(as.character(predictions$truth), levels = labels)
  pred_f <- factor(as.character(predictions$prediction), levels = labels)
  eval_tbl <- tibble(truth = truth_f, prediction = pred_f)

  metrics <- list(
    accuracy = unname(mean(pred_f == truth_f)),
    macro_f1 = unname(f_meas(eval_tbl, truth = truth, estimate = prediction, estimator = "macro")$.estimate),
    macro_precision = unname(precision(eval_tbl, truth = truth, estimate = prediction, estimator = "macro")$.estimate),
    macro_recall = unname(recall(eval_tbl, truth = truth, estimate = prediction, estimator = "macro")$.estimate),
    n_cells = nrow(predictions)
  )
  payload <- c(metadata, metrics, list(labels = labels))
  metrics_row <- data.frame(
    method_name = metadata$method_name,
    display_name = metadata$display_name,
    embedding_key = metadata$embedding_key,
    label_key = metadata$label_key,
    k = metadata$k,
    reference_sample_ids = paste(metadata$reference_sample_ids, collapse = ";"),
    reference_n_cells = metadata$reference_n_cells,
    query_n_cells = metadata$query_n_cells,
    accuracy = metrics$accuracy,
    macro_f1 = metrics$macro_f1,
    macro_precision = metrics$macro_precision,
    macro_recall = metrics$macro_recall,
    n_cells = metrics$n_cells,
    stringsAsFactors = FALSE
  )

  tables <- build_confusion_tables(truth_f, pred_f)
  write_json(payload, file.path(out_dir, "metrics.json"), pretty = TRUE, auto_unbox = TRUE)
  write.csv(metrics_row, file.path(out_dir, "metrics.csv"), row.names = FALSE)
  write.csv(predictions, file.path(out_dir, "predictions.csv"), row.names = FALSE)

  counts_df <- tibble::rownames_to_column(tables$counts, var = "truth")
  row_norm_df <- tibble::rownames_to_column(tables$row_norm, var = "truth")
  write.csv(counts_df, file.path(out_dir, "confusion_matrix_counts.csv"), row.names = FALSE)
  write.csv(row_norm_df, file.path(out_dir, "confusion_matrix_row_normalized.csv"), row.names = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
results_dir <- if (length(args) >= 1) args[[1]] else "results"
label_key <- if (length(args) >= 2) args[[2]] else "Level.2.Annotation"
k_neighbors <- if (length(args) >= 3) as.integer(args[[3]]) else 10L

data_dir <- "/cristealab/rtan/scGPT/Dan/data"
reference_ids <- c("3", "4", "11", "2540", "MGHR1")

snrna <- readRDS(file.path(data_dir, "harmony.RDS"))
snrna$sampleid_normalized <- vapply(snrna$sampleid, normalize_sample_id, character(1))

ref <- subset(snrna, subset = sampleid_normalized %in% reference_ids)
query <- subset(snrna, subset = !sampleid_normalized %in% reference_ids)

ref <- ScaleData(ref) |> RunPCA(npcs = 21, verbose = FALSE)
query <- ScaleData(query) |> RunPCA(npcs = 21, verbose = FALSE)

anchors <- FindTransferAnchors(
  reference = ref,
  query = query,
  reference.reduction = "pca",
  reduction = "pcaproject",
  dims = 1:21,
  normalization.method = "LogNormalize"
)

pred <- TransferData(
  anchorset = anchors,
  refdata = ref[[label_key]][, 1],
  weight.reduction = query[["pca"]],
  dims = 1:21,
  k.weight = k_neighbors
)

query <- AddMetaData(query, pred)
predictions <- data.frame(
  cell_id = colnames(query),
  truth = as.character(query[[label_key]][, 1]),
  prediction = as.character(query$predicted.id),
  stringsAsFactors = FALSE
)

metadata <- list(
  method_name = "seurat_label_transfer",
  display_name = "Seurat label transfer",
  embedding_key = "pca_projected",
  label_key = label_key,
  k = k_neighbors,
  reference_sample_ids = reference_ids,
  reference_n_cells = ncol(ref),
  query_n_cells = ncol(query)
)

save_result_bundle(
  predictions = predictions,
  out_dir = file.path(results_dir, "seurat_label_transfer"),
  metadata = metadata
)
