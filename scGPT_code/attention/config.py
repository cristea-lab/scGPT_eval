from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Static model hyperparameters copied from the legacy notebooks."""

    gene_col: str = "gene"
    input_layer_key: str = "X_binned"
    max_seq_len: int = 3001
    batch_size: int = 32
    domain_spec_batchnorm: str | bool = "batchnorm"
    input_emb_style: str = "continuous"
    cell_emb_style: str = "cls"
    n_input_bins: int = 51
    mvc_decoder_style: str = "inner product"
    ecs_threshold: float = 0.0
    explicit_zero_prob: bool = False
    use_fast_transformer: bool = True
    fast_transformer_backend: str = "flash"
    pre_norm: bool = False
    n_layers_cls: int = 3
    nhead: int = 8
    embsize: int = 512
    d_hid: int = 512
    nlayers: int = 12
    dropout: float = 0.2
    n_bins: int = 51
    include_zero_gene: bool = False
    pad_token: str = "<pad>"
    cls_token: str = "<cls>"
    mask_value: int = -1
    pad_value: int = -2


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration for the end-to-end extraction and GSEA pipeline.

    Paths default to the locations used in the legacy notebooks so the pipeline
    can be adopted without changing upstream data preparation immediately.
    """

    data_dir: Path = Path("/cristealab/rtan/scGPT/Dan/data")
    model_dir: Path = Path("/cristealab/rtan/scGPT/models/whole_human")
    subraw_filename: str = "subraw.h5ad"
    binned_counts_path: Path = Path("/cristealab/rtan/scGPT/manuscript/save/subraw_X_binned.pickle")
    output_root: Path = Path("/cristealab/rtan/scGPT/manuscript/scGPT_code/attention/outputs")
    annotation_column: str = "Level 1 Annotation"
    max_cells_per_type: int = 100
    sample_seed: int = 42
    excluded_cell_types: tuple[str, ...] = (
        "Pericyte",
        "Vascular smooth muscle",
        "Schwann",
        "Adipocyte",
        "Intra-pancreatic neurons",
    )
    attention_layers: tuple[int, ...] = (0,)
    selected_heads: tuple[int, ...] = tuple(range(8))
    attention_batch_size: int = 128
    rank_normalize_attention: bool = True
    zero_pad_scores: bool = True
    matrix_axes: str = "targets_by_rows"
    min_cells_per_token: int = 5
    aggregation_method: str = "top_p"
    top_p: float = 0.30
    gene_sets: str = "KEGG_2021_Human"
    gsea_min_size: int = 5
    gsea_max_size: int = 5000
    gsea_permutations: int = 2000
    gsea_threads: int = 4
    gsea_seed: int = 2024
    model: ModelConfig = field(default_factory=ModelConfig)


DEFAULT_PIPELINE_CONFIG = PipelineConfig()
