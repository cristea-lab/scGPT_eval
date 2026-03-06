from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/cristealab/rtan/scGPT/Dan/data")
SAVE_DIR = Path("/cristealab/rtan/scGPT/manuscript/save")
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_LABEL_KEY = "Level.2.Annotation"
DEFAULT_SAMPLE_KEY = "sampleid"
DEFAULT_K = 10

# The legacy notebook and presentation describe the five reference patients
# as 003, 004, MGHR1, 011, and 2540. We normalize numeric ids before matching.
DEFAULT_REFERENCE_SAMPLE_IDS = ("3", "4", "11", "2540", "MGHR1")

PYTHON_METHODS = {
    "harmony_pca": {
        "display_name": "Harmony-PCA",
        "embedding_key": "X_harmony",
        "source": "raw_h5ad",
    },
    "scgpt_second_layer": {
        "display_name": "scGPT (second layer)",
        "embedding_key": "X_scGPT_second_layer",
        "source": "novalue_pickle",
        "pickle_index": 1,
    },
}
