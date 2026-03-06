from __future__ import annotations

from pathlib import Path

import torch

from .config import PipelineConfig


def load_vocab(vocab_path: Path):
    """Load the scGPT vocabulary used by the original model."""

    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    return GeneVocab.from_file(vocab_path)


def build_model(vocab, config: PipelineConfig):
    """
    Build the TransformerModel with the same architecture as the legacy notebooks.

    The pipeline keeps the original backbone intact; only the orchestration code
    changes.
    """

    from scgpt.model import TransformerModel

    model = TransformerModel(
        len(vocab),
        config.model.embsize,
        config.model.nhead,
        config.model.d_hid,
        config.model.nlayers,
        vocab=vocab,
        pad_token=config.model.pad_token,
        pad_value=config.model.pad_value,
        domain_spec_batchnorm=config.model.domain_spec_batchnorm,
        n_input_bins=config.model.n_input_bins,
        use_fast_transformer=config.model.use_fast_transformer,
        pre_norm=config.model.pre_norm,
    )

    # The manuscript workflow does not provide batch labels. The legacy notebooks
    # disabled domain-specific batch norm manually for the same reason.
    model.domain_spec_batchnorm = False
    return model


def load_pretrained_weights(model, model_path: Path):
    """
    Load only compatible parameters from the pretrained checkpoint.

    This preserves the legacy notebook behavior and avoids size mismatches if the
    installed scGPT package changes minor implementation details.
    """

    model_state = model.state_dict()
    pretrained_state = torch.load(model_path, map_location="cpu")
    compatible_state = {
        name: value
        for name, value in pretrained_state.items()
        if name in model_state and tuple(value.shape) == tuple(model_state[name].shape)
    }
    model_state.update(compatible_state)
    model.load_state_dict(model_state)
    return model
