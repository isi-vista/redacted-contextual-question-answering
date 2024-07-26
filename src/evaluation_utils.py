"""Load model and run evaluation metrics."""

from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer, Pipeline, PreTrainedTokenizer

from src.paths import TRANSFORMERS_CACHE_DIR

DEVICE_AUTO = "auto"
CUDA = "cuda"

STR_TO_TORCH_DTYPE = {
    "float": torch.float,
    "double": torch.double,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_transformers_pipeline(
    model: str, dtype_str: str, device: str, cache_dir: Path = TRANSFORMERS_CACHE_DIR
) -> tuple[PreTrainedTokenizer, Pipeline]:
    """Loads model and builds a transformers pipeline for text-generation.

    Args:
        model: The name or path of model to load.
        dtype_str: The torch data type to use, represented as a string.
        device: The device to run inference on.
        cache_dir: Directory to store files downloaded from the Hugging Face Hub.

    Returns:
        A Tokenizer and Pipeline for text generation.
    """
    kw_args: dict[str, Any] = {}
    if device == DEVICE_AUTO:
        kw_args["device_map"] = DEVICE_AUTO
    else:
        kw_args["device"] = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=STR_TO_TORCH_DTYPE[dtype_str],
        model_kwargs={"cache_dir": str(cache_dir)},
        **kw_args,
    )
    return tokenizer, pipeline
