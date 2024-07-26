"""Generate text with chat completion models from the OpenAI API.

I used the following commands to generate text for evaluation:

```bash
python -m src.query_openai_models --input-file data/prompts/test.all.jsonl --output-file data/output/test_gpt3_5_turbo.jsonl --model gpt-3.5-turbo-1106
python -m src.query_openai_models --input-file data/prompts/test.all.jsonl --output-file data/output/test_gpt4_turbo.jsonl --model gpt-4-1106-preview
```
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import socket

from openai import OpenAI
import tiktoken

from src.config import settings
from src.utils import DEFAULT_RANDOM_SEED, return_logger, write_jsonl

logger = return_logger(__name__)

# See https://platform.openai.com/docs/models for description
# See https://openai.com/pricing for prices
GPT3_5_TURBO_MODEL = "gpt-3.5-turbo-1106"
GPT4_TURBO_MODEL = "gpt-4-1106-preview"

DEFAULT_MODEL = GPT3_5_TURBO_MODEL
MODELS = [
    GPT3_5_TURBO_MODEL,
    GPT4_TURBO_MODEL,
]

# This is a general limit to use because the API doesn't provide information on
# the limits. This seems simpler than reproducing the limits from OpenAI docs
# in this file.
MAX_OPENAI_INPUT_TOKENS_PER_PROMPT = 2048

# Set higher than needed to prevent excessive generation
MAX_OPENAI_OUTPUT_TOKENS_PER_PROMPT = 256


@lru_cache(maxsize=64)
def initialize_openai_api(model_name: str = DEFAULT_MODEL) -> tuple[OpenAI, str]:
    """Initializes OpenAI API client and verifies the given model is available.

    Args:
        model_name: Name of model to check for availability.

    Returns:
        An initialized API client and the user to call the API with.
    """
    if not settings.openai_usable:
        raise RuntimeError("OpenAI API key or organization not found")

    client = OpenAI(api_key=settings.openai_key, organization=settings.openai_organization)

    openai_user = socket.getfqdn()

    # Verify selected model is available
    # Also serves as a general check that API works
    client.models.retrieve(model_name)

    return client, openai_user


def generate_chat_completion(
    text: str,
    *,
    model_name: str = DEFAULT_MODEL,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[str | None, dict[str, int | str | None]]:
    """Generate text using chat completion API.

    Args:
        text: Input prompt.
        model_name: Name of model to use for generation.
        seed: Random seed for text generation.

    Returns:
        Generated text and associated generation metadata, if generation succeeded.
    """
    enc = tiktoken.encoding_for_model(model_name)
    logger.debug("Input text: %s", repr(text))
    # The value of 16 was derived using experimentation
    # https://platform.openai.com/docs/guides/text-generation/managing-tokens did not help
    token_count = len(enc.encode(text)) + 16
    logger.debug("%s", f"{token_count = }")
    if token_count > MAX_OPENAI_INPUT_TOKENS_PER_PROMPT:
        logger.warning("Input text too long at %d tokens", token_count)
        return None, {}

    client, openai_user = initialize_openai_api(model_name)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": text},
        ],
        max_tokens=MAX_OPENAI_OUTPUT_TOKENS_PER_PROMPT,
        n=1,
        seed=seed,
        user=openai_user,
    )
    assert response.usage is not None  # For mypy because API is weird
    prompt_tokens = response.usage.prompt_tokens
    if token_count != prompt_tokens:
        logger.warning(
            "Projected prompt token count was %d, but it ended up being %d instead",
            token_count,
            prompt_tokens,
        )
    completion_tokens = response.usage.completion_tokens
    choice = response.choices[0]
    finish_reason = choice.finish_reason
    logger.debug("%s", f"{finish_reason = }")
    if finish_reason != "stop":
        logger.warning("Generation terminated early with reason %s", repr(finish_reason))
    raw_prediction_string = choice.message.content
    logger.debug("%s", f"{raw_prediction_string = }")
    system_fingerprint = response.system_fingerprint
    logger.debug("%s", f"{seed = }, {system_fingerprint = }")

    generation_metadata = {
        "model_name": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "seed": seed,
        "system_fingerprint": system_fingerprint,
    }
    # mypy's type inference is broken for `generation_metadata`
    return raw_prediction_string, generation_metadata  # type: ignore[return-value]


def main() -> None:
    """Entrypoint for OpenAI querying script."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input JSON Lines file.",
    )
    p.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output JSON Lines file.",
    )
    p.add_argument(
        "--model",
        choices=MODELS,
        required=True,
        help="Name of LLM model to use.",
    )
    args = p.parse_args()
    input_file: Path = args.input_file
    output_file: Path = args.output_file
    model_name: str = args.model

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file {str(input_file)!r} does not exist or is not a file.")
    if output_file.exists() and not output_file.is_file():
        raise OSError(f"Output must be a writeable file path, but got {str(output_file)!r}.")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as file:
        input_rows = [json.loads(line) for line in file]

    output_rows = []
    for input_row in input_rows:
        prompt = input_row["prompt"]
        logger.debug({k: v for k, v in input_row.items() if k not in {"prompt", "full"}})
        output_text, generation_metadata = generate_chat_completion(prompt, model_name=model_name)
        output_row = {
            **input_row,
            "generated_text": output_text,
            "generation_metadata": generation_metadata,
        }
        output_rows.append(output_row)

    write_jsonl(output_rows, output_file)


if __name__ == "__main__":
    main()
