"""Convert RC-QA data to include a chat-compatible format."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from src.processing_utils import read_jsonl, write_jsonl


class RcQaExample(TypedDict):
    """Format of a single example from the RC-QA JSONL dataset."""

    text: str


class ChatTurn(TypedDict):
    """Format that Hugging Face's apply_chat_template() expects."""

    role: str
    content: str


class ChatExample(TypedDict):
    """Format of a chat-ified example from the RC-QA JSONL dataset."""

    chat: list[ChatTurn]


class UnifiedExample(RcQaExample, ChatExample):
    """Format of a unified example from the RC-QA JSONL dataset."""


def chatify(example: RcQaExample) -> ChatExample:
    """Convert an RC-QA-formatted example to a chat-formatted example."""
    parts = example["text"].split("\n\n")
    system_part = "\n\n".join(parts[:-1])

    user_raw, assistant_raw = parts[-1].split("\n")
    assert user_raw.startswith("Question: ")
    user_part = user_raw.removeprefix("Question: ")
    assert assistant_raw.startswith("Answer: ")
    assistant_part = assistant_raw.removeprefix("Answer: ")

    return {
        "chat": [
            {"role": "system", "content": system_part},
            {"role": "user", "content": user_part},
            {"role": "assistant", "content": assistant_part},
        ]
    }


def unify(example: RcQaExample) -> UnifiedExample:
    """Unify the example with a chat-ified version of the same example."""
    return {
        **example,
        **chatify(example),
    }


def main() -> None:
    """Entrypoint for preprocessing script."""
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
    args = p.parse_args()
    input_file: Path = args.input_file
    output_file: Path = args.output_file

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file {str(input_file)!r} does not exist or is not a file.")
    if output_file.exists() and not output_file.is_file():
        raise OSError(f"Output must be a writeable file path, but got {str(output_file)!r}.")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rcqa_examples: Sequence[RcQaExample] = list(read_jsonl(input_file))
    unified = [unify(rcqa_example) for rcqa_example in rcqa_examples]
    write_jsonl(unified, output_file)


if __name__ == "__main__":
    main()
