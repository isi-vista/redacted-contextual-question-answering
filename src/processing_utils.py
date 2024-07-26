"""Common logic shared by data processing scripts."""

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import re
from string import Template
from typing import Any

DEFAULT_NUM_EXAMPLES = 500
DEFAULT_RANDOM_SEED = 0

PROMPT_TEMPLATE = Template(
    "I will provide you a sentence, a word, and a definition, "
    'and you need to tell me "Yes" if the word and definition applies to an event in the sentence '
    'and "No" if it does not.\n'
    "Sentence: ${sentence}\n"
    "Word: ${word}\n"
    "Definition: ${definition}\n"
    "Answer: ${answer}"
)


def clean_text(text: str) -> str:
    """Removes unwanted substrings from text."""
    text = re.sub(r"\[unused\d+]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def convert_row_to_example(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert row of raw data into a prompt that can be used for training.

    Args:
        row: Row of raw data.

    Returns:
        A dict containing the prompt and its corresponding label.
    """
    prompt = PROMPT_TEMPLATE.substitute(
        sentence=clean_text(row["text"]),
        word=row["title"],
        definition=row["description"],
        answer="Yes" if row["label"] else "No",
    )
    return {"text": prompt, "label": row["label"]}


def read_jsonl(path: Path) -> Iterable[Any]:
    """Read data in JSON Lines format.

    Args:
        path: Path to file.
    """
    with path.open(mode="r", encoding="utf-8") as jsonl_in:
        for line in jsonl_in:
            yield json.loads(line)


def write_jsonl(rows: Iterable[Any], path: Path, **kwargs: Any) -> None:
    """Write data in JSON Lines format.

    Args:
        rows: List of JSON-serializable values to write to file.
        path: Path to file.
        kwargs: Additional parameters to pass to `json.dumps()`.
    """
    # `separators` is used here to write JSON compactly
    lines = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), **kwargs) for row in rows)
    output = "\n".join(lines) + "\n"
    path.write_text(output, encoding="utf-8")
