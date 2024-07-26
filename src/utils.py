"""Common logic shared by scripts."""

from collections.abc import Iterable
import csv
import json
import logging
from pathlib import Path
from typing import Any

DEFAULT_RANDOM_SEED = 0

NO_ANSWER_TEXT = "I am unable to answer this question."

CONSTRAINT_TO_LABEL = {
    "": "no constraint",
    "- Do not include the name of any person or place": "no name",
    "- Never mention more than two characters": "not >2 characters",
    "- Do not mention injury or death": "no injury/death",
}


class TsvDialect(csv.unix_dialect):
    """Custom dialect for easy-to-read TSV files."""

    delimiter = "\t"
    quoting = csv.QUOTE_MINIMAL


def return_logger(name: str) -> logging.Logger:
    """Returns a properly set up logger.

    Args:
        name: Name of logger.

    Returns:
        An initialized logger.
    """
    # Monkey patch module to show milliseconds
    logging.Formatter.default_msec_format = "%s.%03d"

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


def format_markdown_blockquote(text: str) -> str:
    """Format text as a Markdown blockquote."""
    lines = text.split("\n")
    formatted_lines = []
    for summary_line in lines:
        formatted_lines.append(f"> {summary_line}".strip())
    formatted_text = "\n".join(formatted_lines)
    return formatted_text


def write_jsonl(rows: Iterable[Any], path: Path, **kwargs: Any) -> None:
    """Write data in JSON Lines format.

    Args:
        rows: List of JSON-serializable objects to write to file.
        path: Path to file.
        kwargs: Additional parameters to pass to `json.dumps()`.
    """
    # `separators` is used here to write JSON compactly
    lines = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), **kwargs) for row in rows)
    output = "\n".join(lines) + "\n"
    path.write_text(output, encoding="utf-8")
