"""Convert JSON example files into a much more readable Markdown file.

I used the following commands to prepare this dataset:

```bash
python -m src.convert_json_to_md
```
"""

import argparse
import itertools
import json
from pathlib import Path

from src.paths import DATA_DIR, EXAMPLE_DIR
from src.utils import CONSTRAINT_TO_LABEL, NO_ANSWER_TEXT, format_markdown_blockquote


def main() -> None:
    """Entrypoint for conversion script."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        default=EXAMPLE_DIR,
        type=Path,
        help="Input directory containing JSON files.",
    )
    p.add_argument(
        "--output-file",
        default=DATA_DIR / "examples.md",
        type=Path,
        help="Output Markdown file.",
    )
    args = p.parse_args()
    input_dir: Path = args.input_dir
    output_file: Path = args.output_file

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory {str(input_dir)!r} does not exist or is not a directory."
        )
    if output_file.exists() and not output_file.is_file():
        raise OSError(f"Output must be a writeable file path, but got {str(output_file)!r}.")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    example_files = sorted(EXAMPLE_DIR.glob("*.json"))
    examples = [json.loads(file.read_text(encoding="utf-8")) for file in example_files]
    formatted_examples = []
    for example in examples:
        title = example["title"]
        source_url = example["source"]

        formatted_summary = format_markdown_blockquote(example["summary"])

        constraints = list(
            itertools.chain.from_iterable(
                answer["constraints"] for answer in example["questions"][0]["answers"]
            )
        )
        if constraints:
            constraint_lines = [
                "The answer must obey the following constraint(s):",
                "",
                *(f"- {constraint}" for constraint in constraints),
                "",
            ]
        else:
            constraint_lines = []

        question_lines = []
        for question in example["questions"]:
            question_lines.append(f"Question: {question['question']}")
            for answer in question["answers"]:
                answer_constraints = "\n".join(
                    f"- {constraint}" for constraint in answer["constraints"]
                )
                label_text = CONSTRAINT_TO_LABEL[answer_constraints]
                answer_text = answer["answer"] if answer["answer"] else NO_ANSWER_TEXT
                question_lines.append(f"Answer ({label_text}): {answer_text}")
            question_lines.append("")

        display_lines = (
            f"<!-- Summary of {title!r} -->",
            f"<!-- From {source_url} -->",
            "",
            "Answer a question using the following story:",
            "",
            formatted_summary,
            "",
            *constraint_lines,
            f"Answer the question to the best of your ability using a single sentence"
            f"{' and without violating the provided constraint(s)' if constraints else ''}. "
            f'If no answer is possible, answer "{NO_ANSWER_TEXT}" instead.',
            "",
            *question_lines,
        )
        display_text = "\n".join(display_lines).strip()
        formatted_examples.append(display_text)

    output = "\n\n---\n\n".join(formatted_examples) + "\n"
    output_file.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
