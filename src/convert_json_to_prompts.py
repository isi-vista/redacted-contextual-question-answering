"""Convert JSON example files into prompt-formatted training data.

I used the following commands to prepare this dataset:

```bash
python -m src.convert_json_to_prompts
```
"""

import argparse
import itertools
import json
from pathlib import Path
import random
from typing import TypedDict

from src.paths import DATA_DIR, EXAMPLE_DIR
from src.utils import DEFAULT_RANDOM_SEED, NO_ANSWER_TEXT, format_markdown_blockquote, write_jsonl


class FormattedExample(TypedDict):
    """Structure to store different formats of an example."""

    title: str
    constraints: str
    question: str
    answer: str
    prompt: str
    full: str


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
        "--output-dir",
        default=DATA_DIR / "prompts",
        type=Path,
        help="Output directory.",
    )
    p.add_argument("--seed", default=DEFAULT_RANDOM_SEED, type=int, help="Random seed.")
    args = p.parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    seed: int = args.seed

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory {str(input_dir)!r} does not exist or is not a directory."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            "Output path must be a directory, but "
            f"{str(output_dir)!r} already exists and is not a directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random()
    rng.seed(seed)

    example_files = sorted(EXAMPLE_DIR.glob("*.json"))
    examples = [json.loads(file.read_text(encoding="utf-8")) for file in example_files]
    formatted_examples: dict[str, list[FormattedExample]] = {}
    for example in examples:
        title = example["title"]
        assert title not in formatted_examples, title
        formatted_examples[title] = []

        formatted_summary = format_markdown_blockquote(example["summary"])

        assert len(example["questions"]) == 5, title
        for question in example["questions"]:
            for answer in question["answers"]:
                constraints = answer["constraints"]
                if constraints:
                    constraint_lines = [
                        "The answer must obey the following constraint(s):",
                        "",
                        *(f"- {constraint}" for constraint in constraints),
                        "",
                    ]
                else:
                    constraint_lines = []

                prompt_lines = [
                    "Answer a question using the following story:",
                    "",
                    formatted_summary,
                    "",
                    *constraint_lines,
                    f"Answer the question to the best of your ability using a single sentence"
                    f"{' and without violating the provided constraint(s)' if constraints else ''}. "
                    f'If no answer is possible, answer "{NO_ANSWER_TEXT}" instead.',
                    "",
                    f"Question: {question['question']}",
                    "Answer:",
                ]

                prompt_text = "\n".join(prompt_lines)
                constraint_text = (
                    "\n".join(f"- {constraint}" for constraint in constraints)
                    if constraints
                    else ""
                )
                answer_text = answer["answer"] if answer["answer"] else NO_ANSWER_TEXT
                full_text = f"{prompt_text} {answer_text}"
                formatted_example: FormattedExample = {
                    "title": title,
                    "constraints": constraint_text,
                    "question": question["question"],
                    "answer": answer_text,
                    "prompt": prompt_text,
                    "full": full_text,
                }
                formatted_examples[title].append(formatted_example)

    output = "\n\n---\n\n".join(
        e["full"] for e in itertools.chain.from_iterable(formatted_examples.values())
    )
    output += "\n"
    (output_dir / "prompts.md").write_text(output, encoding="utf-8")

    shuffled_titles = rng.sample(list(formatted_examples), k=len(formatted_examples))
    train_titles = sorted(shuffled_titles[: len(shuffled_titles) // 2])
    test_titles = sorted(shuffled_titles[len(shuffled_titles) // 2 :])

    for split, titles in (("train", train_titles), ("test", test_titles)):
        split_examples = list(
            itertools.chain.from_iterable(formatted_examples[title] for title in titles)
        )
        write_jsonl(split_examples, output_dir / f"{split}.all.jsonl")

        shuffled_examples = rng.sample(split_examples, k=len(split_examples))
        text_only = [{"text": e["full"]} for e in shuffled_examples]
        write_jsonl(text_only, output_dir / f"{split}.text.jsonl")


if __name__ == "__main__":
    main()
