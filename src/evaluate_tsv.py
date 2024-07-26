r"""Evaluate annotated TSVs and calculate model accuracies.

I used the following commands to evaluate the TSV files:

```bash
python -m src.evaluate_tsv --input-file data/annotated/test_gpt3_5_turbo.tsv
python -m src.evaluate_tsv --input-file data/annotated/test_gpt4_turbo.tsv
python -m src.evaluate_tsv --input-file data/annotated/test_falcon_untrained.tsv
python -m src.evaluate_tsv --input-file data/annotated/test_falcon_trained.tsv
python -m src.evaluate_tsv --input-file data/annotated/train_falcon_untrained.tsv
python -m src.evaluate_tsv --input-file data/annotated/train_falcon_trained.tsv
```

I used the following command to evaluate all files at once:

```bash
echo -n test_gpt3_5_turbo test_gpt4_turbo {test,train}_falcon_{un,}trained | xargs -d ' ' -I {} bash -c 'echo -e "\n=== {} ===" && python -m src.evaluate_tsv --input-file data/annotated/{}.tsv'
```
"""

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from pathlib import Path

from src.utils import CONSTRAINT_TO_LABEL, TsvDialect

YES = "y"
NO = "n"


def analyze_results(annotations: Sequence[Mapping[str, str]]) -> None:
    """Analyze the model accuracy."""
    scores: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "incorrect": 0, "invalid_answer": 0}
    )
    for annotation in annotations:
        correct = annotation["Correct?"]
        if correct == YES:
            key = "correct"
        elif correct == NO:
            key = "incorrect"
        else:
            print(f"Annotation was not {YES!r} or {NO!r}, got {correct!r}.")
            key = "invalid_answer"
        scores["all"][key] += 1
        scores[annotation["Constraints"]][key] += 1

    for subset_key in scores:
        subset = scores[subset_key]
        num_total = sum(subset.values())
        num_correct = subset["correct"]
        # The following code is messy but needed to avoid divide-by-0 errors
        if num_total == 0:
            accuracy_str = "n/a"
        else:
            accuracy_str = f"{num_correct / num_total:.1%}"
        subset_label = CONSTRAINT_TO_LABEL.get(subset_key, subset_key)
        print(f"Accuracy ({subset_label}): {accuracy_str} ({num_correct}/{num_total})")


def main() -> None:
    """Entrypoint for evaluation script."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input TSV file.",
    )
    args = p.parse_args()
    input_file: Path = args.input_file

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file {str(input_file)!r} does not exist or is not a file.")

    with open(input_file, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, dialect=TsvDialect)
        rows = list(reader)

    analyze_results(rows)


if __name__ == "__main__":
    main()
