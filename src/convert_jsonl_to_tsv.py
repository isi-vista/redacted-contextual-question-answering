"""Convert JSON Lines output files into TSV files for evaluation.

I used the following commands to prepare this dataset:

```bash
python -m src.convert_jsonl_to_tsv --input-file data/output/test_gpt3_5_turbo.jsonl --output-file data/output/test_gpt3_5_turbo.tsv
python -m src.convert_jsonl_to_tsv --input-file data/output/test_gpt4_turbo.jsonl --output-file data/output/test_gpt4_turbo.tsv
```
"""

import argparse
import csv
import json
from pathlib import Path

from src.utils import TsvDialect


def main() -> None:
    """Entrypoint for conversion script."""
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
        help="Output TSV file.",
    )
    args = p.parse_args()
    input_file: Path = args.input_file
    output_file: Path = args.output_file

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file {str(input_file)!r} does not exist or is not a file.")
    if output_file.exists() and not output_file.is_file():
        raise OSError(f"Output must be a writeable file path, but got {str(output_file)!r}.")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as file:
        input_rows = [json.loads(line) for line in file]

    output_rows = []
    for input_row in input_rows:
        output_row = {
            "Title": input_row["title"],
            "Constraints": input_row["constraints"],
            "Question": input_row["question"],
            "Gold answer": input_row["answer"],
            "Generated answer": input_row["generated_text"],
            "Correct?": "",
        }
        output_rows.append(output_row)

    fieldnames = list(output_rows[0])
    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, dialect=TsvDialect)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
