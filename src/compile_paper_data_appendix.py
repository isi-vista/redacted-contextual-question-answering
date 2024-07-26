"""A script to compile the questions from the paper's raw JSONL data into suitable form for the paper's appendix."""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict[str, str]]:
    """Load the given JSONL file as a list of dicts."""
    with path.open(mode="r", encoding="utf-8") as jsonl_in:
        result = [json.loads(line.strip()) for line in jsonl_in if line.strip()]
    return result


def extract_synopsis(prompt: str) -> str:
    """Extract the synopsis part of the given prompt."""
    synopsis_lines = []
    in_synopsis = False
    for line in prompt.splitlines():
        if in_synopsis:
            if line.startswith(">"):
                synopsis_lines.append(line.removeprefix(">").strip())
            else:
                break
        else:
            in_synopsis = line.startswith(">")
    return "\n".join(synopsis_lines)


def recursive_sort_by_key(dict_: dict[str, Any]) -> dict[str, Any]:
    """Key function for sorting a dictionary on its keys."""
    return {k: recursive_sort_by_key(v) if isinstance(v, dict) else v for k, v in dict_.items()}


def main() -> None:
    """Compiles raw JSON Lines data into a suitable form for the paper."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "train_jsonl", type=Path, help="The path to the JSONL file of the paper's training data."
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="Logging level to use.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.logging_level),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    train_jsonl_path = args.train_jsonl

    if not train_jsonl_path.is_file():
        raise FileNotFoundError(
            f"The training data file {str(train_jsonl_path)!r} does not exist/isn't a file."
        )

    data = load_jsonl(train_jsonl_path)

    title_to_synopsis: dict[str, str] = {}
    title_to_question_to_constrained_answers: dict[str, dict[str, dict[str, str]]] = {}
    for datum in data:
        synopsis = extract_synopsis(datum["prompt"])
        if title_to_synopsis.get(datum["title"], synopsis) != synopsis:
            print(f"Conflicting synopses for title {datum['title']}. Taking the latter...")
        title_to_synopsis[datum["title"]] = synopsis

        question_to_constrained_answers = title_to_question_to_constrained_answers.setdefault(
            datum["title"], {}
        )
        constrained_answers = question_to_constrained_answers.setdefault(datum["question"], {})
        assert datum["constraints"] not in constrained_answers
        constrained_answers[datum["constraints"]] = datum["answer"]

    title_to_question_to_constrained_answers = recursive_sort_by_key(
        title_to_question_to_constrained_answers
    )

    # Print synopses first
    print("\\subsection{Synopses}")
    for title, synopsis in title_to_synopsis.items():
        question_to_constrained_answers = title_to_question_to_constrained_answers[title]
        print(f"\\subsubsection{{{title}}}")
        print(synopsis)
        print("\\subsubsubsection{Questions}")
        for question, constrained_answers in question_to_constrained_answers.items():
            print(question)
            print("\\begin{tabular}{|p{3cm}|p{3cm}|}")
            print("\\hline")
            print("\\textbf{Constraints} & \\textbf{Answer} \\\\")
            print("\\hline")
            for constraints, answer in constrained_answers.items():
                constraints_gloss = constraints if constraints.strip() else "(none)"
                print(f"{constraints_gloss} & {answer} \\\\")
                print("\\hline")
            print("\\end{tabular}")


if __name__ == "__main__":
    main()
