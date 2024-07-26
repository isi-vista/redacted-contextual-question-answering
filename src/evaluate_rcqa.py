"""Evaluation script for LLMs with RC-QA data examples."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from tqdm import tqdm
from transformers import set_seed

from src.evaluation_utils import CUDA, DEVICE_AUTO, STR_TO_TORCH_DTYPE, build_transformers_pipeline
from src.paths import RCQA_EVAL_DIR, TRANSFORMERS_CACHE_DIR
from src.processing_utils import write_jsonl

set_seed(0)


@dataclass(frozen=True, slots=True)
class RcQaTask:
    """Dataclass for task results."""

    title: str
    constraints: str
    question: str
    answer: str
    prompt: str
    model_answer: str
    model_name: str

    @property
    def to_json_mapping(self) -> dict[str, str | dict[str, str]]:
        """Output Dataclass in JSONL format."""
        return {
            "title": self.title,
            "constraints": self.constraints,
            "question": self.question,
            "answer": self.answer,
            "prompt": self.prompt,
            "generated_text": self.model_answer,
            "generation_metadata": {"model_name": self.model_name},
        }


def main() -> None:
    """Runs a LLM model and performs inference evaluation."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--name", type=str, required=True, help="The name to use when saving this experiment."
    )
    parser.add_argument(
        "--eval-data", type=Path, required=True, help="Path to use to run evaluation over."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RCQA_EVAL_DIR,
        help="Directory to write output data to.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name or path to LLM model to load."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=TRANSFORMERS_CACHE_DIR,
        help="Directory to store files downloaded from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=sorted(STR_TO_TORCH_DTYPE),
        help="The torch datatype to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=CUDA,
        choices=[CUDA, DEVICE_AUTO],
        help=f"{DEVICE_AUTO} uses Hugging Face's accelerate framework.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="The maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--run-as-chat",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to run the model in chat mode. In chat mode, "
        "we pass the chat version of each input as-is to the pipeline.",
    )
    args = parser.parse_args()

    name: str = args.name
    eval_data: Path = args.eval_data
    output_dir: Path = args.output_dir
    model: str = args.model
    dtype_str: str = args.dtype
    device: str = args.device
    cache_dir: Path = args.cache_dir
    max_new_tokens: str = args.max_new_tokens
    run_as_chat: bool = args.run_as_chat

    if not eval_data.is_file():
        raise FileNotFoundError(f"The evaluation data file {str(eval_data)!r} does not exist.")
    output_dir.mkdir(exist_ok=True, parents=True)

    params = {
        "max_new_tokens": max_new_tokens,
        "min_length": 1,
        "do_sample": False,
        "num_return_sequences": 1,
        "temperature": None,  # Unset because `do_sample=False`
        "top_p": None,  # Unset because `do_sample=False`
    }

    tokenizer, pipeline = build_transformers_pipeline(model, dtype_str, device, cache_dir)

    annotations = []
    with eval_data.open(encoding="utf-8") as eval_file:
        for line in tqdm(eval_file):
            line_data = json.loads(line)
            prompt: str = line_data["prompt"]
            cleaned_response: str
            if run_as_chat:
                # jac: we don't attempt system prompting here. I learned while experimenting that
                # Mistral and Gemma do not support the system role -- only user and assistant
                history = [{"role": "user", "content": prompt}]
                sequences = pipeline(
                    history,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    **params,
                )
                cleaned_response = sequences[0]["generated_text"][-1]["content"].strip()
            else:
                sequences = pipeline(
                    prompt,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    **params,
                )
                cleaned_response = sequences[0]["generated_text"].removeprefix(prompt).strip()
            annotations.append(
                RcQaTask(
                    title=line_data["title"],
                    constraints=line_data["constraints"],
                    question=line_data["question"],
                    answer=line_data["answer"],
                    prompt=line_data["prompt"],
                    model_answer=cleaned_response,
                    model_name=model,
                )
            )

    write_jsonl([task.to_json_mapping for task in annotations], output_dir / f"{name}.jsonl")


if __name__ == "__main__":
    main()
