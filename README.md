# Redacted contextual question answering

## Publications

For more information about this project, see the related paper:

> TODO: Add citation once paper is published

## Installation

Use the provided Makefile to install this project by running the following from the project root directory (the same directory as this README). Ensure the `python` in `PATH` is 3.11 before running this command:

```shell
make install
```

DeepSpeed must be installed manually. See [Installation Details - DeepSpeed](https://www.deepspeed.ai/tutorials/advanced-install/) for instructions on how to do so.

Note that the installation command will attempt to download all used models from the Hugging Face Hub. To do this, you will need to create a Hugging Face account and request access on the pages for the following models:

- [Gemma-7b](https://huggingface.co/google/gemma-7b)
- [Gemma-7b-it](https://huggingface.co/google/gemma-7b-it)
- [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [Llama2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- [Mistral-7b-instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

Once your request has been approved, authenticate on your local machine using a user access token, using the official [User access tokens](https://huggingface.co/docs/hub/security-tokens) documentation as a guide.

If the installation process fails, is interrupted, or for any reason needs to be restarted, run `git clean -xdf` to reset the repository's state.

## Dataset

We have collected a dataset of 10 openly licensed summaries for movies and television episodes. These abstracts were found on Wikipedia with [List of American films of 2023 - Wikipedia](https://en.wikipedia.org/wiki/List_of_American_films_of_2023) and [Category:2023 works - Wikipedia](https://en.wikipedia.org/wiki/Category:2023_works) being used as the main way to find materials. We only used works published in July 2023 or later to avoid materials that might have been used to train SOTA LLMs.

Once we collected the summaries, we wrote 5 questions for each one. For each question, we then wrote 4 example answers, one for each for the 3 different constraints and one without any constraints. This resulted in 20 unique _(question, constraint, answer)_ tuples for each summary.

The dataset is stored in the following directories/files:

- [`rcqa_data/`](rcqa_data/): Directory of data files used in experiments. Most of these files are also in the paper's supplemental materials.
  - [`datasets/`](rcqa_data/datasets/): Directory of JSON Lines files containing the output of [`convert_json_to_prompts.py`](src/convert_json_to_prompts.py) using the files in [`summaries/`](summaries/) as input.
  - [`prompts/`](rcqa_data/prompts/): Directory of JSON Lines files used as input data for [`run_paper_experiments.sh`](scripts/run_paper_experiments.sh).
  - [`prompts.md`](rcqa_data/prompts.md): File containing the prompts in an easier-to-read Markdown format.
  - [`RedactedContextualQuestionAnsweringAnnotation.xlsx`](rcqa_data/RedactedContextualQuestionAnsweringAnnotation.xlsx): Model output with annotations of correctness, along with various relevant calculations and visualizations.
- [`summaries/`](summaries/): Directory of individual JSON files for each summary.
  - Each file contains a single object with the following fields:
    - `title`: Title of the television episode or movie.
    - `source`: Permalink to Wikipedia page version the summary was copied from.
    - `summary`: Markdown-formatted summary of episode or movie, copied from Wikipedia.
    - `questions`: Array of questions about each summary, with each question being an object with the following fields:
      - `question`: Question about the episode or movie that can be answered using the provided summary.
      - `answers`: Array of answers given specific constraints, with each answer being an object with the following fields:
        - `constraints`: Array of constraints to follow when answering the question.
        - `answer`: Example complete sentence that correctly answers the question and follows the constraints. If no answer is possible, then the value is `null` instead.

The summaries (and therefore the dataset) are licensed under the [Creative Commons 4.0 BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) (Attribution-ShareAlike) license.

## Redacted contextual question answering experiments

### Data

All data for the experiments can be found in [`rcqa_data/`](rcqa_data/). See the "[Dataset](#dataset)" section above for a complete description.

### Running

Run the following command to run training, inference, and evaluation for the paper:

```shell
bash scripts/run_paper_experiments.sh
```

You will likely need to make changes to the codebase to run in your specific environment.

## Contributing

This project uses various code quality tooling, all of which is automatically installed with the rest of the development requirements.

All checks can be run with `make check`, and some additional automatic changes can be run with `make fix`.

To test GitHub Actions workflows locally, install [`act`](https://github.com/nektos/act) and run it with `act`.

## License

- The dataset is under the [Creative Commons 4.0 BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) (Attribution-ShareAlike) license, which is used by the dataset (Wikipedia) that it is derived from.
- The code is under the [MIT](https://choosealicense.com/licenses/mit/) license.
- The paper ([`paper`](paper/)) is under the [Creative Commons 4.0 BY](https://creativecommons.org/licenses/by/4.0/) (Attribution) license, which is used for all publications in the ACL Anthology.
- [`src/run_clm.py`](src/run_clm.py) is originally under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license, with all changes from the original file being under the MIT license.
