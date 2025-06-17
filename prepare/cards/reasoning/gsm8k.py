from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    Deduplicate,
    FilterByCondition,
    ListFieldValues,
    MapInstanceValues,
    Rename,
    Set,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card
from unitxt.operators import Apply
from unitxt.processors import PostProcess, ExtractWithRegex

subsets = [
    "main",
    "socratic"
]


is_first = True
for subset in subsets:
        card = TaskCard(
            loader=LoadHF(
                path="openai/gsm8k",
                splits=["train", "test"],
                data_dir=subset,
            ),
            preprocess_steps=[
                Deduplicate(by=["question", "answer"]),
            ],
            task="tasks.qa.math",
            templates=["templates.qa.math.reasoning.gsm8k.few_shot"],
            __tags__={
                "annotations_creators": "no-annotation",
                "arxiv": ["2110.14168"],
                "language": subset,
                "language_creators": "expert-generated",
                "license": "mit",
                "multilinguality": "monolingual",
                "region": "in",
                "size_categories": "10K<n<100K",
                "source_datasets": "original",
                "task_categories": "question-answering",
                "task_ids": "math-qa",
            },
            __description__=(
                "GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems.\n"
                "The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.\n"
                "More information about the original dataset can be found on the GSM8K page: https://huggingface.co/datasets/openai/gsm8k\n"
            ),
        )

        if is_first:
            test_card(card, strict=False)
            is_first = False

        add_to_catalog(
            card,
            f"cards.reasoning.gsm8k.{subset}",
            overwrite=True,
        )
  