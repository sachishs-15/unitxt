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

languages = [
    "bn",
    "en",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te"
]


is_first = True
for language in languages:
        card = TaskCard(
            loader=LoadHF(
                path="sarvamai/gsm8k-indic",
                data_dir=language,
                splits=["test"],
            ),
            preprocess_steps=[
                Deduplicate(by=["question", "answer", "original_question"]),
                ExtractWithRegex(regex='####\s*(.*)', field="answer", to_field="answer"),
            ],
            task="tasks.qa.math",
            templates=["templates.qa.math.reasoning"],
            __tags__={
                "annotations_creators": "no-annotation",
                "arxiv": ["2110.14168"],
                "language": language,
                "language_creators": "expert-generated",
                "license": "mit",
                "multilinguality": "multilingual",
                "region": "in",
                "size_categories": "10K<n<100K",
                "source_datasets": "original",
                "task_categories": "question-answering",
                "task_ids": "math-qa",
            },
            __description__=(
                "Translated version of the GSM8K dataset into 10 Indian languages.\n"
                "GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems.\n"
                "The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.\n"
                "More information can be found on the dataset page: https://huggingface.co/datasets/sarvamai/gsm8k-indic\n"
                "More information about the original dataset can be found on the GSM8K page: https://huggingface.co/datasets/openai/gsm8k\n"
            ),
        )

        if is_first:
            test_card(card, strict=False)
            is_first = False

        add_to_catalog(
            card,
            f"cards.reasoning.gsmk8k_indic.{language}",
            overwrite=True,
        )
  