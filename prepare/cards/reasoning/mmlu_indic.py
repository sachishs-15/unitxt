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
                path="sarvamai/mmlu-indic",
                data_dir=language,
                splits=["validation", "test"],
            ),
            preprocess_steps=[
                Deduplicate(by=["question", "answer", "choices"]),
                RenameSplits({"validation": "train"}),
            ],
            task="tasks.qa.multiple_choice.open",
            templates=["templates.qa.multiple_choice.reasoning"],
            __tags__={
                "annotations_creators": "no-annotation",
                "arxiv": ["2009.03300", "2005.00700", "2005.14165", "2008.02275"],
                "language": language,
                "language_creators": "expert-generated",
                "license": "mit",
                "multilinguality": "multilingual",
                "region": "in",
                "size_categories": "100K - 1M",
                "source_datasets": "original",
                "task_categories": "question-answering",
                "task_ids": "multiple-choice-qa",
            },
            __description__=(
                "A multilingual version of the Massive Multitask Language Understanding (MMLU) benchmark, translated from English into 10 Indian languages. This version contains the translations of the development and test sets only.\n"
                "More information can be found on the dataset page: https://huggingface.co/datasets/sarvamai/mmlu-indic"
                "More information about the original MMLU benchmark can be found here: https://huggingface.co/datasets/cais/mmlu"
            ),  
        )

        if is_first:
            test_card(card, strict=False)
            is_first = False

        add_to_catalog(
            card,
            f"cards.reasoning.mmlu_indic.{language}",
            overwrite=True,
        )
