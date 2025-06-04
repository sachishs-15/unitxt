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
from unitxt.operators import Copy

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
                path="sarvamai/arc-challenge-indic",
                data_dir=language,
                splits=["validation", "test"],
            ),
            preprocess_steps=[
                Deduplicate(by=["question", "answerKey", "id", "choices"]),
                RenameSplits({"validation": "train"}),
                Rename(field_to_field={"answerKey": "answer"}),
                MapInstanceValues(
                    mappers={
                        "answer": {
                            "A": 0,
                            "B": 1,
                            "C": 2,
                            "D": 3,
                        }
                    }
                ),
                Copy(field="choices/text", to_field="choices")
            ],
            task="tasks.qa.multiple_choice.open",
            templates=["templates.qa.multiple_choice.reasoning"],
            __tags__={
                "annotations_creators": "found",
                "arxiv": "1803.05457",
                "language": "en",
                "language_creators": "found",
                "license": "cc-by-sa-4.0",
                "multilinguality": "multilingual",
                "region": "in",
                "size_categories": "10K<n<100K",
                "source_datasets": "original",
                "task_categories": "question-answering",
                "task_ids": ["open-domain-qa", "multiple-choice-qa"],
            },
            __description__=(
                "ARC Challenge Indic is a multilingual version of the AI2 Reasoning Challenge (ARC) dataset, translated into 10 Indian languages.\n"
                "A new dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. We are also including a corpus of over 14 million science sentences.\n"
                "More information can be found on the dataset page: https://huggingface.co/datasets/sarvamai/arc-challenge-indic\n"
                "More information about the original dataset can be found on the ARC Challenge page: https://huggingface.co/datasets/allenai/ai2_arc"
            ),
        )

        if is_first:
            test_card(card, strict=False)
            is_first = False

        add_to_catalog(
            card,
            f"cards.reasoning.arc_challenge_indic.{language}",
            overwrite=True,
        )
  