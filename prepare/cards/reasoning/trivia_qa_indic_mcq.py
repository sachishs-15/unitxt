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
                path="sarvamai/trivia-qa-indic-mcq",
                data_dir=language,
                splits=["validation"],
            ),
            preprocess_steps=[
                Deduplicate(by=["question", "answer", "choices"]),
                RenameSplits({"validation": "test"}),
            ],
            task="tasks.qa.multiple_choice.open",
            templates=["templates.qa.multiple_choice.reasoning"],
            __tags__={
                "annotations_creators": "no-annotation",
                "arxiv": ["1705.03551"],
                "language": language,
                "language_creators": "expert-generated",
                "license": "unknown",
                "multilinguality": "multilingual",
                "region": "in",
                "size_categories": "100K<n<1M",
                "source_datasets": "original",
                "task_categories": "question-answering",
                "task_ids": "multiple-choice-qa",
            },
            __description__=(
                "A multilingual version of the TriviaQA dataset, translated into 10 Indian languages.\n"
                "TriviaqQA is a reading comprehension dataset containing over 650K question-answer-evidence triples.\n"
                "TriviaqQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions.\n"
                "More information can be found on the dataset page: https://huggingface.co/datasets/sarvamai/trivia-qa-indic-mcq\n"
                "More information about the original dataset can be found on the TriviaQA page: https://huggingface.co/datasets/mandarjoshi/trivia_qa\n"
            ),
        )

        if is_first:
            test_card(card, strict=False)
            is_first = False

        add_to_catalog(
            card,
            f"cards.reasoning.trivia_qa_indic_mcq.{language}",
            overwrite=True,
        )
