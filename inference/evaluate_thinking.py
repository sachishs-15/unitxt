from unitxt import get_logger
from unitxt.api import create_dataset, evaluate, load_dataset
from unitxt.formats import HFSystemFormat
from unitxt.inference import CrossProviderInferenceEngine, HFPipelineBasedInferenceEngine
from unitxt.processors import ExtractWithRegex, PostProcess
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate

logger = get_logger()

for thinking in [True, False]:
    
    dataset = load_dataset(
        card="cards.reasoning.gsm8k.main",
        num_demos=8,
        demos_pool_size=9,
        demos_taken_from="train",
        split="test",
        format=HFSystemFormat(
            model_name="ibm-granite/granite-3.3-8b-instruct",
            chat_kwargs_dict={"thinking": thinking},
            place_instruction_in_user_turns=True,
        )
    )

    model = CrossProviderInferenceEngine(
        model="granite-3-3-8b-instruct", provider="rits", use_cache=False
    )

    predictions = model(dataset)

    results = evaluate(predictions=predictions, data=dataset)

    with open(f"results/gsmk_thinking_{thinking}_fewshot_global_rits.txt", "w") as f:
        print(results.global_scores.summary, file=f)
    with open(f"results/gsmk_thinking_{thinking}_fewshot_instance_rits.txt", "w") as f:
        print(results.instance_scores.summary, file=f)

    # print(results.instance_scores)