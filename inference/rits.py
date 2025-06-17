from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.formats import HFSystemFormat
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.processors import ExtractWithRegex, PostProcess
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate
from unitxt import load_dataset, evaluate
import torch

logger = get_logger()
weights_path = "/dccstor/indiclm/rudra/granite-magpie-translated-sft-ckps"  # Path to the model weights file

for thinking in [True, False]:

    dataset = load_dataset(
        card="cards.reasoning.gsm8k.main",
        num_demos=8,
        demos_pool_size=10,
        max_test_instances=10,
        format=HFSystemFormat(
            model_name="ibm-granite/granite-3.3-8b-instruct",
            chat_kwargs_dict={"thinking": thinking},
            place_instruction_in_user_turns=True,
        ),
    )

    model = CrossProviderInferenceEngine(
        model="granite-3-3-8b-instruct", provider="rits", use_cache=False
    )

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    predictions = model(dataset)

    results = evaluate(predictions=predictions, data=dataset)

    print("Instance Results when Thinking=", thinking)
    print(results.instance_scores)