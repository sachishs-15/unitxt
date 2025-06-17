from unitxt import get_logger
from unitxt.api import create_dataset, evaluate, load_dataset
from unitxt.formats import HFSystemFormat
from unitxt.inference import CrossProviderInferenceEngine, HFPipelineBasedInferenceEngine, HFAutoModelInferenceEngine
from unitxt.processors import ExtractWithRegex, PostProcess
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate
import torch

logger = get_logger()
model_path = "/dccstor/indiclm/rudra/granite-magpie-translated-sft-ckps/nb-indic-run3-chkp900"  # Path to the model weights file

for thinking in [True, False]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(
        card="cards.reasoning.gsm8k.main",
        num_demos=8,
        demos_pool_size=9,
        demos_taken_from="train",
        max_test_instances=10,
        split="test",
        format=HFSystemFormat(
            model_name="ibm-granite/granite-3.3-8b-instruct",
            chat_kwargs_dict={"thinking": thinking},
            place_instruction_in_user_turns=True,
        ),
        loader_limit=100
    )   

    model_args_dict = {
        "batch_size": 4, 
        "model_kwargs": {
            "trust_remote_code": True,
            "device_map": "auto"
        }
    }
    
    # dataset.to(device)
    model = HFAutoModelInferenceEngine(
        model_name=model_path, **model_args_dict, max_new_tokens=8192
    )

    breakpoint()

    predictions = model(dataset)

    results = evaluate(predictions=predictions, data=dataset)
    
    print(results.global_scores)

    # with open(f"results/gsmk_thinking_{thinking}_fewshot_global_hf.txt", "w") as f:
    #     print(results.global_scores.summary, file=f)
    # with open(f"results/gsmk_thinking_{thinking}_fewshot_instance_hf.txt", "w") as f:
    #     print(results.instance_scores.summary, file=f)

    # print(results.instance_scores)

