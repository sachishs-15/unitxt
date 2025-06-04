import pandas as pd
from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate
from unitxt.processors import PostProcess, ExtractWithRegex
from unitxt.operators import Cast

add_to_catalog(
    InputOutputTemplate(
            instruction=(
                "You are a helpful and accurate math tutor. Solve the math problem following the instructions carefully:\n\n"
                "1. Think step-by-step to arrive at the correct answer.\n"
                "2. Enclose your entire reasoning within <think> ... </think> tags.\n"
                "3. After the reasoning, provide your final answer only as a single number with no units or text within <response> ... </response> tags.\n"
                "4. Do not include any words, symbols, or explanations inside the <response> tag — only the numeric answer.\n"
            ),
            input_format=(
                "Question: {question}\n\n"
            ),
            output_format="{answer}",
            postprocessors=[PostProcess(ExtractWithRegex(regex='<response>(.*?)</response>'), process_references=False), PostProcess(process_references=True, process_prediction=True, operator=Cast(to="float", failure_default=0.0,))],
    ), 
    "templates.qa.math.reasoning",
    overwrite=True,
)

