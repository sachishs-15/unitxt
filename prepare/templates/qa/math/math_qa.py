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

add_to_catalog(
    InputOutputTemplate(
            input_format=(
                "Here is a mathematical question. Give the final numerical answer as a single number with no units or text within <answer> ... </answer> tags.\n"
                "Question: {question}\n\n"
            ),
            output_format="{answer}",
            postprocessors=[PostProcess(ExtractWithRegex(regex='<answer>(.*?)</answer>'), process_references=False), PostProcess(process_references=True, process_prediction=True, operator=Cast(to="float", failure_default=0.0,))],
    ), 
    "templates.qa.math.enclosed_answer",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
            instruction=(
                "You are a helpful and accurate math tutor. Solve the math problem following the instructions carefully:\n\n"
                "1. Think step-by-step to arrive at the correct answer.\n"
                "2. Enclose your entire reasoning within <think> ... </think> tags.\n"
                "3. After the reasoning, provide your answer within <response> ... </response> tags.\n"
            ),
            input_format=(
                "Here is a mathematical question. Give the final numerical answer as a single number with no units or text within <answer> ... </answer> tags.\n"
                "Question: {question}\n\n"
            ),
            output_format="{answer}",
            postprocessors=[PostProcess(ExtractWithRegex(regex='<answer>(.*?)</answer>'), process_references=False), PostProcess(process_references=True, process_prediction=True, operator=Cast(to="float", failure_default=0.0,))],
    ), 
    "templates.qa.math.reasoning_with_enclosed_answer",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
            instruction=(
                "You are a helpful and accurate math tutor. Solve the math problem following the instructions carefully:\n\n"
                "1. Think step-by-step to arrive at the correct answer.\n"
                "2. Enclose your entire reasoning within <think> ... </think> tags.\n"
                "3. After the reasoning, provide your answer within <response> ... </response> tags.\n"
            ),
            input_format=(
                "Here is a mathematical question for you to solve\n"
                "Question: {question}\n\n"
            ),
            output_format="{answer}",
            postprocessors=["processors.extract_last_number", PostProcess(process_references=True, process_prediction=True, operator=Cast(to="float", failure_default=0.0,))],
    ), 
    "templates.qa.math.reasoning_with_simple_answer",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
            input_format=(
                "Q: {question}\nA: Let's think step by step.\n",
            ),
            output_format="{answer}",
            postprocessors=["processors.extract_last_number", PostProcess(process_references=True, process_prediction=True, operator=Cast(to="float", failure_default=0.0,))],
    ), 
    "templates.qa.math.reasoning.zero_shot",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
            input_format=(
                "Q: {question}\n"
                "A:"
            ),
            output_format="{answer}",
            postprocessors=["processors.extract_last_number", PostProcess(process_references=True, process_prediction=True, operator=Cast(to="float", failure_default=0.0,))],
    ), 
    "templates.qa.math.reasoning.few_shot",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
            input_format=(
                "Q: {question}\n"
                "A:"
            ),
            output_format="{answer}",
            postprocessors=[PostProcess(ExtractWithRegex(regex='####\s*(.*)'),  process_prediction=False), "processors.extract_last_number", PostProcess(process_references=True, process_prediction=True, operator=Cast(to="float", failure_default=0.0,))],
    ), 
    "templates.qa.math.reasoning.gsm8k.few_shot",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
            input_format=(
                "Question: {question}\nAnswer:\n"
            ),
            output_format="{answer}",
            postprocessors=["processors.extract_last_number", PostProcess(process_references=True, process_prediction=True, operator=Cast(to="float", failure_default=0.0,))],
    ), 
    "templates.qa.math.simple",
    overwrite=True,
)
