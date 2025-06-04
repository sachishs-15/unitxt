from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
         __description__="""This is a Math Reasoning Question Answering Task where the model is expected to provide a numerical answer (float) to a math-related question.
            The questions require multi-step reasoning and often involve arithmetic, algebraic, or logical problem-solving.
            Only the final answer (a float) is expected as output, without intermediate steps or explanations.
            This task uses accuracy as the default evaluation metric, comparing the predicted float to the ground truth.
            This task belongs to the 'tasks.qa.math' category in the Unitxt catalog.""",
        input_fields={
            "question": str,
        },
        reference_fields={"answer": str},
        prediction_type=str,
        metrics=["metrics.accuracy", "metrics.root_mean_squared_error"],
        augmentable_inputs=["question"],
        default_template="templates.qa.math.reasoning",
    ),
    "tasks.qa.math",
    overwrite=True,
)