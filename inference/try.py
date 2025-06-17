from datasets import load_dataset
from evaluate import evaluator
from transformers import AutoModelForSequenceClassification, pipeline

data = load_dataset("imdb", split="test").shuffle(seed=42).select(range(1000))
task_evaluator = evaluator("text-classification")

print(data)
exit(0)
# 1. Pass a model name or path
eval_results = task_evaluator.compute(
    model_or_pipeline="lvwerra/distilbert-imdb",
    data=data,
    label_mapping={"NEGATIVE": 0, "POSITIVE": 1}
)

# 2. Pass an instantiated model
model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")

eval_results = task_evaluator.compute(
    model_or_pipeline=model,
    data=data,
    label_mapping={"NEGATIVE": 0, "POSITIVE": 1}
)

# 3. Pass an instantiated pipeline
pipe = pipeline("text-classification", model="lvwerra/distilbert-imdb")

eval_results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=data,
    label_mapping={"NEGATIVE": 0, "POSITIVE": 1}
)
print(eval_results)