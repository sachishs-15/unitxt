from unitxt.catalog import add_to_catalog
from unitxt.system_prompts import TextualSystemPrompt

system_prompt = TextualSystemPrompt(
    "You are Granite, developed by IBM. You are a helpful AI assistant."
)

add_to_catalog(system_prompt, "system_prompts.models.default.granite", overwrite=True)

