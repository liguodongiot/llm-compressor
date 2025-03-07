from llmcompressor.modifiers.quantization import GPTQModifier

from llmcompressor.args import (
    DatasetArguments,
    ModelArguments,
    RecipeArguments,
    TrainingArguments,
)

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    HfArgumentParser,
    PreTrainedModel,
    set_seed,
)
from transformers.utils.quantization_config import CompressedTensorsConfig

from common import max

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
dataset = "/xx"
         
def oneshot(**kwargs):
    parser = HfArgumentParser(
        (DatasetArguments, RecipeArguments)
    )

    parsed_args = parser.parse_dict(kwargs)
    dataset_args, recipe_args = parsed_args

    if recipe_args.recipe_args is not None:
        if not isinstance(recipe_args.recipe_args, dict):
            arg_dict = {}
            for recipe_arg in recipe_args.recipe_args:
                key, value = recipe_arg.split("=")
                arg_dict[key] = value
            recipe_args.recipe_args = arg_dict
    print(recipe_args)


print("-------")
print(max(4,5))

print("=======")


oneshot(dataset=dataset, recipe=recipe)


