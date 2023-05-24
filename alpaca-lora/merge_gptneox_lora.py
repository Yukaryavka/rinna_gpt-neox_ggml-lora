import os
import sys
import torch
import peft

from typing import Optional
from dataclasses import dataclass, field
from peft import PeftConfig, PeftModel
from peft.utils import _get_submodules
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

if len(sys.argv) < 3:
    print("\n")
    print("<< merge_gpt_neox_lora.py - Reference >>")
    print("")
    print("Usage: python merge_gptneox_lora.py base_model_name lora_model_name output_dir")
    print("-------------------------------------------------------------------------------")
    print("  base_model_name:  Define the base model name or dirpath or huggingFace repo to merge LoRA from / Example: 'rinna/japanese-gpt-neox-3.6b-instruction-sft'")
    print("  lora_model_name:  Define the dirpath of the LoRA model to be merged into the base model. [Basically, set the path to the directory where adapter_model.bin and adapter_config.json are stored.]")
    print("  output_dir:       Define a directory to store the pytorch model in which the base and LoRA models are merged")
    sys.exit(1)

base_model_name = sys.argv[1]
lora_model_name = sys.argv[2]
output_dir = sys.argv[3]

# make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load base model and tokenizer
print(">>> Loading base model: ", base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name, return_dict=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print(">>> Loading LoRA model: ", lora_model_name)
# Load PeftModel
model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            torch_dtype=torch.float16,
            device_map={'': 0},
        )
model.eval()

print("/// ")
print(">>> Start the process of merging the base model and the LoRA model.")
print("/// ")

# Merge: gpt-neox base model + LoRA model
key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
for key in key_list:
    parent, target, target_name = _get_submodules(model.base_model.model, key)
    if isinstance(target, peft.tuners.lora.Linear):
        bias = target.bias is not None
        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
        model.base_model._replace_module(parent, target_name, new_module, target)

model = model.base_model.model

print(">>> Saving merge model to: ", output_dir)
model.save_pretrained(output_dir)
print("--- DONE!")

