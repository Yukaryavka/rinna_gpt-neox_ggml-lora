from dataclasses import dataclass, field
from typing import Optional

import peft
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from peft.utils import _get_submodules

# Base model load
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", return_dict=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft")

model = PeftModel.from_pretrained(
            model,
            "./rinna-3.6b-inst-lora",
            torch_dtype=torch.float16,
            device_map={'': 0},
        )
model.eval()

# Merge: Base model + LoRA
key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
for key in key_list:
    parent, target, target_name = _get_submodules(model.base_model.model, key)
    if isinstance(target, peft.tuners.lora.Linear):
        bias = target.bias is not None
        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
        model.base_model._replace_module(parent, target_name, new_module, target)

model = model.base_model.model

model.save_pretrained("./Merged-rinna-3.6b-inst-lora")