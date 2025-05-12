import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

from ..model.ea_model import EaModel
from ..model.kv_cache import initialize_past_key_values
from ..model.utils import *

model = EaModel.from_pretrained(
        base_model_path="/workspace/weights/Meta-Llama-3-8B-Instruct",
        ea_model_path="/workspace/weights/EAGLE-LLaMA3-Instruct-8B",
        torch_dtype=torch.float16,
        device_map="cuda"
).eval()
model.ea_layer.top_k = 1
model.hybrid_tree = True 

# prompt = "Who"
prompt = "Who decides the number of judges in the high court?"
tok = model.get_tokenizer()
inp = tok(prompt, return_tensors="pt").input_ids.to("cuda")

out_ids = model.eaself_generate(inp, max_new_tokens=10)
print(tok.decode(out_ids[0], skip_special_tokens=True))