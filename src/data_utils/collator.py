import torch
import transformers
from typing import Sequence, Dict
import random
from dataclasses import dataclass

def sft_collator(tokenizer, examples):
    prompt_ids = [k['prompt_ids'] for k in examples]
    label_ids = [k['label_ids'] for k in examples]
    input_ids = [a + b + [tokenizer.eos_token_id] for a, b in zip(prompt_ids, label_ids)]
    labels = [[-100]*len(a) + b + [tokenizer.eos_token_id] for a, b in zip(prompt_ids, label_ids)]
    padded_inputs = tokenizer.pad({"input_ids": input_ids}, padding_side='right', return_tensors='pt')
    padded_labels = tokenizer.pad({"input_ids": labels}, padding_side='right', return_tensors='pt')
    return {"input_ids": padded_inputs['input_ids'], "attention_mask": padded_inputs['attention_mask'], "labels": padded_labels['input_ids']}