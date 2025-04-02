import torch
import transformers
from typing import Sequence, Dict
import random
from dataclasses import dataclass

def enclone_collator(tokenizer, model_args, examples):
    ids = [k['input_ids'] for k in examples]
    padded_inputs = tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding=True)
    bsz, seqlen =  padded_inputs['input_ids'].size()
    cipher_mask = padded_inputs['attention_mask'].masked_fill((padded_inputs['input_ids'] == tokenizer.eos_token_id), 0)

    if model_args.codeswitch_ratio != None:
        preserve_mask = torch.rand(bsz, seqlen) < model_args.codeswitch_ratio
        cipher_mask = cipher_mask.masked_fill(preserve_mask.to(cipher_mask.device), 0)

    if model_args.translate_ratio is not None:
        clone_ids = padded_inputs['input_ids'].clone()
        clone_ids[cipher_mask.bool()] += len(tokenizer)
        trans_mask = torch.rand(bsz) < model_args.translate_ratio
        en_batch_ids = torch.where(trans_mask)[0]
        en_batch_ids = en_batch_ids.tolist()
        enids = padded_inputs['input_ids'][en_batch_ids, seqlen//2:]
        clone_ids[en_batch_ids, :seqlen//2] = enids
        padded_inputs['input_ids'] = clone_ids
    else:
        padded_inputs['input_ids'][cipher_mask.bool()] += len(tokenizer)

    if model_args.replay_ratio is not None:
        replay_inputs = tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding=True)
        if model_args.replay_cs_ratio is not None:
            cipher_mask = torch.zeros_like(replay_inputs['attention_mask'])
            preserve_mask = torch.rand(bsz, seqlen) < model_args.replay_cs_ratio
            cipher_mask = cipher_mask.masked_fill(preserve_mask.to(cipher_mask.device), 1)
            cipher_mask = cipher_mask.masked_fill((replay_inputs['input_ids'] == tokenizer.eos_token_id), 0)
            cipher_mask = cipher_mask.masked_fill((replay_inputs['attention_mask'] == 0), 0)
            replay_inputs['input_ids'][cipher_mask.bool()] += len(tokenizer)
        padded_inputs['input_ids'] = torch.cat([padded_inputs['input_ids'], replay_inputs['input_ids']])
        padded_inputs['attention_mask'] = torch.cat([padded_inputs['attention_mask'], replay_inputs['attention_mask']])

    padded_inputs['labels'] = padded_inputs['input_ids'].clone()
    return padded_inputs

def lang_mask_collator(tokenizer, model_args, examples):
    input_ids = [k['input_ids'] for k in examples]
    inputs = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors='pt')
    lang_mask = torch.ones_like(inputs['input_ids'])
    lang_mask = lang_mask.masked_fill(inputs['input_ids'] < model_args.src_vocab_size, 0)
    inputs['lang_mask'] = lang_mask
    inputs['labels'] = inputs['input_ids'].clone()
    return inputs

def sft_collator(tokenizer, examples):
    prompt_ids = [k['prompt_ids'] for k in examples]
    label_ids = [k['label_ids'] for k in examples]
    input_ids = [a + b + [tokenizer.eos_token_id] for a, b in zip(prompt_ids, label_ids)]
    labels = [[-100]*len(a) + b + [tokenizer.eos_token_id] for a, b in zip(prompt_ids, label_ids)]
    padded_inputs = tokenizer.pad({"input_ids": input_ids}, padding_side='right', return_tensors='pt')
    padded_labels = tokenizer.pad({"input_ids": labels}, padding_side='right', return_tensors='pt')
    return {"input_ids": padded_inputs['input_ids'], "attention_mask": padded_inputs['attention_mask'], "labels": padded_labels['input_ids']}


@dataclass
class CodeSwitchCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    codeswitch_ratio: float
    codeswitch_table: Dict[tuple,tuple]
    def is_start(self,i):
        if i >= self.tokenizer:
            i -= len(self.tokenizer)
        return self.tokenizer.convert_ids_to_tokens(i).startswith("Ġ") or self.tokenizer.convert_ids_to_tokens(i).startswith("▁") or i == 0

    def codeswitch(self, input_ids):
        codeswitched_ids = []
        codeswitched_align_mask = []
        original_align_mask = []
        cur = [input_ids[0]]
        for i in input_ids[1:] + [0]:
            if self.tokenizer.convert_ids_to_tokens(i).startswith("▁") or self.tokenizer.convert_ids_to_tokens(i).startswith("Ġ") or i == 0:
                original_word_seq = cur
                if random.random() < self.codeswitch_ratio and tuple(original_word_seq) in self.codeswitch_table:
                    codeswitch_word_seq = random.choice(self.codeswitch_table[tuple(original_word_seq)])
                    codeswitched_ids.extend(codeswitch_word_seq)
                else:
                    codeswitched_ids.extend(original_word_seq)
                    codeswitched_align_mask.extend([1]*len(original_word_seq))
                    original_align_mask.extend([1]*(len(original_word_seq)))
                cur = [i]
            else:
                cur.append(i)
        return {
            "codeswitched_ids": codeswitched_ids,
            "original_ids": input_ids
        }


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dict = {}
        codeswitched =[self.codeswitch(instance["input_ids"]) for instance in instances]
        codeswitched_ids = [torch.tensor(codeswitch["codeswitched_ids"]).long() for codeswitch in codeswitched]

        inputs = self.tokenizer.pad({"input_ids": codeswitched_ids}, return_tensors='pt', padding_side='right')

        return_dict = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "labels": inputs['input_ids'].masked_fill(inputs['input_ids'].eq(self.tokenizer.pad_token_id),-100)
        }
        return return_dict