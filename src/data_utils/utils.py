from collections import defaultdict
from tqdm import tqdm

translation_prompt =  "Translate from {} to {}: "

def load_codeswitch_table(data_args, tokenizer):
    dict_data = []
    with open(data_args.codeswitch_table) as f:
        for line in f:
            dict_data.append(line.strip())

    codeswitch_table = defaultdict(list)
    for d in tqdm(dict_data):
        words = d.split(" $ ")
        src = words[0]
        src_tokenized = tokenizer(" "+src)["input_ids"]
        if tokenizer.convert_ids_to_tokens(src_tokenized[0]) in tokenizer.special_tokens_map.values():
            src_tokenized = src_tokenized[1:]
        tgts = words[1:]
        for tgt in tgts:
            if tgt == src:
                continue
            tgt_tokenized = tokenizer(" "+tgt)["input_ids"]
            if tokenizer.convert_ids_to_tokens(tgt_tokenized[0]) in tokenizer.special_tokens_map.values():
                tgt_tokenized = tgt_tokenized[1:]
            codeswitch_table[tuple(src_tokenized)].append(tgt_tokenized)
    return codeswitch_table