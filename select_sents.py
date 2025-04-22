import os
import random
import re
import argparse
from functools import partial
from collections import Counter
from datasets import load_dataset, concatenate_datasets

def split_document(document):
    # 使用正则表达式分割句子，并保留分隔符
    sentences = re.split(r'(\s*[\.\!\?。！？;；\n]+\s*)', document)
    # 将句子和分隔符分开存储
    sentence_list = sentences[::2]
    separators = sentences[1::2] + ['']
    return sentence_list, separators

cstypes = ['s-annt', 's-repl', 't-annt', 't-repl']

def replace_sentences(sentences, replace_fraction=0.05, min_length=3):
    res = []
    # 过滤掉空字符串和过短的句子
    valid_indices = [i for i, sentence in enumerate(sentences) if len(sentence.strip()) >= min_length]
    
    # 计算需要替换的句子数量
    num_sentences_to_replace = int(len(valid_indices) * replace_fraction)
    
    # 随机选择需要替换的句子索引
    indices_to_replace = random.sample(valid_indices, num_sentences_to_replace)
    
    # 替换选定的句子
    for index in indices_to_replace:
        res.append({"sent-idx": index, "text": sentences[index], "cstype": random.choice(cstypes)})
    return res

def select(file_name, replace_fraction, examples, doc_ids):
    ids = []
    filenames = []
    sent_idxs = []
    texts = []
    cstypes_list = []
    
    key = "text_en" if "text_en" in examples else "text"
    if 'meta_data' in examples:
        ori_ids = [k['id'] for k in examples['meta_data']]
    else:
        ori_ids = doc_ids
    for text, idx in zip(examples[key], ori_ids):
        sentences, separators = split_document(text)
        res = replace_sentences(sentences, replace_fraction=replace_fraction)
        for k in res:
            ids.append(idx)
            filenames.append(file_name)
            sent_idxs.append(k['sent-idx'])
            texts.append(k['text'])
            cstypes_list.append(k['cstype'])
    
    return {
        "id": ids,
        "filename": filenames,
        "sent-idx": sent_idxs,
        "text": texts,
        "cstype": cstypes_list
    }

def main(data_path, replace_fraction, output_path):
    final = []
    if data_path.endswith("jsonl") or data_path.endswith("json"):
        files = [data_path]
    else:
        files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    for file in files:
        if not file.endswith("jsonl") and not data_path.endswith("json"):
            print(f"Jump {file}")
            continue
        print(f"Processing {file}")
        data = load_dataset("json", data_files=file)['train']

        processed_data = data.map(partial(select, file.split("/")[-1].split(".")[0], replace_fraction), num_proc=96, with_indices=True,
                                    batched=True, remove_columns=data.column_names, desc=f"Processing {file}")
        
        # 统计每类句子的数量
        cstype_counter = Counter(processed_data['cstype'])
        print(f"{file} finally get {len(processed_data)} sentences. Breakdown by cstype: {dict(cstype_counter)}")

        final.append(processed_data)

    final = concatenate_datasets(final)
    final.to_json(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and replace sentences in JSONL files.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing JSONL files.')
    parser.add_argument('--replace_fraction', type=float, default=0.05, help='Fraction of sentences to replace.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed output JSON.')

    args = parser.parse_args()

    main(args.data_path, args.replace_fraction, args.output_path)
