import os
import random
import re
import argparse
from functools import partial
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import json

# 将dic定义为全局变量
dic = {}


def split_document(document):
    # 使用正则表达式分割句子，并保留分隔符
    sentences = re.split(r'(\s*[\.\!\?。！？]\s*)', document)
    # 将句子和分隔符分开存储
    sentence_list = sentences[::2]
    separators = sentences[1::2] + ['']
    return sentence_list, separators
    
cstypes = ['annotation', 'replace', 'sentlevel']

def replace_sentences(sentences, replace, preserve_src=False):
    replaced_sents = 0
    for k, v in replace.items():
        if sentences[k] != v['src']:
            print(f"Error: src not equal!!!")
        if preserve_src:
            sentences[k] += " ({}) ".format(v['tgt'])
        else:
            sentences[k] = v['tgt']
        replaced_sents += 1
    return sentences, replaced_sents

def replace(file_name, flag, preserve_src, example, indice):
    key = "text_en" if "text_en" in example else "text"
    text = example[key]
    id = indice if flag else example['meta_data']['id']
    temp = dic[file_name]
    if id not in temp:
        replaced_text = text
        num_sents_replaced = 0
    else:
        sentences, separators = split_document(text)
        res, num_sents_replaced = replace_sentences(sentences, temp[id], preserve_src=preserve_src)
        replaced_text = ''.join(sentence + separator for sentence, separator in zip(res, separators))
    
    return {
        "id": id,
        "text": replaced_text,
        "n_replaced": num_sents_replaced
    }

def get_dic(path, cur_filename):
    global dic  # 使用全局变量
    dic = {}
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f):
            try:
                k = json.loads(line)
            except json.JSONDecodeError as e:
                print("JSON 解码错误:", e)
                print(line)
                continue
            filename = k['filename']
            if filename != cur_filename:
                continue
            if filename not in dic: dic[filename] = {}
            id = k['id']
            if id not in dic[filename]: dic[filename][id] = {}
            dic[filename][id][k['sent-idx']] = {"src": k['text'], "tgt": k['output'][0] if 'output' in k else k['model_output'][0]}

def main(data_path, replace_path, output_path, args):
    global dic  # 使用全局变量
    if data_path.endswith("jsonl"):
        files = [data_path]
    else:
        files = [os.path.join(data_path, file) for file in os.listdir(data_path)]

    for file in files:
        flag = False
        if not file.endswith("jsonl"):
            print(f"Jump {file}")
            continue
        
        if output_path.endswith("jsonl"):
            filename = file.split("/")[-1].split(".")[0]
            flag = True
        else:
            filename = file.split("/")[-1].split(".")[0]
            chunkid = int(filename[-2:])
            if chunkid not in list(range(44, 47)): 
                print(f"Jump {file}")
                continue

        print(f"Processing {file}")
        get_dic(replace_path, filename)  # 初始化全局变量
        data = load_dataset("json", data_files=file)['train']
        if len(dic) == 0:
            print(f"{file} do not have replaced sents, save and skipping......")
            if output_path.endswith("jsonl"):
                print(f"Saving to {output_path}")
                data.to_json(output_path)
            else:
                print(f"Saving to {os.path.join(output_path, filename)}.jsonl")
                data.to_json(os.path.join(output_path, filename + ".jsonl"))
        else:
            processed_data = data.map(partial(replace, filename, flag, args.preserve_src), num_proc=64, 
                                        batched=False, remove_columns=data.column_names, desc=f"Processing {filename}", with_indices=True)
            
            # 统计每类句子的数量
            n_replaced = sum(processed_data['n_replaced'])
            nsents_in_dic = sum([len(v) for k,v in dic[filename].items()])
            print(f"{file} finally get {n_replaced}/{nsents_in_dic} sentences replaced.")

            if output_path.endswith("jsonl"):
                print(f"Saving to {output_path}")
                processed_data.to_json(output_path)
            else:
                print(f"Saving to {os.path.join(output_path, filename)}.jsonl")
                processed_data.to_json(os.path.join(output_path, filename + ".jsonl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and replace sentences in JSONL files.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing JSONL files.')
    parser.add_argument('--replace_path', type=str, required=True, help='Fraction of sentences to replace.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed output JSON.')
    parser.add_argument('--preserve_src', type=bool, required=False, default=False, help='Path to save the processed output JSON.')

    args = parser.parse_args()
    if args.preserve_src:
        print("Preserving src sentence.")
    main(args.data_path, args.replace_path, args.output_path, args)
