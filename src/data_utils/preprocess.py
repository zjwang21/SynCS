from datasets import load_dataset, concatenate_datasets, load_from_disk
from itertools import chain
import  os
import torch
from .utils import translation_prompt
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def preprocess_enclone_text(tokenizer, data_args, training_args):
    #dataset = []
    #for file in os.listdir(data_args.dataset_name):
    #    dataset.append(load_dataset("parquet", data_files=os.path.join(data_args.dataset_name, file))['train'])
    dataset = load_dataset("parquet", data_files=data_args.dataset_name)['train']
    #dataset = dataset.select(range(50000))

    tokenizer_func = tokenizer
    def tokenize_data(examples):
        text_examples = [k + tokenizer.eos_token for k in examples["text"]]
        ids =  tokenizer_func(text_examples, add_special_tokens=False).input_ids
        return {"input_ids": ids}
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_dataset = dataset.map(tokenize_data, 
                                        num_proc=data_args.preprocessing_num_workers,
                                        remove_columns=dataset.column_names,
                                        batched=True)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // data_args.seq_length) * data_args.seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + data_args.seq_length] for i in range(0, total_length, data_args.seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    logger.info(tokenized_dataset)
    with training_args.main_process_first(desc="dataset groups"):
        tokenized_dataset = tokenized_dataset.map(group_texts, 
                                        num_proc=data_args.preprocessing_num_workers,
                                        batched=True)
        
    dataset = tokenized_dataset
    return dataset

def preprocess_pretrain_data(tokenizer , data_args, training_args):
    if data_args.cache_path is not None:
        cache_paths = data_args.cache_path.split(",")
        if len(cache_paths) > 1:
            tokenized_dataset = []
            max_tokens_per_split = data_args.max_tokens.split(",")
            assert len(cache_paths) == len(max_tokens_per_split)
            for max_tokens, cache_path in zip(max_tokens_per_split, cache_paths):
                dataset = load_from_disk(cache_path)
                seq_length = len(dataset[0]['input_ids'])
                total_samples = len(dataset)
                assert seq_length == data_args.seq_length, f"data length: {seq_length}; params: {data_args.seq_length}"
                max_samples = int(float(max_tokens) * 1e9 / seq_length) if max_tokens != "all" else total_samples
                logger.info(f"Loading {max_samples} tokenized samples ({max_tokens} B tokens) from split {cache_path} (total_samples: {total_samples}. seq_length: {seq_length})")
                tokenized_dataset.append(dataset.select(range(max_samples)))
            tokenized_dataset = concatenate_datasets(tokenized_dataset)
        else:
            tokenized_dataset = load_from_disk(data_args.cache_path)
            seq_length = len(tokenized_dataset[0]['input_ids'])
            total_samples = len(tokenized_dataset)
            assert seq_length == data_args.seq_length, f"data length: {seq_length}; params: {data_args.seq_length}"
            if data_args.max_tokens is not None and data_args.max_tokens != 'all':
                max_samples = float(max_tokens) * 1e9 / seq_length
                max_tokens = data_args.max_tokens
            else:
                max_samples = total_samples
                max_tokens = max_samples * seq_length / 1e9
            tokenized_dataset = tokenized_dataset.select(range(max_samples))
            logger.info(f"Loading {max_samples} tokenized samples ({max_tokens} B tokens) from split {data_args.cache_path} (total_samples: {total_samples}. seq_length: {seq_length})")
        
        tokenized_dataset = tokenized_dataset.shuffle(seed=training_args.seed)
        logger.info(tokenized_dataset)
        return tokenized_dataset
    
    if os.path.isdir(data_args.dataset_name):
        dataset = []
        filelist = [k for k in os.listdir(data_args.dataset_name) if k.endswith("jsonl")]
        if data_args.max_parquet_files is not None:
            filelist = filelist[:data_args.max_parquet_files]

        for file in filelist:
            dataset.append(load_dataset("json", data_files=os.path.join(data_args.dataset_name, file))['train'])
        dataset = concatenate_datasets(dataset)
        #dataset = dataset.select(range(10000))
    else:
        dataset = load_dataset("json", data_files=data_args.dataset_name)['train']
    #dataset = dataset.select(range(50000))

    def tokenize_data(examples):
        text_examples = [k + tokenizer.eos_token for k in examples["text"]]
        inputs =  tokenizer(examples['text'], add_special_tokens=False)
        return inputs
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_dataset = dataset.map(tokenize_data, 
                                        num_proc=data_args.preprocessing_num_workers,
                                        remove_columns=dataset.column_names,
                                        batched=True)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // data_args.seq_length) * data_args.seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + data_args.seq_length] for i in range(0, total_length, data_args.seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    with training_args.main_process_first(desc="dataset groups"):
        tokenized_dataset = tokenized_dataset.map(group_texts, 
                                        num_proc=data_args.preprocessing_num_workers,
                                        batched=True)
    logger.info(tokenized_dataset)
    dataset = tokenized_dataset
    return dataset


def preprocess_sft_data(tokenizer , data_args, training_args):
    dataset = load_dataset("json", data_files=data_args.dataset_name)['train']

    def tokenize_data(examples):
        instructions = examples['instruction']
        inputs = examples['input']
        labels = examples['output']
        prompt_ids =  tokenizer([a + b for a, b in zip(instructions, inputs)], add_special_tokens=False).input_ids
        label_ids = tokenizer(labels, add_special_tokens=False).input_ids
        return {"prompt_ids": prompt_ids, "label_ids": label_ids}
    
    with training_args.main_process_first(desc="tokenization"):
        tokenized_dataset = dataset.map(tokenize_data, 
                                        num_proc=data_args.preprocessing_num_workers,
                                        batched=True)
    return tokenized_dataset