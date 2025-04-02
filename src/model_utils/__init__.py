import logging
from functools import partial
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from ..utils.logging_utils import get_logger
from .utils import smart_tokenizer_and_embedding_resize

logger = get_logger(__name__)

def zero_grad_hook(k, grad):
    grad[:k, :] = 0  # 将前1000个词向量的梯度置为0
    return grad

def get_tokenizer(model_args):
    if model_args.tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_model_and_tokenizer(model_args, training_args):
    tokenizer = get_tokenizer(model_args)
    if model_args.train_from_scatch:
        #tokenizer = get_tokenizer(model_args)
        config = AutoConfig.from_pretrained(model_args.model_path)
        model = AutoModelForCausalLM.from_config(config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        return model, tokenizer
    if model_args.exp in ['ct-noexp', 'sft']:
        #tokenizer = get_tokenizer(model_args)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, trust_remote_code=True)
        return model, tokenizer
    if model_args.exp in ['sft-lora']:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, trust_remote_code=True)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules="all-linear"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model, tokenizer
    if model_args.exp in ['ct', 'moe-ct', 'moea-ct', 'only-embed']:
        #tokenizer = get_tokenizer(model_args)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, trust_remote_code=True)
        smart_tokenizer_and_embedding_resize(tokenizer, model, vocab_size=model_args.src_vocab_size)
        freeze_and_hook(model_args, model)
        return model, tokenizer
    if model_args.exp in ['enclone', 'enclone-full', 'enclone-moea']:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, trust_remote_code=True)
        smart_tokenizer_and_embedding_resize(tokenizer, model)
        freeze_and_hook(model_args, model)
        return model, tokenizer

def freeze_and_hook(model_args, model):
    if model_args.freeze_en_embeddings:
        hook_func = partial(zero_grad_hook, model_args.src_vocab_size)
        embedding_layer = model.get_input_embeddings()
        embedding_layer.weight.register_hook(hook_func)
        lm_head =  model.get_output_embeddings()
        lm_head.weight.register_hook(hook_func)
        logger.info(f"Freezing the Src tokenizer embeddings: index < {model_args.src_vocab_size}")

    if model_args.exp in ['moe-ct', 'moea-ct', 'enclone-moea']:
        for n, p in model.named_parameters():
            if "experts.1" in n or 'embed_tokens' in n or 'lm_head' in n or 'router' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        logger.info("Mark only new expert trainable......")
    if model_args.exp in ['enclone', 'only-embed']:
        for n, p in model.named_parameters():
            if 'embed_token' in n or 'lm_head' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        logger.info("Mark only embeddings trainable......")

    logger.info(model)

def count_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param