import logging
from functools import partial
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from ..utils.logging_utils import get_logger
from .utils import smart_tokenizer_and_embedding_resize

logger = get_logger(__name__)

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
        config = AutoConfig.from_pretrained(model_args.model_path)
        model = AutoModelForCausalLM.from_config(config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        return model, tokenizer
    if model_args.exp in ['sft']:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, trust_remote_code=True)
        return model, tokenizer

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