import os
import torch
import transformers

def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    new_tokenizer=None,
    vocab_size=None,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    if vocab_size is not None:
        new_vocab_size = len(tokenizer)
    elif new_tokenizer is not None:
        new_vocab_size = len(tokenizer) + len(new_tokenizer)
        vocab_size = len(tokenizer)
    else:
        new_vocab_size = 2 * len(tokenizer)
        vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=8, mean_resizing=False)
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:vocab_size].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:vocab_size].mean(dim=0, keepdim=True)

    input_embeddings[vocab_size:] = input_embeddings_avg
    output_embeddings[vocab_size:] = output_embeddings_avg
