# Investigating and Scaling up Code-Switching for Multilingual Language Model Pre-Training
This repository contains the code for our Paper: 

[Investigating and Scaling up Code-Switching for Multilingual Language Model Pre-Training](https://arxiv.org/abs/2504.01801)

# Code-Switching Data Synthesis

## Overview

This project synthesizes code-switching data for training purposes.

## Usage

1. **Training Code-Switching Synthesis Model**:
   Run the following command to execute the training script:
   ```bash
   bash scripts/sft.sh
   ```
   You can find our sft data for Chinese, Romanian, and Bengali in the directory: data.
   We use Qwen2.5-3B-Instruct as the base model.
2. **Split Documents**:
   We then use the trained model to synthesize code-switching data base on the pretraining documents.

   First split the documents to sentences. Prepare your pretraining documents as jsonl format with the key being "text".
   Run the following command:
   ```bash
   python3 select_sents.py \
    --data_path $data_path \
    --replace_fraction $replace_ratio \
    --output_path $sents_output_path
   ```
3. **Generating Code-Switching Content**:
   The next step is use the trained model to generate desired code-switching content.
   You can use vllm or sglang backend to achieve this.
   After getting the results, replace the original sentences with these new ones:
   ```bash
   python3 replace.py \
    --data_path $data_path \  # original pretraining docs path
    --replace_path $replace_path \ # the result file path of generation.
    --output_path $output_path
   ```
   The final output file is the new code-switched documents.

4. **Pre-training**:
   We use Megatron-LM for pre-training without modification to its code.
   