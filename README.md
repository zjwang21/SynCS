# Project Title

Investigating and Scaling up Code-Switching for Multilingual Language Model Pre-Training

# Code-Switching Data Synthesis

## Overview

This project synthesizes code-switching data for training purposes.

## Usage

1. **Training Code-Switching Data**:
   Run the following command to execute the training script:
   ```bash
   bash scripts/sft.sh
   ```
   You can find our sft data for Chinese, Romanian, and Bengali in the directory: data.
2. **Replacing Code-Switching Data**:
   Run the following command to replace the original sentences by the generated ones in your documents:
   ```bash
   python generate.py
   ```
3. **Pre-training**:
   We use Megatron-LM for pre-training without modification to its code.
   