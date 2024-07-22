# Supporting Evidence Detection

This directory contains codes for the supporting evidence detection experiments.

## Dataset

* Subset of non-factual events in MAVEN-FACT, you can use `../preprocess/evidence_subset.py` to generate the subset.

## Overview

* `data.py` is used for data preprocessing.
* `train.py` is used for training and evaluating models.
* `llm.py` is used for large language model inference.

## Usage

* Running bert-based models/GenEFD

  ```bash
  python train.py \
      --train_data train.jsonl \
      --test_data  test.jsonl \
      --model_dir models \
      --log_dir logs \
      --model_name roberta-large \
      --ckpt roberta-large \
      --batch_size 16 \
      --max_length 160 \
      --lr 1e-5 \
      --epochs 10 \
      --gpu 0 \
      --seed 42
  ```

  * `train_data` is the path of the training dataset.
  * `test_data` is the path of the test dataset.
  * `model_dir` is the directory where the model will be saved.
  * `log_dir` is the directory where the log will be saved.
  * `model_name` is the name of the backbone model. 
  * `ckpt` is the path of the checkpoint.
  * `max_length` is the maximum length for a single sentence.
  * `lr` is the learning rate for the optimizer.

* Running inference of large language models

  ```bash
  python llm.py \
      --data_path test.jsonl \
      --prompt_path ../prompts/prompt_for_sw.txt \
      --model_path Meta-Llama-3-8B-Instruct \
      --output_length 256 \
      --output_dir output \
      --gpu 2
  ```

  * `model_path` is the path to the checkpoint/API name of large language models. Options: [gpt-3.5-turbo, gpt-4, Meta-Llama-3-8B-Instruct]
  * `output_dir` is the directory where the output will be saved.
  * `output_length` is the maximum length of the generated output.
  * `data_path` is the path of the test dataset.
  * `prompt_path` is the path to the prompt template for inference.

