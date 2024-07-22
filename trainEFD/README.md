# Event Factuality Detection

This directory contains codes for the event factuality detection experiments.

## Dataset

* MAVEN-FACT

## Overview

* `data.py`  is used for data preprocessing.
* `model.py` contains the implementations of DMBert.
* `train.py` is used for training and evaluating the BERT-based models.
* `train_t5.py` is used for training and evaluating the GenEFD.
* `llm.py` is used for large language model inference.

## Usage

* Running bert-based models

  ```bash
  python train.py \
      --train_data train.jsonl \
      --test_data  test.jsonl \
      --model_dir models \
      --log_dir logs \
      --model_name roberta-large \
      --model DMBert \
      --pooling_type cls \
      --ckpt roberta-large \
      --batch_size 16 \
      --max_length 160 \
      --lr 1e-5 \
      --epochs 10 \
      --gpu 0 \
      --dropout 0.1 \
      --seed 42 \
      --add_argument \
      --add_relation
  ```

  * `train_data` is the path of the training dataset.

  * `test_data` is the path of the test dataset.

  * `model_dir` is the directory where the model will be saved.

  * `log_dir` is the directory where the log will be saved.

  * `model_name` is the name of the backbone model.

  * `model` is the type of model to be trained. Options: [RawBert, DMBert]

  * `pooling_type` is the pooling method used in DMBERT. Options: [cls, mean, max]

  * `ckpt` is the path of the checkpoint.

  * `max_length` is the maximum length for a single sentence.

  * `lr` is the learning rate for the optimizer.

  * `add_argument` indicates whether to include argument information.

  * `add_relation` indicates whether to include relation information.

* Running GenEFD

  ```bash
  python train_t5.py \
      --train_data train.jsonl \
      --test_data  test.jsonl \
      --model_dir models \
      --log_dir logs \
      --model_name t5 \
      --ckpt flan-t5-base \
      --batch_size 16 \
      --max_length 384 \
      --lr 1e-5 \
      --epochs 10 \
      --gpu 0 \
      --seed 42 \
      --add_argument \
      --add_relation
  ```

  The parameters with the same name as in the previous section have the same meaning.

* Running inference of large language models

  ```bash
  python llm.py \
      --model_path Meta-Llama-3-8B-Instruct \
      --output_dir output \
      --output_length 16 \
      --data_dir test.jsonl \
      --num_shot 5 \
      --seed 42 \
      --batch_size 32 \
      --prompt_path ../prompts/prompt_for_efd.json \
      --shot_path ../prompts/shots_for_efd.json \
      --add_relation \
      --add_argument	\
      --add_cot
  
  ```

  * `model_path` is the path to the checkpoint/API name of large language models. Options: [gpt-3.5-turbo, gpt-4, Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2]
  * `output_dir` is the directory where the output will be saved.
  * `output_length` is the maximum length of the generated output.
  * `data_dir` is the path of the test dataset.
  * `num_shot` is the number of examples to use for few-shot learning.
  * `prompt_path` is the path to the prompt template for inference.
  * `shot_path` is the path to the examples for few-shot learning.
  * `add_cot`  indicates whether to include chain-of-thought reasoning.

