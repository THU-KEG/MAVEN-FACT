import argparse
import os
import logging
import random

import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, T5Tokenizer, T5ForConditionalGeneration

from sklearn.metrics import f1_score, accuracy_score
from data import EFDDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="train.jsonl")
    parser.add_argument("--test_data", type=str, default="test.jsonl")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="t5")
    parser.add_argument("--ckpt", type=str, default="flan-t5-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add_argument", action="store_true")
    parser.add_argument("--add_relation", action="store_true")
    return parser.parse_args()


def evaluate(preds, labels, mode):
    gold_label = {"CT+": 0, "CT-": 1, "PS+": 2, "PS-": 3, "Uu": 4}
    gold_label_pair = {"CT": [0, 1], "PS": [2, 3], "p": [0, 2], "n": [1, 3]}

    if mode in gold_label:
        tp = sum([1 for p, l in zip(preds, labels) if p == gold_label[mode] and l == gold_label[mode]])
        fp = sum([1 for p, l in zip(preds, labels) if p == gold_label[mode] and l != gold_label[mode]])
        fn = sum([1 for p, l in zip(preds, labels) if p != gold_label[mode] and l == gold_label[mode]]) 
    elif mode in gold_label_pair:
        tp = sum([1 for p, l in zip(preds, labels) if p in gold_label_pair[mode] and l in gold_label_pair[mode]])
        fp = sum([1 for p, l in zip(preds, labels) if p in gold_label_pair[mode] and l not in gold_label_pair[mode]])
        fn = sum([1 for p, l in zip(preds, labels) if p not in gold_label_pair[mode] and l in gold_label_pair[mode]])
    else:
        raise ValueError("Invalid evaluation mode")

    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return precision, recall, f1

        

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available: ", torch.cuda.is_available())

    model_save_path = os.path.join(args.model_dir, args.model_name)
    os.makedirs(model_save_path, exist_ok=True)
    log_save_path = os.path.join(args.log_dir, args.model_name)
    os.makedirs(log_save_path, exist_ok=True)

    logger = logging.getLogger("EFD")
    log_file_name = f'log_t5_bs{args.batch_size}_ml{args.max_length}_lr{args.lr}'
    if args.add_argument:
        log_file_name += "_argument"
    if args.add_relation:
        log_file_name += "_relation"
    log_file_name += ".log"
    handler = logging.FileHandler(os.path.join(log_save_path, log_file_name))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    tokenizer = T5Tokenizer.from_pretrained(args.ckpt)
    if args.add_relation:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>", "<c>", "</c>", "<p>", "</p>"]})
    else:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})

    model = T5ForConditionalGeneration.from_pretrained(args.ckpt)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_dataset = EFDDataset(data_dir=args.train_data, tokenizer=tokenizer, max_length=args.max_length, add_argument=args.add_argument, add_relation=args.add_relation, model_type="t5")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = EFDDataset(data_dir=args.test_data, tokenizer=tokenizer, max_length=args.max_length, add_argument=args.add_argument, add_relation=args.add_relation, model_type="t5")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_test_f1 = 0
    label2id = {"CT+": 0, "CT-": 1, "PS+": 2, "PS-": 3, "Uu": 4}
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        logger.info(f"Epoch {epoch} finished in {time.time() - start_time} seconds")

        model.eval()
        with torch.no_grad():
            test_preds = []
            test_labels = []
            for batch in tqdm(test_dataloader, desc=f"Epoch {epoch} test"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=5)
                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                test_preds.extend(preds)
                test_labels.extend(labels)
            invalid_preds = [p for p in test_preds if p not in label2id]
            if invalid_preds:
                logger.info(f"Invalid test preds: {len(invalid_preds)}")
            test_preds = [label2id[p] if p in label2id else label2id["Uu"] for p in test_preds]
            test_labels = [label2id[l] for l in test_labels]
            test_preds = np.array(test_preds)
            test_labels = np.array(test_labels)
            tst_CTp_precision, tst_CTp_recall, tst_CTp_f1 = evaluate(test_preds, test_labels, "CT+")
            tst_CTn_precision, tst_CTn_recall, tst_CTn_f1 = evaluate(test_preds, test_labels, "CT-")
            tst_PSp_precision, tst_PSp_recall, tst_PSp_f1 = evaluate(test_preds, test_labels, "PS+")
            tst_PSn_precision, tst_PSn_recall, tst_PSn_f1 = evaluate(test_preds, test_labels, "PS-")
            tst_Uu_precision, tst_Uu_recall, tst_Uu_f1 = evaluate(test_preds, test_labels, "Uu")
            tst_macro_f1 = f1_score(test_labels, test_preds, average='macro')
            tst_micro_f1 = f1_score(test_labels, test_preds, average="micro")
            tst_accuracy = accuracy_score(test_labels, test_preds)

            logger.info(f"Epoch {epoch} test results:")
            logger.info(f"CT+ precision: {tst_CTp_precision}, recall: {tst_CTp_recall}, f1: {tst_CTp_f1}")
            logger.info(f"CT- precision: {tst_CTn_precision}, recall: {tst_CTn_recall}, f1: {tst_CTn_f1}")
            logger.info(f"PS+ precision: {tst_PSp_precision}, recall: {tst_PSp_recall}, f1: {tst_PSp_f1}")
            logger.info(f"PS- precision: {tst_PSn_precision}, recall: {tst_PSn_recall}, f1: {tst_PSn_f1}")
            logger.info(f"Uu precision: {tst_Uu_precision}, recall: {tst_Uu_recall}, f1: {tst_Uu_f1}")
            logger.info(f"Macro F1: {tst_macro_f1}")
            logger.info(f"Micro F1: {tst_micro_f1}")
            logger.info(f"Accuracy: {tst_accuracy}")

            if tst_macro_f1 > best_test_f1:
                best_test_f1 = tst_macro_f1
                # model.save_pretrained(model_save_path)
                # tokenizer.save_pretrained(model_save_path)
                logger.info(f"Best test model saved at epoch {epoch}")


if __name__ == "__main__":
    main()








            

