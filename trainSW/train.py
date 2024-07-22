import argparse
import os
import logging
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from data import EvidenceDataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="train.jsonl")
    parser.add_argument("--test_data", type=str, default="test.jsonl")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="roberta-large")
    parser.add_argument("--ckpt", type=str, default="roberta-large")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_properties(device))
    model_save_path = os.path.join(args.model_dir, args.model_name)
    os.makedirs(model_save_path, exist_ok=True)
    log_save_path = os.path.join(args.log_dir, args.model_name)
    os.makedirs(log_save_path, exist_ok=True)


    logger = logging.getLogger("Evidence")
    handler = logging.FileHandler(os.path.join(log_save_path, f'log_sw_{args.model_name}_bs{args.batch_size}_lr{args.lr}_seed{args.seed}_ml{args.max_length}.log'))
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

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<PS+>', '</PS+>', '<PS->', '</PS->', '<CT->', '</CT->']})
    tokenizer.add_prefix_space = True

    train_dataset = EvidenceDataset(args.train_data, tokenizer, args.max_length)
    test_dataset = EvidenceDataset(args.test_data, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    id2label = {0: 'O', 1: 'B', 2: 'I'}
    label2id = {'O': 0, 'B': 1, 'I': 2}

    model = AutoModelForTokenClassification.from_pretrained(
        args.ckpt, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def postprocess(predictions, labels):
        label_names = ['O', 'B', 'I']
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # remove ignored index and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    best_test_f1 = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        all_preds = []
        all_labels = []
        for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch}"):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch['labels']
                true_labels, true_predictions = postprocess(predictions, labels)
                all_preds.extend(true_predictions)
                all_labels.extend(true_labels)

        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        logger.info(f"Epoch {epoch} test results:")
        logger.info(f"F1: {f1}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        if f1 > best_test_f1:
            best_test_f1 = f1
            logger.info(f"New best test f1: {f1}")
            # model.save_pretrained(model_save_path)


if __name__ == "__main__":
    main()