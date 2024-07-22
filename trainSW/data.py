import os
import sys
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm

sys.path.append(os.getcwd())

class EvidenceDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len, model="roberta"):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model = model
        self.token_texts, self.token_labels = self.load_data()

    def load_data(self):
        with open(self.data_dir, "r", encoding="utf8") as f:
            data = [json.loads(line) for line in f]
        token_texts = []
        token_labels = []
        # label2id = {'O': 0, 'B': 1, 'I': 2}
        for item in tqdm(data):
            for event in item['events']:
                for mention in event['mention']:
                    evi_offsets = []
                    for offset in mention["evidence_offset"]:
                        evi_offsets.append(offset[1])
                    tokens = item['tokens'][mention['sent_id']]
                    label = [0] * len(tokens)
                    
                    label[evi_offsets[0]] = 1
                    for i in range(1, len(evi_offsets)):
                        if evi_offsets[i] == evi_offsets[i-1] + 1:
                            label[evi_offsets[i]] = 2
                        else:
                            label[evi_offsets[i]] = 1
                    # add <PS+> forword to the mention token and <PS-> for the following tokens
                    text_tokens = tokens[:mention['offset'][0]] + [f"<{mention['factuality']}>"] + tokens[mention['offset'][0]:mention['offset'][1]] + [f"</{mention['factuality']}>"] + tokens[mention['offset'][1]:]
                    label = label[:mention['offset'][0]] + [0] + label[mention['offset'][0]:mention['offset'][1]] + [0] + label[mention['offset'][1]:]
                    token_texts.append(text_tokens)
                    token_labels.append(label)
        return token_texts, token_labels
    
    def show_data(self, index):
        print(" ".join(self.token_texts[index]))
        for i in range(len(self.token_texts[index])):
            print(self.token_labels[index][i], end="")
            length = len(str(self.token_labels[index][i]))
            print(" "*(len(self.token_texts[index][i])-length), end="")
        print()
    
    def load_from_file(self, file_path):
        token_texts = []
        with open(file_path, "r", encoding="utf8") as f:
           data = [json.loads(line) for line in f]
        for item in data:
            texts = item['text'].replace("<e>", f"<{item['pred']}> ").replace("</e>", f" </{item['pred']}>")
            if self.model == "t5":
                texts = texts.replace("Event factuality prediction : ", "")
            token_texts.append(texts.split())
        for i in range(len(token_texts)):
            if len(token_texts[i]) != len(self.token_texts[i]):
                print(token_texts[i])
                print(self.token_texts[i])
        self.token_texts = token_texts


    def __len__(self):
        return len(self.token_texts)
    
    def __getitem__(self, idx):
        def align_labels_with_tokens(labels, word_ids):
            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    new_labels.append(-100)
                else:
                    label = labels[word_id]
                    # If the label is B, we need to change it to I
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)
            return new_labels
        
        inputs = self.tokenizer(self.token_texts[idx], is_split_into_words=True, padding="max_length", truncation=True, max_length=self.max_len)
        input_ids = torch.tensor(inputs.input_ids)
        attention_mask = torch.tensor(inputs.attention_mask)
        word_ids = inputs.word_ids()
        labels = align_labels_with_tokens(self.token_labels[idx], word_ids)
        labels = torch.tensor(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        
    