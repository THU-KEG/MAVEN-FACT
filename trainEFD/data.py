import os
import sys
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm

sys.path.append(os.getcwd())

class EFDDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length, add_relation, add_argument, model_type="bert"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dir = data_dir
        self.add_relation = add_relation
        self.add_argument = add_argument
        self.model_type = model_type
        self.texts, self.labels, self.arguments, self.causes, self.preconditions = self.load_data()

    def load_data(self):
        def parse_sentence(doc, sent_id, offsets, special_mark):
            if isinstance(offsets[0], int):
                doc['tokens'][sent_id][offsets[0]] = special_mark[0] + doc['tokens'][sent_id][offsets[0]]
                doc['tokens'][sent_id][offsets[1] - 1] = doc['tokens'][sent_id][offsets[1] - 1] + special_mark[1]
                text = " ".join(doc['tokens'][sent_id])
                doc['tokens'][sent_id][offsets[0]] = doc['tokens'][sent_id][offsets[0]].replace(special_mark[0], "")
                doc['tokens'][sent_id][offsets[1] - 1] = doc['tokens'][sent_id][offsets[1] - 1].replace(special_mark[1], "")
            else:
                for offset in offsets:
                    doc['tokens'][sent_id][offset[0]] = special_mark[0] + doc['tokens'][sent_id][offset[0]]
                    doc['tokens'][sent_id][offset[1] - 1] = doc['tokens'][sent_id][offset[1] - 1] + special_mark[1]
                text = " ".join(doc['tokens'][sent_id])
                for offset in offsets:
                    doc['tokens'][sent_id][offset[0]] = doc['tokens'][sent_id][offset[0]].replace(special_mark[0], "")
                    doc['tokens'][sent_id][offset[1] - 1] = doc['tokens'][sent_id][offset[1] - 1].replace(special_mark[1], "")
            return text
        
        with open(self.data_dir, 'r', encoding='utf8') as f:
            raw_data = [json.loads(line) for line in f]
        label2id = {'CT+': 0, 'CT-': 1, "PS+": 2, "PS-": 3, "Uu": 4}
        texts = []
        labels = []
        arguments = []
        causes = []
        preconditions = []

        for item in tqdm(raw_data, desc="Loading data"):
            events = {}
            for event in item['events']:
                events[event['id']] = event
            for event in item['events']:
                if self.add_relation:
                    cause = []
                    precondition = []
                    cause_map = {}
                    for relation in item["causal_relation"]["CAUSE"]:
                        if event['id'] == relation[1]:
                            for mention in events[relation[0]]['mention']:
                                if mention['sent_id'] not in cause_map:
                                    cause_map[mention['sent_id']] = [mention['offset']]
                                else:
                                    cause_map[mention['sent_id']].append(mention['offset'])
                    for key in cause_map:
                        cause.append(parse_sentence(item, key, cause_map[key], ["<c>", "</c>"]))

                    precondition_map = {}
                    for relation in item["causal_relation"]["PRECONDITION"]:
                        if event['id'] == relation[1]:
                            for mention in events[relation[0]]['mention']:
                                if mention['sent_id'] not in precondition_map:
                                    precondition_map[mention['sent_id']] = [mention['offset']]
                                else:
                                    precondition_map[mention['sent_id']].append(mention['offset'])
                    for key in precondition_map:
                        precondition.append(parse_sentence(item, key, precondition_map[key], ["<p>", "</p>"]))

                for mention in event['mention']:
                    text = parse_sentence(item, mention['sent_id'], mention['offset'], ["<e>", "</e>"])
                    if self.model_type == "t5":
                        text = "Event factuality prediction : " + text
                    texts.append(text)
                    if self.model_type == "bert":
                        labels.append(label2id[mention['factuality']])
                    else:
                        labels.append(mention["factuality"])
                    if self.add_argument:
                        arg = ""
                        if self.model_type == "llm":
                            for argument in event['arguments']:
                                arg += "TYPE: " + argument['type'] + "; ENTITY: " + argument['mentions'][0]['mention'] + ". "
                        else:
                            for argument in event['arguments']:
                                arg += "type: " + argument['type'] + " text: " + argument['mentions'][0]['mention'] + "; "
                        arguments.append(arg)
                    if self.add_relation:
                        causes.append(cause)
                        preconditions.append(precondition)
        return texts, labels, arguments, causes, preconditions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.model_type == "bert":
            source = self.tokenizer.encode_plus(
                self.texts[idx],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            maskL = torch.zeros(self.max_length)
            maskR = torch.zeros(self.max_length)
            try:
                pos1 = source['input_ids'].squeeze().tolist().index(self.tokenizer.convert_tokens_to_ids('<e>'))
                pos2 = source['input_ids'].squeeze().tolist().index(self.tokenizer.convert_tokens_to_ids('</e>'))
            except Exception as e:
                pos1 = 0
                pos2 = len(source['input_ids'].squeeze())
            maskL[:pos2+1] = 1.0
            maskR[pos1:] = 1.0

            source_ids = source['input_ids'].squeeze()
            source_mask = source['attention_mask'].squeeze()
            label = torch.tensor(self.labels[idx])
            return_dict = {
                'input_ids': source_ids,
                'attention_mask': source_mask,
                'labels': label,
                'maskL': maskL,
                'maskR': maskR
            }
            if self.add_argument:
                arg_source = self.tokenizer.encode_plus(
                    self.arguments[idx],
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                arg_ids = arg_source['input_ids'].squeeze()
                arg_mask = arg_source['attention_mask'].squeeze()
                return_dict['arg_ids'] = arg_ids
                return_dict['arg_mask'] = arg_mask
            if self.add_relation:
                cause_all = " ".join(self.causes[idx])
                cause_source = self.tokenizer.encode_plus(
                    cause_all,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                cause_ids = cause_source['input_ids'].squeeze()
                cause_attention_mask = cause_source['attention_mask'].squeeze()

                precondition_all = " ".join(self.preconditions[idx])
                precondition_source = self.tokenizer.encode_plus(
                    precondition_all,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                precondition_ids = precondition_source['input_ids'].squeeze()
                precondition_attention_mask = precondition_source['attention_mask'].squeeze()
                return_dict['cause_ids'] = cause_ids
                return_dict['precondition_ids'] = precondition_ids
                return_dict['cause_mask'] = cause_attention_mask
                return_dict['precondition_mask'] = precondition_attention_mask
            return return_dict
        else:
            input_texts = self.texts[idx]
            if self.add_argument:
                if self.arguments[idx] != "":
                    input_texts = input_texts + "ARGUMENTS: " + self.arguments[idx]
            if self.add_relation:
                if self.causes[idx] != []:
                    cause_texts = " ".join(self.causes[idx])
                    input_texts = input_texts + "EVENT CAUSE: " + cause_texts
                if self.preconditions[idx] != []:
                    precondition_texts = " ".join(self.preconditions[idx])
                    input_texts = input_texts + "EVENT PRECONDITION: " + precondition_texts
            source = self.tokenizer.encode_plus(
                input_texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            target = self.tokenizer.encode_plus(
                self.labels[idx],
                max_length=5,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = source['input_ids'].squeeze()
            attention_mask = source['attention_mask'].squeeze()
            labels = target['input_ids'].squeeze()
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }