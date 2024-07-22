import json
from tqdm import tqdm
import sys


def get_evidence_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf8') as f:
        raw_data = f.readlines()
    raw_data = [json.loads(d) for d in raw_data]
    data = []
    for d in tqdm(raw_data):
        new_events = []
        for event in d['events']:
            new_mentions = []
            for mention in event['mention']:
                if mention['factuality'] == "CT+" or mention['factuality'] == "Uu":
                    continue
                if mention["evidence_word"] != None:
                    align = True
                    for offset in mention["evidence_offset"]:
                        if mention['sent_id'] != offset[0]:
                            align = False
                            break
                    if align:
                        new_mentions.append(mention)
            if new_mentions != []:
                event['mention'] = new_mentions
                new_events.append(event)
        if new_events != []:
            d['events'] = new_events
            data.append(d)
    with open(output_path, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    get_evidence_data("train.jsonl", "evidence_train.jsonl")
    get_evidence_data("valid.jsonl", "evidence_valid.jsonl")
    get_evidence_data("test.jsonl", "evidence_valid.jsonl")
