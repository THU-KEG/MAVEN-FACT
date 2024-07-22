import json
import os
from tqdm import tqdm
from openai import OpenAI
import re
from sklearn.metrics import precision_score, recall_score

API_KEY = ""

def generate_factuality(prompt, model="gpt-3.5-turbo"):
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        temperature=0.0,
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content


def document_filter(data_item):
    sentences = data_item['tokens'].copy()
    for event in data_item['events']:
        for mention in event['mention']:
            sentences[mention["sent_id"]][mention['offset'][0]] = "(" + "**" + sentences[mention["sent_id"]][mention['offset'][0]]
            sentences[mention["sent_id"]][mention["offset"][1] - 1] = sentences[mention["sent_id"]][mention["offset"][1] - 1] + "**" + ")"
    for j in range(len(sentences)):
        sentences[j] = " ".join(sentences[j])
    event_number = 0
    for event in data_item['events']:
        event_number += len(event['mention'])
    return ' '.join(sentences), event_number


def factuality_filter(document):
    pattern = re.compile(r"\(\*\*.*?\)\(([A-Za-z0-9\.+-]*)\)")
    factuality = []
    for match in pattern.finditer(document):
        factuality.append(match.group(1))
    return factuality

def get_annotation(prompt_path, output_path, data_path):
    print("Running inference on prompt: ", prompt_path)
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_dict = json.load(f)
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]

    data_cnt = 0
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            data_cnt = len(f.readlines())

    with open(output_path, 'a+', encoding='utf-8') as f:
        for item in tqdm(data[data_cnt:]):
            document, event_number = document_filter(item)
            if "rules" in prompt_dict:
                prompt = prompt_dict['task_definition'] + prompt_dict['label_definition'] + prompt_dict["rules"] + prompt_dict['input_prefix'] + document + prompt_dict['input_suffix']
            else:
                prompt = prompt_dict['task_definition'] + prompt_dict['label_definition'] + prompt_dict['input_prefix'] + document + prompt_dict['input_suffix']
            if prompt_dict["output_example"] != "":
                prompt += prompt_dict["output_example"]
            factuality = generate_factuality(prompt)
            retry = 0
            while len(factuality_filter(factuality)) != event_number:
                factuality = generate_factuality(prompt, model=prompt_dict["model"])
                print("retrying: {}".format(retry), end="\r")
                retry += 1
                if retry > 10:
                    factuality = " ".join(["(****)(OT)"] * event_number)
                    break
            item["document"] = document
            item["factuality"] = factuality.replace("\n", "")
            f.write(str(json.dumps(item, ensure_ascii=False)) + "\n")


def post_process(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    for item in data:
        event_number = 0
        for event in item['events']:
            event_number += len(event['mention'])
        factuality = factuality_filter(item['factuality'])
            
        factuality_list = []
        for event in item['events']:
            for mention in event['mention']:
                factuality_list.append({"sentence_id": mention["sent_id"], "trigger_word": mention["trigger_word"], "offset": mention["offset"]})
        factuality_list.sort(key=lambda x: x['sentence_id']*100000+x['offset'][0])
        for i in range(len(factuality_list)):
            factuality_list[i]['factuality'] = factuality[i]
        item['lm_label'] = factuality_list
    with open(input_path.replace(".jsonl", "_processed.jsonl"), 'w', encoding='utf-8') as f:
        for item in data:
            f.write(str(json.dumps(item, ensure_ascii=False)) + "\n") 


def match_factuality(input_path, output_path, prompt_path, pos="CT+"):
    print("Running inference on prompt: ", input_path)
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_dict = json.load(f)
    res = {
        "OT_recall": 0, 
        "CT+_recall": 0,    
        "CT+_precision": 0,
        "OT_precision": 0,
        "prompt": prompt_dict,
    }
    with open('./template/template_data/sample_annotated.jsonl', 'r', encoding='utf-8') as f:
        data_annotated = [json.loads(line) for line in f.readlines()]

    gold = []    

    for item in data_annotated:
        factuality = []
        for event in item['events']:
            for mention in event['mention']:
                factuality.append({"sentence_id": mention["sent_id"], "trigger_word": mention["trigger_word"], "offset": mention["offset"], "factuality": mention["factuality"]})
        factuality.sort(key=lambda x: x['sentence_id']*100000+x['offset'][0])
        gold.append(factuality)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data_lm = [json.loads(line) for line in f.readlines()]
    
    test = []
    for item in data_lm:
        test.append(item['lm_label'])
    
    gold_label = []
    test_label = []
    for i in range(len(gold)):
        for j in range(len(gold[i])):
            if(gold[i][j]['sentence_id'] != test[i][j]['sentence_id']):
                print("Sentence ID Error")
                raise ValueError
            if(gold[i][j]['trigger_word'] != test[i][j]['trigger_word']):
                print("Trigger Word Error")
                raise ValueError
            if(gold[i][j]['offset'] != test[i][j]['offset']):
                print("Offset Error")
                raise ValueError
            gold_label.append(gold[i][j]['factuality'])
            test_label.append(test[i][j]['factuality'])

    CTp_gold = [1 if label == "CT+" else 0 for label in gold_label]
    CTp_test = [1 if label == pos else 0 for label in test_label]
    res["CT+_precision"] = precision_score(CTp_gold, CTp_test)
    res["CT+_recall"] = recall_score(CTp_gold, CTp_test)
    OT_gold = [1 if label != "CT+" else 0 for label in gold_label]
    OT_test = [1 if label != "CT+" else 0 for label in test_label]
    res["OT_precision"] = precision_score(OT_gold, OT_test)
    res["OT_recall"] = recall_score(OT_gold, OT_test)


    with open(output_path, 'a+', encoding='utf-8') as f:
        f.write(str(json.dumps(res, ensure_ascii=False)) + "\n")


if __name__ == "__main__":
    while True:
        try:
            get_annotation("prompt_bi_cot.json")
            break
        except Exception as e:
            print(e)
            continue