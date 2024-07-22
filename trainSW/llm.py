import os
import sys
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from openai import OpenAI
import argparse
from data import EvidenceDataset


API_KEY = ""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompt.txt")
    parser.add_argument("--output_length", type=int, default=256)
    parser.add_argument("--model_path", type=str, default="Llama")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()
    

def main():
    args = parse_args()
    
    # create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, "output.jsonl")

    test_set = EvidenceDataset(args.data_path, None, 160)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_input_for_SW(text):
        with open(args.prompt_path, "r") as f:
            prompt = f.read()
        prompt = prompt.replace("[INSERT TOKEN LIST HERE]", str(text))
        system_prompt = prompt.split("\n\n\n")[0]
        remain_prompt = prompt.split("\n\n\n")[1]
        return system_prompt, remain_prompt, prompt

    if "gpt" in args.model_path:
        client = OpenAI(api_key=API_KEY)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.add_special_tokens({'additional_special_tokens': ['<PS+>', '</PS+>', '<PS->', '</PS->', '<CT->', '</CT->']})
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto").to(device)
        model.resize_token_embeddings(len(tokenizer))


    preds = []
    for i in tqdm(range(len(test_set))):
        token_list = test_set.token_texts[i]
        system_prompt, remain_prompt, prompt = generate_input_for_SW(token_list)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": remain_prompt}
        ]
        if "gpt" in args.model_path:
            outputs = client.chat.completions.create(
                model=args.model_path,
                messages=messages,
                max_tokens=args.output_length,
                temperature=0.0
            )
            response = outputs.choices[0].message.content
        else:
            input_ids = tokenizer.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                return_tensors="pt"
            ).to(device)
            terminators = [
                            tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=args.output_length,
                            eos_token_id=terminators,
                            pad_token_id=tokenizer.pad_token_id,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            response = tokenizer.decode(response, skip_special_tokens=True)
        preds.append(response)

    results = []
    for i in range(len(preds)):
        results.append(
            {"text": test_set.token_texts[i], "label": test_set.token_labels[i], "pred": preds[i], "system": system_prompt, "remain": remain_prompt}
        )

    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()