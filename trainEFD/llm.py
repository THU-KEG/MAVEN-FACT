import os
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import EFDDataset
import torch
from openai import OpenAI

API_KEY = "" 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="Mistral-7B-Instruct-v0.2")
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--output_length', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='test.jsonl')
    parser.add_argument("--num_shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prompt_path", type=str, default="prompt.json")
    parser.add_argument("--shot_path", type=str, default="shot.json")
    parser.add_argument("--add_relation", action="store_true")
    parser.add_argument("--add_argument", action="store_true")
    parser.add_argument("--add_cot", action="store_true")
    return parser.parse_args()


def generate_input(prompt_path, shot_path, text, num_shot, cot=None, cause=None, precondition=None, argument=None):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = json.load(f)
    output = prompt["prompt"]
    if cot:
        output += prompt["rules"]
    with open(shot_path, "r", encoding="utf-8") as f:
        shot = json.load(f)
    output += "Here are some examples:\n\n"
    for i in range(num_shot):
        output += "TEXT: " + shot[i]["text"] + "\n"
        output += "LABEL: " + shot[i]["label"] + "\n"
        output += "\n"
    system_output = output
    
    if (cause and cause != []) or (precondition and precondition != []) or (argument and argument != ""):
        output += "For your reference,\n"
    if (cause and cause != []) or (precondition and precondition != []):
        output += prompt["relation"]
    if cause and cause != []:
        output += "Cause Relations: "
        for i in range(len(cause)):
            output += cause[i] + "\n"
        output += "\n"
    if precondition and precondition != []:
        output += "Precondition Relations: "
        for i in range(len(precondition)):
            output += precondition[i] + "\n"
        output += "\n"
    if argument and argument != "":
        output += prompt["argument"]
        output += "Arguments:"
        output += argument
        output += "\n\n"
    output += "Here is the text you need to generate the label for, please do not output other information other than the label.\n"
    output += "TEXT: " + text + "\n"
    output += f"LABEL: "
    remain_output = output[len(system_output):]
    return system_output, output, remain_output
  

def generate_output():
    args = parse_args()
    print("num_shot: ", args.num_shot)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    output_name = f"result_{os.path.basename(args.model_path)}_{args.num_shot}shot"
    if args.add_cot:
        output_name += "_cot"
    if args.add_relation:
        output_name += "_relation"
    if args.add_argument:
        output_name += "_argument"
    output_name += ".jsonl"

    output_path = os.path.join(args.output_dir, output_name)
            
    
    if "Llama" in args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.add_special_tokens({'additional_special_tokens': ['<e>', '</e>', '<c>', '</c>', '<p>', '</p>']})
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto").to(device)
        model.resize_token_embeddings(len(tokenizer))
    elif "Mistral" in args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.add_special_tokens({'additional_special_tokens': ['<e>', '</e>', '<c>', '</c>', '<p>', '</p>']})
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16).to(device)
        model.resize_token_embeddings(len(tokenizer))
    else:
        client = OpenAI(api_key=API_KEY)


    dataset = EFDDataset(args.data_dir, None, None, args.add_relation, args.add_argument, "llm")
    label_set = dataset.labels
    texts = dataset.texts
    causes = dataset.causes
    preconditions = dataset.preconditions
    arguments = dataset.arguments
    for i in tqdm(range(len(dataset)), desc=f"Generating outputs with {args.model_path}"):
        if args.add_relation and args.add_argument:
            system_prompt, prompt, remain_prompt = generate_input(args.prompt_path, args.shot_path, texts[i], args.num_shot, cot=args.add_cot, cause=causes[i], precondition=preconditions[i], argument=arguments[i])
        elif args.add_argument:
            system_prompt, prompt, remain_prompt = generate_input(args.prompt_path, args.shot_path, texts[i], args.num_shot, cot=args.add_cot, argument=arguments[i])
        elif args.add_relation:
            system_prompt, prompt, remain_prompt = generate_input(args.prompt_path, args.shot_path, texts[i], args.num_shot, cot=args.add_cot, cause=causes[i], precondition=preconditions[i])
        else:
            system_prompt, prompt, remain_prompt = generate_input(args.prompt_path, args.shot_path, texts[i], args.num_shot, cot=args.add_cot)

        if "Llama" in args.model_path:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": remain_prompt}
            ]
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
            res = {"text": texts[i], "label": label_set[i], "output": tokenizer.decode(response, skip_special_tokens=True)}
        elif "Mistral" in args.model_path:
            messages = [
                {"role": "user", "content": prompt}
            ]
            model_inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            outputs = model.generate(
                model_inputs,
                max_new_tokens=args.output_length,
                do_sample=True
            )
            res = {"text": texts[i], "label": label_set[i], "output": tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("[/INST]")[1]}
        else: # gpt
            output = client.chat.completions.create(
                model=args.model_path,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": remain_prompt}
                ],
                max_tokens=args.output_length,
                temperature=0.0
            )
            output = output.choices[0].message.content
            res = {"text": texts[i], "label": label_set[i], "output": output}
        res["system_prompt"] = system_prompt
        res["prompt"] = prompt
        res["remain_prompt"] = remain_prompt

        with open(output_path, "a+", encoding="utf-8") as f:
            f.write(str(json.dumps(res, ensure_ascii=False)) + "\n")

if __name__ == "__main__":
    generate_output()               