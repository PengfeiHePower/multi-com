"""
generate trust scores and attention matrix, store for future use
example:
response_file: mmlubio_correct_circle.json
save_name: mmlubio_correct
Structure of save file: list of dictionaries. Each dictionary is like: d['benign']=[list of attention matrix]
"""

import os
from llm_loading import load_model_and_tokenizer
import torch
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import argparse
parser = argparse.ArgumentParser(description="generate trust scores")
parser.add_argument('-response_file', type=str, default = "responses/mmlubio_correct_circle.json")
parser.add_argument('-save_name', type=str, default = "mmlubio_correct")
parser.add_argument('-prompt', type=str, default='trust-5')
parser.add_argument('-model', type=str, default='llama3')
parser.add_argument('-size', type=str, default='8B')
parser.add_argument('-gpu', type=str, default='3')

args = parser.parse_args()

def extract_trust_rate(text):
    # Use a regular expression to find 'Trust rate: [number]%'
    match = re.search(r"Trust rate:\s*(\d+)", text)
    if match:
        # Convert the captured number to an integer
        return float(match.group(1))
    
def geometric_mean(tensor):
    log_tensor = torch.log(tensor)
    geometric_mean = torch.exp(torch.mean(log_tensor))
    return geometric_mean

def get_attentions(model,tokenizer, system_prompt, user_prompt):
    messages = [
        {"role": "assistant", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model(inputs)
    attentions = outputs.attentions
    generated_ids = model.generate(inputs, max_new_tokens=1000, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    generated_ids=generated_ids[:,inputs.shape[1]:]
    
    #include system prompt
    geo_head_layer_mat_full = torch.zeros([32,32])
    for layer_num in range(32):
        for attention_head_num in range(32):
            attention=attentions[layer_num][0, attention_head_num].cpu()
            geo_head_layer_mat_full[layer_num, attention_head_num]=geometric_mean(attention[-1,:-1])
    
    #only user_prompts
    inputs_assistant = tokenizer.apply_chat_template([{"role": "assistant", "content": system_prompt}], return_tensors="pt")
    user_start = len(inputs_assistant[0])
    geo_head_layer_mat_user = torch.zeros([32,32])
    for layer_num in range(32):
        for attention_head_num in range(32):
            attention=attentions[layer_num][0, attention_head_num].cpu()
            geo_head_layer_mat_user[layer_num, attention_head_num]=geometric_mean(attention[-1,user_start+4:-1])
    
    return geo_head_layer_mat_full, geo_head_layer_mat_user


# load models
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
model, tokenizer = load_model_and_tokenizer(args.model, args.size)
model.config.output_attentions = True

#load prompts
prompt_file_name="prompts/circle_prompt.json"
with open(prompt_file_name, 'r') as file:
    prompts = json.load(file)

#load responses
response_file_name=args.response_file
with open(response_file_name, 'r') as file:
    response_data = json.load(file)

geo_attentions_full_file = "attentions/" + args.prompt + "_" + args.model + "_" + args.size + "_" + args.save_name + "_geo_attentions_full.json"
geo_attention_user_file = "attentions/" + args.prompt + "_" + args.model + "_" + args.size + "_" + args.save_name + "_geo_attention_user.json"
# trust_scores_file = "self-aware-scores/" + args.prompt + "_" + args.model + "_" + args.size + "_" + args.save_name + ".json"

geo_attentions_full = []
geo_attentions_user = []

for i in range(len(response_data)):
    print(f"i:{i}")
    response=response_data[i]
    question = response['question']
    response_types = list(response.keys())[2:]
    system_prompt = prompts[args.prompt]+ "\n\nThe following is the problem to discuss:\n" +question
    geo_full_dict={}
    geo_user_dict={}
    for response_type in response_types:
        print(f"response_type:{response_type}")
        geo_full_dict[response_type]=[]
        geo_user_dict[response_type]=[]
        for j in range(len(response[response_type])):
            print(f"j:{j}")
            user_prompt = "Solver_1: " + response[response_type][j]
            geo_head_layer_mat_full, geo_head_layer_mat_user = get_attentions(model,tokenizer, system_prompt, user_prompt)
            geo_full_dict[response_type].append(geo_head_layer_mat_full.tolist())
            geo_user_dict[response_type].append(geo_head_layer_mat_user.tolist())
    geo_attentions_full.append(geo_full_dict)
    geo_attentions_user.append(geo_user_dict)

    with open(geo_attentions_full_file, 'w') as file:
        json.dump(geo_attentions_full, file)

    with open(geo_attention_user_file, 'w') as file:
        json.dump(geo_attentions_user, file)