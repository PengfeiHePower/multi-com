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
parser.add_argument('-response_file', type=str, default = "responses/mmlubio_benign.json")
parser.add_argument('-score_file', type=str, default = "mmlubio_scores_benign-1.json")
parser.add_argument('-prompt', type=str, default='trust-5')
args = parser.parse_args()

# support functions
def format_multi(problem):
    question = problem['question']
    option_list = problem['options']
    formatted_text = ":\n"
    for index, option in enumerate(option_list, start=1):
        # Adding a letter label (A, B, C, D) before each option
        formatted_text += f"{chr(64 + index)}) {option}\n"
    options = formatted_text.strip()
    return question + "\n" +options

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


# load models
os.environ["CUDA_VISIBLE_DEVICES"]="1"
model, tokenizer = load_model_and_tokenizer('llama3', '8B')
model.config.output_attentions = True

#load prompts
prompt_file_name="prompts/circle_prompt.json"
with open(prompt_file_name, 'r') as file:
    prompts = json.load(file)

#load malicious responses
response_file_name=args.response_file
with open(response_file_name, 'r') as file:
    response_data = json.load(file)
    
malicious_mean_attentions_full = []
malicious_mean_attention_user = []
malicious_geo_attentions_full = []
malicious_geo_attention_user = []
malicious_trust_rates = []

for i in range(len(response_data)):
    response=response_data[i]
    # if 'mmlu' in args.response_file:
    #     question = format_multi(response)
    # elif 'gsm8k' in args.response_file:
    #     question = response['question']
    # elif 'math' in args.response_file:
    #     question = response['problem']
    question = response['question']
    system_prompt = prompts[args.prompt]+ "\n\nThe following is the problem to discuss:\n" +question
    for j in range(len(response['responses'])):
        user_prompt = "Solver_1: " + response['responses'][j]
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
        output_texts = tokenizer.batch_decode(generated_ids)
        
        trust_rate = extract_trust_rate(output_texts[0])
        malicious_trust_rates.append(trust_rate)
        
        head_layer_mat_full = torch.zeros([32,32])
        for layer_num in range(32):
            for attention_head_num in range(32):
                attention=attentions[layer_num][0, attention_head_num].cpu()
                head_layer_mat_full[layer_num, attention_head_num]=attention[-1,:-1].sum()
        geo_head_layer_mat_full = torch.zeros([32,32])
        for layer_num in range(32):
            for attention_head_num in range(32):
                attention=attentions[layer_num][0, attention_head_num].cpu()
                geo_head_layer_mat_full[layer_num, attention_head_num]=geometric_mean(attention[-1,:-1])
        
        # only user query
        inputs_assistant = tokenizer.apply_chat_template([{"role": "assistant", "content": system_prompt}], return_tensors="pt")
        user_start = len(inputs_assistant[0])
        head_layer_mat_user = torch.zeros([32,32])
        for layer_num in range(32):
            for attention_head_num in range(32):
                attention=attentions[layer_num][0, attention_head_num].cpu()
                head_layer_mat_user[layer_num, attention_head_num]=attention[-1,user_start+4:-1].sum()
        geo_head_layer_mat_user = torch.zeros([32,32])
        for layer_num in range(32):
            for attention_head_num in range(32):
                attention=attentions[layer_num][0, attention_head_num].cpu()
                geo_head_layer_mat_user[layer_num, attention_head_num]=geometric_mean(attention[-1,user_start+4:-1])
        
        # aggregated score
        mean_attention_full = torch.mean(head_layer_mat_full).item()
        mean_attention_user = torch.mean(head_layer_mat_user).item()
        mean_geo_attention_full = torch.mean(geo_head_layer_mat_full).item()
        mean_geo_attention_user = torch.mean(geo_head_layer_mat_user).item()
        
        malicious_mean_attentions_full.append(mean_attention_full)
        malicious_mean_attention_user.append(mean_attention_user)
        malicious_geo_attentions_full.append(mean_geo_attention_full)
        malicious_geo_attention_user.append(mean_geo_attention_user)

all_datas = [{
    'trust_rates':malicious_trust_rates,
    'mean_attention_full':malicious_mean_attentions_full,
    'mean_attention_user':malicious_mean_attention_user,
    'mean_geo_attention_full':malicious_geo_attentions_full,
    'mean_geo_attention_user':malicious_geo_attention_user,
            }]
with open('/egr/research-dselab/hepengf1/Documents/multi-com/attentionS/results/'+args.score_file, 'w') as file:
    json.dump(all_datas, file, indent=4)