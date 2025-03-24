### generate trust scores and attention matrix, store for future use
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
parser.add_argument('-response_file', type=str, default = "responses/mmlubio_correct_benign.json")
parser.add_argument('-save_name', type=str, default = "mmlubio_correct_benign")
parser.add_argument('-prompt', type=str, default='trust-5')
parser.add_argument('-model', type=str, default='llama3')
parser.add_argument('-size', type=str, default='8B')

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

# load models
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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


geo_attentions_full = []
geo_attention_user = []
trust_scores = []

geo_attentions_full_file = "attentions/" + args.prompt + "_" + args.model + "_" + args.size + "_" + args.save_name + "_geo_attentions_full.json"
geo_attention_user_file = "attentions/" + args.prompt + "_" + args.model + "_" + args.size + "_" + args.save_name + "_geo_attention_user.json"
trust_scores_file = "self-aware-scores/" + args.prompt + "_" + args.model + "_" + args.size + "_" + args.save_name + ".json"

for i in range(len(response_data)):
    response=response_data[i]
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
        generated_ids = model.generate(inputs, max_new_tokens=512, do_sample=True, eos_token_id=tokenizer.eos_token_id)
        generated_ids=generated_ids[:,inputs.shape[1]:]
        output_texts = tokenizer.batch_decode(generated_ids)
        
        trust_score = extract_trust_rate(output_texts[0])
        trust_scores.append(trust_score)
        
        geo_head_layer_mat_full = torch.zeros([32,32])
        for layer_num in range(32):
            for attention_head_num in range(32):
                attention=attentions[layer_num][0, attention_head_num].cpu()
                geo_head_layer_mat_full[layer_num, attention_head_num]=geometric_mean(attention[-1,:-1])
        
        # only user query
        inputs_assistant = tokenizer.apply_chat_template([{"role": "assistant", "content": system_prompt}], return_tensors="pt")
        user_start = len(inputs_assistant[0])
        geo_head_layer_mat_user = torch.zeros([32,32])
        for layer_num in range(32):
            for attention_head_num in range(32):
                attention=attentions[layer_num][0, attention_head_num].cpu()
                geo_head_layer_mat_user[layer_num, attention_head_num]=geometric_mean(attention[-1,user_start+4:-1])

        geo_attentions_full.append(geo_head_layer_mat_full.tolist())
        geo_attention_user.append(geo_head_layer_mat_user.tolist())

    with open(geo_attentions_full_file, 'w') as file:
        json.dump(geo_attentions_full, file)

    with open(geo_attention_user_file, 'w') as file:
        json.dump(geo_attention_user, file)

    with open(trust_scores_file, 'w') as file:
        json.dump(trust_scores, file)