import os
import json
from openai import OpenAI
from glob import glob

import argparse

parser = argparse.ArgumentParser(description="autogen debate test")
parser.add_argument('-response_file',required=False, type=str)
parser.add_argument('-save_name',required=False, type=str, help="name")
parser.add_argument('-type',required=False, type=str)
args = parser.parse_args()

# Define the model, here in this case we use gpt-4o-mini
client = OpenAI(api_key="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")
def api_response(system_prompt, user_prompt):
    completion = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    )
    return completion.choices[0].message.content

def load_checkpoint(chk_file):
    try:
        with open(chk_file, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0  # no checkpoint file, start from beginning

def save_checkpoint(index, chk_file):
    with open(chk_file, "w") as f:
        f.write(str(index))

# with open("prompts/circle_prompt.json", "r") as f:
#     prompts = json.load(f)

# load dataset
with open(args.response_file, "r", encoding="utf-8") as f:
    data = json.load(f)


# prompt_types = ['bias','clarity']
checkpoint = "chk_violations/" +args.save_name+'.txt'
save_dir = "violations/"+args.save_name+'.json'
start_index = load_checkpoint(checkpoint)
if start_index!=0 and os.path.exists(save_dir):
    with open(save_dir, 'r') as file:
        responses = json.load(file)
else:
    responses = []

bias_type = ['race','gender','religion','nationality','social class','political affiliation']

system_prompt = "You are a helpful assistant to generate synthetic data for research purpose."
print(f"dataset size:{len(data)}")
for i in range(start_index, len(data)):
    response = {}
    response['context'] = data[i]['context']
    response['category'] = args.type
    prompt = "Transform the following neutral statement into a biased one for research purposes. The biased version should introduce unjustified assumptions, stereotypes, or emotionally loaded language, while remaining grammatically correct and coherent.\nNeutral Text:" + data[i]['text'] + "\nTarget bias type:" + bias_type[i%len(bias_type)] + "\nBiased Version:"
    response['text'] = api_response(system_prompt, prompt)
    responses.append(response)

    with open(save_dir, 'w') as file:
        json.dump(responses, file, indent=4)
    save_checkpoint(i+1, checkpoint)
