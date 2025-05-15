import os
import json
from openai import OpenAI
from glob import glob

import argparse

parser = argparse.ArgumentParser(description="autogen debate test")
parser.add_argument('-dataset',required=False, type=str, help="dataset name")
parser.add_argument('-type',required=False, type=str, choices=['original','fact', 'logic', 'revelant', 'bias', 'quality', 'clarity'], help="prompt type")
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

with open("prompts/circle_prompt.json", "r") as f:
    prompts = json.load(f)

# load dataset
if args.dataset=='mmlubio':
    with open("../data/mmlu_bio.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    question_name = 'question'
    answer_name = 'answer'
elif args.dataset=='mmluphy':
    with open("../data/mmlu_phy.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    question_name = 'question'
    answer_name = 'answer'
elif args.dataset == 'human':
    with open("../human-eval/data/HumanEval.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    question_name = 'prompt'
    answer_name = 'canonical_solution'
elif args.dataset == 'mbpp':
    with open("../MBPP/mbpp.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    question_name = 'text'
    answer_name = 'code'
elif args.dataset == 'gsm8k':
	data = []
	with open("../data/gsm8k_test.jsonl", "r", encoding="utf-8") as f:
		for line in f:
			data.append(json.loads(line.strip()))
	question_name = 'question'
	answer_name = 'answer'
elif args.dataset == 'math':
    data = []
    dataset_dir = '../data/MATH_test'
    for subdir in os.listdir(dataset_dir):
        subdir_path = os.path.join(dataset_dir, subdir)
        if os.path.isdir(subdir_path):
            json_files = glob(os.path.join(subdir_path, "*.json"))
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data.append(json.load(f))
    question_name = 'problem'
    answer_name = 'solution'


# prompt_types = ['original','fact', 'logic', 'revelant', 'bias', 'quality', 'clarity']
save_dir = "violations/"+args.dataset + "_" + args.type + '.json'
checkpoint = "chk_violations/" +args.dataset+"_" + args.type+'.txt'
start_index = load_checkpoint(checkpoint)

if start_index!=0 and os.path.exists(save_dir):
    with open(save_dir, 'r') as file:
        responses = json.load(file)
else:
    responses = []

system_prompt = "You are an assistant tasked with generating examples for research purposes."
print(f"dataset size:{len(data)}")
# start_index, len(data)
for i in range(start_index, len(data)):
    question = data[i][question_name]
    answer = data[i][answer_name]
    response = {}
    response['context'] = "Here is the question:\n"+question
    response['category'] = args.type
    if args.dataset == 'human' and args.type == 'original':
        prompt = prompts['original-code'] + "\n\n" + response['context']
    else:
        prompt = prompts[args.type] + "\n\n" + response['context']
    response['text'] = 'Solver_1: ' + api_response(system_prompt, prompt)
    responses.append(response)
    with open(save_dir, 'w') as file:
        json.dump(responses, file, indent=4)
    save_checkpoint(i+1, checkpoint)
