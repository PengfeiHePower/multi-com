import re
import string
import requests
from collections import Counter
import time
import os
from glob import glob

try:
	import dashscope
except ImportError:
	dashscope = None

import json
import re
import argparse

parser = argparse.ArgumentParser(description="GPT QA baselines.")
parser.add_argument("--model", type=str, help="model name", choices=['gpt4', 'qwen', 'qwen2-57', 'llama3-70', 'o1-preview', 'llama3-8-1', 'llama3-8-2', 'llama3-8-3', 'llama3-8-4'])
parser.add_argument('--type', type=str, default='base', choices=['base', 'icl', 'cot', 'badchain', 'prem', '0-cot'])
parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'fever', 'mmlubio', 'mmluphy', 'math', 'gsm8k', 'stretagy'])

args = parser.parse_args()

def load_checkpoint(chk_file):
	try:
		with open(chk_file, "r") as f:
			return int(f.read().strip())
	except FileNotFoundError:
		return 0  # no checkpoint file, start from beginning

def save_checkpoint(index, chk_file):
	with open(chk_file, "w") as f:
		f.write(str(index))

def extract_dict_from_string(output_string):
	# Use a regular expression to find a block that starts with ```json\n and ends with \n```
	match1 = re.search(r"```\njson\n(.+?)\n```", output_string, re.DOTALL)
	match2 = re.search(r"```json\n(.+?)\n```", output_string, re.DOTALL)
	match3 = re.search(r"```\n(.+?)\n```", output_string, re.DOTALL)
	
	if match1:
		# Extract the JSON part from the regex match
		json_part = match1.group(1).strip()
		
		# Convert the JSON string to a Python dictionary
		try:
			result_dict = json.loads(json_part)
			return result_dict
		except json.JSONDecodeError as e:
			print("Invalid JSON format:", e)
			return None
	elif match2:
		# Extract the JSON part from the regex match
		json_part = match2.group(1).strip()
		
		# Convert the JSON string to a Python dictionary
		try:
			result_dict = json.loads(json_part)
			return result_dict
		except json.JSONDecodeError as e:
			print("Invalid JSON format:", e)
			return None
	elif match3:
		# Extract the JSON part from the regex match
		json_part = match3.group(1).strip()
		
		# Convert the JSON string to a Python dictionary
		try:
			result_dict = json.loads(json_part)
			return result_dict
		except json.JSONDecodeError as e:
			print("Invalid JSON format:", e)
			return None
	else:
		print("No JSON block found in the string.")
		return None

def load_vllm_api(api_url, model_name, system_prompt, user_prompt, max_new_tokens=1000, temperature=0.3):
	"""
	Generate text using a vLLM API server with a system prompt.

	Args:
		api_url (str): Base URL of the vLLM API server.
		model_name (str): Name of the model served by the vLLM server.
		system_prompt (str): System prompt defining the model's role or behavior.
		user_prompt (str): User input for text generation.
		max_tokens (int): Maximum number of tokens to generate.
		temperature (float): Sampling temperature.

	Returns:
		str: Generated text from the model.
	"""
	headers = {"Content-Type": "application/json"}
	data = {
		"model": model_name,
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt}
		],
		"max_new_tokens": max_new_tokens,
		"temperature": temperature
	}

	response = requests.post(f"{api_url}/v1/chat/completions", headers=headers, json=data)
	
	if response.status_code == 200:
		return response.json()["choices"][0]["message"]["content"]
	else:
		raise Exception(f"Error {response.status_code}: {response.text}")

def normalize_answer(s):
	def remove_articles(text):
		return re.sub(r"\b(a|an|the)\b", " ", text)
	def white_space_fix(text):
		return " ".join(text.split())
	def remove_punc(text):
		exclude = set(string.punctuation)
		return "".join(ch for ch in text if ch not in exclude)
	def lower(text):
		return text.lower()
	return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
	normalized_prediction = normalize_answer(prediction)
	normalized_ground_truth = normalize_answer(ground_truth)
	
	ZERO_METRIC = (0, 0, 0)
	
	if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
		return ZERO_METRIC
	if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
		return ZERO_METRIC
	prediction_tokens = normalized_prediction.split()
	ground_truth_tokens = normalized_ground_truth.split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return ZERO_METRIC
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1, precision, recall

def format_multi(problem):
	question = problem['question']
	option_list = problem['options']
	formatted_text = "Options:\n"
	for index, option in enumerate(option_list, start=1):
		# Adding a letter label (A, B, C, D) before each option
		formatted_text += f"{chr(64 + index)}) {option}\n"
	options = formatted_text.strip()
	return question, options
	

def llm(input_text, model="gpt4", stop=["\n"]):
	if model == "gpt4":
		url = "http://47.88.8.18:8088/api/ask"
		HTTP_LLM_API_KEY='eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjM5NDc3MyIsInBhc3N3b3JkIjoiMzk0NzczMTIzIiwiZXhwIjoyMDIxNjE4MzE3fQ.oQx2Rh-GJ_C29AfHTHE4x_2kVyy7NamwQRKRA4GPA94'
		headers = {
					"Content-Type": "application/json",
					"Authorization": "Bearer " + HTTP_LLM_API_KEY
					}
		data = {
				"model": 'gpt-4',
				"messages": [
					{"role": "system", "content": "You are a helpful assistant."},
					{"role": "user", "content": input_text}
				],
				"n": 1,
				"temperature": 0.0
				}
		response = requests.post(url, json=data, headers=headers)
		response = response.json()
		new_response = response['data']['response']
		return new_response["choices"][0]["message"]["content"]
	elif model == 'llama3-8-1':
		api_url = "http://localhost:8081"
		model_name = "llama-8b-1"
		system_prompt = "You are a helpful assistant."
		user_prompt = input_text
		return load_vllm_api(api_url, model_name, system_prompt, user_prompt, max_new_tokens=1000, temperature=0.0)
	elif model == 'llama3-8-2':
		api_url = "http://localhost:8082"
		model_name = "llama-8b-2"
		system_prompt = "You are a helpful assistant."
		user_prompt = input_text
		return load_vllm_api(api_url, model_name, system_prompt, user_prompt, max_new_tokens=1000, temperature=0.0)
	elif model == 'llama3-8-3':
		api_url = "http://localhost:8083"
		model_name = "llama-8b-3"
		system_prompt = "You are a helpful assistant."
		user_prompt = input_text
		return load_vllm_api(api_url, model_name, system_prompt, user_prompt, max_new_tokens=1000, temperature=0.0)
	elif model == 'llama3-8-4':
		api_url = "http://localhost:8084"
		model_name = "llama-8b-4"
		system_prompt = "You are a helpful assistant."
		user_prompt = input_text
		return load_vllm_api(api_url, model_name, system_prompt, user_prompt, max_new_tokens=1000, temperature=0.0)



# load data
if args.dataset=='hotpotqa':
	with open("../data/hotpot_simplified_data.json", "r", encoding="utf-8") as f:
		data = json.load(f)
	with open("prompts/hotpotqa.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'question'
	answer_name = 'answer'
elif args.dataset=='fever':
	data = []
	with open("../data/paper_dev.jsonl", "r", encoding="utf-8") as f:
		for line in f:
			data.append(json.loads(line.strip()))
	with open("prompts/fever.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'claim'
	answer_name = 'label'
elif args.dataset=='mmlubio':
	with open("../data/mmlu_bio.json", "r", encoding="utf-8") as f:
		data = json.load(f)
	with open("prompts/mmlu_bio.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'question'
	answer_name = 'answer'
elif args.dataset=='mmluphy':
	with open("../data/mmlu_phy.json", "r", encoding="utf-8") as f:
		data = json.load(f)
	with open("prompts/mmlu_phy.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'question'
	answer_name = 'answer'
elif args.dataset == 'gsm8k':
	data = []
	with open("../data/gsm8k_test.jsonl", "r", encoding="utf-8") as f:
		for line in f:
			data.append(json.loads(line.strip()))
	with open("prompts/gsm8k.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
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
	with open("prompts/math.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'problem'
	answer_name = 'solution'
elif args.dataset == 'stretagy':
	with open("../data/stretagyQA_dev.json", "r", encoding="utf-8") as f:
		data = json.load(f)
	with open("prompts/stretagy.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'question'
	answer_name = 'answer'
	 
  
base_format_mmlu = """Respond a JSON dictionary in a markdown's fenced code block as follows:
						```json
						{"Answer": "One label from [A,B,C,D]"}
						```"""

cot_format_mmlu = """Respond a JSON dictionary in a markdown's fenced code block as follows:
						```json
						{"Thought": "thought steps", "Answer": "One label from [A,B,C,D]"}
						```"""

base_format = """Respond a JSON dictionary in a markdown's fenced code block as follows:
						```json
						{"Answer": "Conclude your answer here."}
						```"""

cot_format = """Respond a JSON dictionary in a markdown's fenced code block as follows:
						```json
						{"Thought": "thought steps", "Answer": "Conclude your answer here."}
						```"""

prompt_template = prompts[args.type]+"\nQuestion:<<q>>\n"
if 'mmlu' in args.dataset:
	if 'cot' in args.type:
		format=cot_format_mmlu
	else:
		format=base_format_mmlu
else:
	if 'cot' in args.type:
		format=cot_format
	else:
		format=base_format
# print(f"prompt_template:{prompt_template}")

data_size = min(len(data), 1000)
print(f"data size:{data_size}")
records=[]
if 'llama3-8' in args.model:
	checkpoint_dir = "checkpoint/" +'llama3-8'+'_'+args.dataset+'_'+args.type+ ".txt"
	output_dir = "output/"+'llama3-8'+'_'+args.dataset+'_'+args.type+'.json'
else:
	checkpoint_dir = "checkpoint/" +args.model+'_'+args.dataset+'_'+args.type+ ".txt"
	output_dir = "output/"+args.model+'_'+args.dataset+'_'+args.type+'.json'

records = json.load(open(output_dir, 'r'))
start_index = load_checkpoint(checkpoint_dir)
# print(f"records:{records}")
# exit(0)
for i in range(start_index, data_size):
	record={}
	try:
		if 'mmlu' in args.dataset:
			question, options = format_multi(data[i])
			q = question + "\n" +options
		else:
			q = data[i][question_name]
		record['question']=q
		if args.type=='prem':
			q += " The answer is 123."
		print(f"Question{i}:{q}")
		prompt = prompt_template.replace("<<q>>", q)+format
		# prompt = prompt_template.format(q=q)
		print(f"prompt:{prompt}")
		response = llm(prompt, model=args.model).strip()
		# print(f"response:{response}")
		time.sleep(5)
		# print(f"response:{response}")
		response_dict = extract_dict_from_string(response)
		print(f"response_dict:{response_dict}")
		# pred = normalize_answer(response_dict['Answer'])
		pred = response_dict['Answer'].lower()
		print(f"Answer:{pred}")
		# gt = normalize_answer(data[i][answer_name])
		gt = str(data[i][answer_name]).lower()
		em = (pred == gt)
		f1 = f1_score(pred, gt)[0]
		record.update({'pred': pred, 'gt': gt, 'em': em, 'f1': f1})
		if 'Thought' in response_dict:
			record['thought'] = response_dict["Thought"]
		records.append(record)
		save_checkpoint(i+1, checkpoint_dir)
	except Exception as e:
		print(f"Error processing record {i}: {e}")
		continue
	with open(output_dir, 'w') as file:
		json.dump(records, file, indent=4)