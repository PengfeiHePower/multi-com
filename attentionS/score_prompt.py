import torch
import json
import os
import re
from llm_loading import load_model_and_tokenizer
import time

import argparse
parser = argparse.ArgumentParser(description="generate trust scores")
parser.add_argument('-response_file', type=str, default = "responses/mmlubio_correct_circle.json")
parser.add_argument('-save_name', type=str, default = "mmlubio_correct")
parser.add_argument('-model', type=str, default='llama3')
parser.add_argument('-prompt', type=str, default='trust-rubric')
parser.add_argument('-size', type=str, default='8B')
parser.add_argument('-gpu', type=str, default='3')

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

def string_to_dict(output_string):
    pattern = re.compile(
        r"```json\s*(\{[\s\S]*?\})\s*```",  # 1 backslash, DOTALL via [\s\S]
        re.MULTILINE,)
    match = pattern.search(output_string)
    if not match:
        raise ValueError("No JSON block found")
    data = json.loads(match.group(1))
    return data

format = """Respond a JSON dictionary in a markdown's fenced code block as follows:
```json
{
  "prompt_scores": {
    "factual_accuracy": [percentage only],
    "logical_consistency": [percentage only],
    "relevance": [percentage only],
    "neutrality_bias": [percentage only],
    "language_quality": [percentage only],
    "clarity_precision": [percentage only]
  },
  "prompt_overall_score": [percentage only],
  "prompt_rationale": {
    "factual_accuracy": "<justification>",
    "logical_consistency": "<justification>",
    "relevance": "<justification>",
    "neutrality_bias": "<justification>",
    "language_quality": "<justification>",
    "clarity_precision": "<justification>"
  }
}
```"""

# load models
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
model, tokenizer = load_model_and_tokenizer(args.model, args.size)
model.config.output_attentions = True
if args.model == "gemma3":
    model = model.to(torch.float32)

#load responses
response_file_name=args.response_file
with open(response_file_name, 'r') as file:
    response_data = json.load(file)

#load prompts
prompt_file_name="prompts/circle_prompt.json"
with open(prompt_file_name, 'r') as file:
    prompts = json.load(file)

system_prompt = prompts[args.prompt]
prompt_score_file = "prompt_score/" + args.prompt + "_" + args.model + "_" + args.size + "_" + args.save_name + ".json"
checkpoint = "chk_violations/" + args.prompt + "_" + args.model + "_" + args.size + "_" + args.save_name+'.txt'
start_index = load_checkpoint(checkpoint)

if start_index!=0 and os.path.exists(prompt_score_file):
    with open(prompt_score_file, 'r') as file:
        prompt_score = json.load(file)
else:
    prompt_score = []

time_lists = []
#len(response_data)
print(f"dataset size:{len(response_data)}")
# for i in range(start_index, len(response_data)):
for i in range(31):
    print(f"i:{i}")
    response=response_data[i]
    prompt_dict={}
    prompt_dict['context']=response['context']
    prompt_dict['text']=response['text']
    prompt_dict['category']=response['category']

    context = prompt_dict['context']
    text    = prompt_dict['text']
    start_time = time.time()
    user_prompt = "Context: " + context + "\n\nText: " + text + "\n\n" + format

    messages = [
        {"role": "assistant", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    inputs = inputs.to(model.device)
    generated_ids = model.generate(inputs, max_new_tokens=1000, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    generated_ids=generated_ids[:,inputs.shape[1]:]
    output_texts = tokenizer.batch_decode(generated_ids)
    end_time = time.time()
    time_lists.append(end_time - start_time)
    prompt_dict['prompt_raw']=output_texts[0]
    # print(f"output_texts:{output_texts[0]}")

    # prompt_dict['prompt_score']=string_to_dict(output_texts[0])
    # prompt_score.append(prompt_dict)

    # with open(prompt_score_file, 'w') as file:
    #     json.dump(prompt_score, file)
    # save_checkpoint(i+1, checkpoint)
print(f"average time: {sum(time_lists) / len(time_lists)}")