import torch
import json
import os
from llm_loading import load_model_and_tokenizer
import time
import argparse
parser = argparse.ArgumentParser(description="generate trust scores")
parser.add_argument('-response_file', type=str, default = "responses/mmlubio_correct_circle.json")
parser.add_argument('-save_name', type=str, default = "mmlubio_correct")
parser.add_argument('-model', type=str, default='llama3')
parser.add_argument('-size', type=str, default='8B')
parser.add_argument('-gpu', type=str, default='3')

args = parser.parse_args()

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

perplexity_file = "perplexity/" + args.model + "_" + args.size + "_" + args.save_name + ".json"

perplexity_score = []

time_lists = []
# len(response_data)
for i in range(len(response_data)):
    print(f"i:{i}")
    response=response_data[i]
    perplexity_dict={}
    perplexity_dict['context']=response['context']
    perplexity_dict['text']=response['text']
    perplexity_dict['category']=response['category']

    context = perplexity_dict['context']
    text    = perplexity_dict['text']

    start_time = time.time()
    full = context + text # concatenate
    enc  = tokenizer(full, return_tensors="pt").to(model.device)
    ids  = enc["input_ids"][0]

    # Determine split point
    split = len(tokenizer(context)["input_ids"]) #tokens in context

    # Build labels: mask context tokens with ‑100
    labels = ids.clone()
    labels[:split] = -100 # ‑100 ⇒ ignored by loss
    enc["labels"] = labels.unsqueeze(0)

    with torch.no_grad():
        out   = model(**enc, output_hidden_states=False)
        loss  = out.loss # average NLL over *unmasked* tokens
        ppl   = torch.exp(loss).item()
    end_time = time.time()
    time_lists.append(end_time - start_time)

    perplexity_dict['perplexity_score']=ppl
    perplexity_score.append(perplexity_dict)

    # with open(perplexity_file, 'w') as file:
    #     json.dump(perplexity_score, file)
print(f"average time: {sum(time_lists) / len(time_lists)}")