import openai
import requests
import os
from llm_loading import load_model_and_tokenizer
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

parser = argparse.ArgumentParser(description="autogen debate test")
parser.add_argument('-model_type', type=str, default="llama3")
parser.add_argument('-model_variant',type=str, default="8B")
args = parser.parse_args()


model, tokenizer = load_model_and_tokenizer(args.model_type, args.model_variant)



