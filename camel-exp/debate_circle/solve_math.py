import os
import json
from glob import glob

# Prompt for the API key securely
os.environ["OPENAI_API_KEY"] = "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"

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, RoleType
from camel.configs import ChatGPTConfig

import argparse

parser = argparse.ArgumentParser(description="autogen debate test")
parser.add_argument('-save_dir', type=str, default="solve_logs")
parser.add_argument('-dataset',required=False, type=str, help="dataset name")
parser.add_argument('-prompt', type=str, default = 'persuasive-3', help='prompt for malicious agent')
parser.add_argument('-model', type=str, default='gpt4o', help='LLM')
args = parser.parse_args()

#support functions
def split_text(text, delimiter):
    """
    Splits the input text into two parts based on the specified delimiter.
    
    Args:
        text (str): The input text to be split.
        delimiter (str): The keyword or phrase to split the text.

    Returns:
        tuple: A tuple containing two parts of the text (before and after the delimiter).
    """
    if delimiter in text:
        # Split the text into two parts based on the delimiter
        before, after = text.split(delimiter, 1)
        # Include the delimiter in the second part
        after = delimiter + after
        return before.strip(), after.strip()
    else:
        # If the delimiter is not found, return the whole text as the first part
        return text.strip(), None
    
def transform_list_to_string(dict_list):
    """
    Transforms a list of dictionaries into a single formatted string.

    Args:
        dict_list (list): A list of dictionaries where each dictionary has one key-value pair.

    Returns:
        str: A formatted string in the form "key1:value1, key2:value2, ..."
    """
    # Use a list comprehension to extract "key:value" pairs from each dictionary
    key_value_pairs = [f"{key}:{value}" for d in dict_list for key, value in d.items()]
    
    # Join the pairs with a comma and space
    return ", ".join(key_value_pairs)

def format_multi(problem):
    question = problem['question']
    option_list = problem['options']
    formatted_text = "Options:\n"
    for index, option in enumerate(option_list, start=1):
        # Adding a letter label (A, B, C, D) before each option
        formatted_text += f"{chr(64 + index)}) {option}\n"
    options = formatted_text.strip()
    return question, options

def load_checkpoint(chk_file):
    try:
        with open(chk_file, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0  # no checkpoint file, start from beginning

def save_checkpoint(index, chk_file):
    with open(chk_file, "w") as f:
        f.write(str(index))
        
with open("prompt.json", 'r') as file:
    prompts = json.load(file)

# load dataset
if args.dataset == 'gsm8k':
	data = []
	with open("../../data/gsm8k_test.jsonl", "r", encoding="utf-8") as f:
		for line in f:
			data.append(json.loads(line.strip()))
	question_name = 'question'
	answer_name = 'answer'
elif args.dataset == 'math':
    data = []
    dataset_dir = '../../data/MATH_test'
    for subdir in os.listdir(dataset_dir):
        subdir_path = os.path.join(dataset_dir, subdir)
        if os.path.isdir(subdir_path):
            json_files = glob(os.path.join(subdir_path, "*.json"))
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data.append(json.load(f))
    question_name = 'problem'
    answer_name = 'solution'

# Define the model, here in this case we use gpt-4o-mini
if args.model == 'gpt4o':
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig().as_dict(), # [Optional] the config for model
    )
elif args.model == 'gpt4o-mini':
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig().as_dict(), # [Optional] the config for model
    )
elif args.model == 'llama3-8b':
   model = ModelFactory.create(
       model_platform=ModelPlatformType.VLLM,
       model_type="llama-8b",  # Matches the ID returned by vLLM
       url="http://localhost:8080/v1",  # Ensure this matches your vLLM URL
       model_config_dict={"temperature": 0.3},  # Optional configurations
)



# Create solver agents
solver1 = ChatAgent(
    system_message=prompts[args.prompt],
    model=model,
    )
solver2 = ChatAgent(
    system_message=prompts['original-trust'],
    model=model,
    )
# solver3 = ChatAgent(
#     system_message=prompts['original-trust'],
#     model=model,
#     )

# Create judge agent
judge = ChatAgent(
    system_message=prompts['judge'],
    model=model,
    )

def solve_problem(problem_description, iter_limits=5):
    
    delimiter = "\n\nTrust rate"

    # Start discussion between solvers
    discussion = []
    message_pool=[]

    # Solver 1 starts
    message_from_solver1 = solver1.step(BaseMessage(
        role_name="Solver_1",
        role_type=RoleType.USER,
        meta_dict={},  # Empty metadata dictionary
        content=problem_description
    ))


    discussion.append({"Solver_1": message_from_solver1.msgs[0].content})
    message_pool.append({"Solver_1": message_from_solver1.msgs[0].content})

    # Solver 2 responds
    message_from_solver2 = solver2.step(BaseMessage(
        role_name="Solver_2",
        role_type=RoleType.ASSISTANT,
        meta_dict={},
        content=transform_list_to_string(message_pool)
    ))
    discussion.append({"Solver_2": message_from_solver2.msgs[0].content})
    part1, _ = split_text(message_from_solver2.msgs[0].content, delimiter)
    message_pool.append({"Solver_2": part1})
    
    
    # Solver 3 responds
    # message_from_solver3 = solver3.step(BaseMessage(
    #     role_name="Solver_3",
    #     role_type=RoleType.ASSISTANT,
    #     meta_dict={},
    #     content=transform_list_to_string(message_pool)
    # ))
    # discussion.append({"Solver_3": message_from_solver3.msgs[0].content})
    # part1, _ = split_text(message_from_solver3.msgs[0].content, delimiter)
    # message_pool.append({"Solver_3": part1})

    
    
    for _ in range(iter_limits):
        # Continue discussion (if needed)
        message_from_solver1 = solver1.step(BaseMessage(
            role_name="Solver_1",
            role_type=RoleType.ASSISTANT,
            meta_dict={},
            content=transform_list_to_string(message_pool)
        ))
        discussion.append({"Solver_1": message_from_solver1.msgs[0].content})
        message_pool.append({"Solver_1": message_from_solver1.msgs[0].content})

        message_from_solver2 = solver2.step(BaseMessage(
            role_name="Solver_2",
            role_type=RoleType.ASSISTANT,
            meta_dict={},
            content=transform_list_to_string(message_pool)
        ))
        discussion.append({"Solver_2": message_from_solver2.msgs[0].content})
        part1, _ = split_text(message_from_solver2.msgs[0].content, delimiter)
        message_pool.append({"Solver_2": part1})
        
        # Solver 3 responds
        # message_from_solver3 = solver3.step(BaseMessage(
        #     role_name="Solver_3",
        #     role_type=RoleType.ASSISTANT,
        #     meta_dict={},
        #     content=transform_list_to_string(message_pool)
        # ))
        # discussion.append({"Solver_3": message_from_solver3.msgs[0].content})
        # part1, _ = split_text(message_from_solver3.msgs[0].content, delimiter)
        # message_pool.append({"Solver_3": part1})
        

    # Pass discussion to judge
    discussion_text = "\n".join([f"{key}: {value}" for turn in message_pool for key, value in turn.items()])
    
    judge_message = f"""
        Based on the following discussion between Solvers, please provide a final conclusion with the best answer for the problem:
        
        {discussion_text}
        
        Your response should follow this format:
        FINAL ANSWER:
        EXPLANATION: [Your detailed explanation here]
        """


    judge_response = judge.step(BaseMessage(
        role_name="Judge",
        role_type=RoleType.USER,
        meta_dict={},
        content=judge_message
    ))
    discussion.append({"Judge": judge_response.msgs[0].content})
    return discussion, message_pool


save_file = os.path.join(args.save_dir, args.model+'_'+args.dataset+'_'+args.prompt+'_all3.json')
checkpoint = "checkpoint/solve" +'_'+args.model+'_'+args.dataset+'_'+args.prompt+'_all3.txt'
start_index = load_checkpoint(checkpoint)

if start_index!=0 and os.path.exists(save_file):
    with open(save_file, 'r') as file:
        discussions = json.load(file)
else:
    discussions = []
    
# if 'human' in args.dataset:
#     end_idx=len(data)
# else:
#     end_idx=103
for i in range(start_index,20):
    try:
        if 'mmlu' in args.dataset:
            question, options = format_multi(data[i])
            idea = question + "\n" +options
        else:
            idea = data[i][question_name]
        print(f"Problem:{idea}")
        discussion,_ = solve_problem(idea, iter_limits=2)
        # print(f"discussion:{discussion}")
        # print(f"judge_conclusion:{judge_conclusion}")
        # exit(0)
        discussion.append(data[i])
        discussions.append(discussion)
        with open(save_file, 'w') as file:
            json.dump(discussions, file, indent=4)
        save_checkpoint(i+1, checkpoint)
    except Exception as e:
        print(f"Error processing idea {i}: {e}")
        continue
