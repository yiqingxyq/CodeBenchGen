import json
import numpy as np
from tqdm import tqdm 
import argparse

import os
import re
import sys
import time
import argparse

import openai
from src.ts_utils import ts_parser, find_func_node

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

openai.api_base = "https://api.openai.com/v1"
openai.api_key = os.environ.get("OPENAI_API_KEY_OPENAI")
API_NAME = "gpt-4-turbo-2024-04-09"

prompt_template = "You are preparing an interview for software engineers. The interviewees are going to complete the __FUNCTION_NAME__ function. Write a clear instruction describing this function in around 45 words, which includes the functionality, the input arguments (if any), and the outputs (if any). Do not reveal test cases. Generate the instruction with the following format:\n```\nFunctionality: ...\nInputs: ...\nOutputs: ...\n```.\n\n__CODE__\n\n"


INST_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"
def extract_inst(text: str, pattern: str = INST_BLOCK_PATTERN):
    match = re.findall(pattern, text, flags=re.DOTALL) # [('', 'Python Code'), ('bash', 'ssh test')]
    if not match:
        return None
    else:
        return match[0][1]

def call_openai_completion(**kwargs):
    is_ok = False
    retry_count = 0
    while not is_ok:
        retry_count += 1
        try:
            response = openai.ChatCompletion.create(**kwargs)
            is_ok = True
        except openai.error.RateLimitError as error:
            if retry_count <= 20:
                print(f"OpenAI API retry for {retry_count} times ({error})")
                time.sleep(5)
                continue
            else:
                print('Retry limit reached.')
                return {}
        except Exception as error:
            print(error)
            return {}
    return response
        
def fill_in_gt_answer(code, func_name, target):
    tree = ts_parser.parse(bytes(code, 'utf-8'))
    func_node = find_func_node(tree.root_node, func_name)
    func_text = '\n'.join(code.split('\n')[func_node.start_point[0]:func_node.end_point[0]+1])
    
    masked_content = func_text.split('\n')[-1]
    unmasked_func_text = func_text.replace(masked_content, target)
    unmasked_code = code.replace(func_text, unmasked_func_text)
    
    return unmasked_code
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_round", type=int, default=3)
    args = parser.parse_args()
    
    data_file = os.path.join(dataset_generation_DIR, f"test_set_final_round{args.max_round}.json")
    examples = json.load(open(data_file, 'r'))
    masked_code_key = "masked_method_no_test"
    
    compute_count = 0
    for idx,p in enumerate(tqdm(examples)):
        if masked_code_key not in p:
            continue
        
        if 'revised_instruction' in p:
            continue
        
        code_no_test_main = fill_in_gt_answer(p[masked_code_key], p['func_name'], p['gt_answer'])
                
        prompt = [
            {
                "role": "user",
                "content": prompt_template.replace('__FUNCTION_NAME__', p['func_name']).replace('__CODE__', code_no_test_main)
            }
        ]

        response = call_openai_completion(
            model=API_NAME, messages=prompt, max_tokens=2000,
        )
    
        try:
            response_text = response['choices'][0]['message']['content']
            instruction = extract_inst(response_text)
            
            if not instruction:
                continue
            
            p['revised_instruction'] = instruction
            print('-'*30, 'Instruction', '-'*30)
            print(instruction)
            print('='*60)
            compute_count += 1
        except:
            print('Wrong format')
            pass 
        
        if compute_count % 10 == 0 and compute_count > 0:
            print(f'Saving {compute_count}/{(idx+1)} examples to file..')
            json.dump(examples, open(data_file, 'w'), indent=4)
            
    json.dump(examples, open(data_file, 'w'), indent=4)