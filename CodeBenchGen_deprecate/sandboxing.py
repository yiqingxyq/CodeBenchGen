import json
from tqdm import tqdm 
import argparse

import os
import re
import time
import argparse

import openai

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

openai.api_base = "https://api.openai.com/v1"
openai.api_key = os.environ.get("OPENAI_API_KEY")
API_NAME = "gpt-4-turbo-2024-04-09"
# API_NAME = "gpt-3.5-turbo-0125"

prompt_template = "Rewrite the following piece of python code so that we can test the implementation of the __FUNCTION_NAME__ function by directly executing this piece of code. You should keep the input code as much as possible and output every token of the content with no abbreviation. You can add code that is necessary. The code will be run in an isolated environment with no access to GPU or external API keys. \n\nKeep the function name and input arguments of the __FUNCTION_NAME__ function as the same. Output one complete piece of code in the format of ```python ```.\n\nIf necessary, output `pip` or `python -m` commands in another code block in the format of ```bash ```.\n\n__CODE__\n\n"


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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='cleaned_python_test.json')
    parser.add_argument("--skip_prefiltering", default=False, action='store_true')
    args = parser.parse_args()
    
    input_data_file = os.path.join(dataset_generation_DIR, args.input_file)
    output_data_file = os.path.join(dataset_generation_DIR, 'simplified_round0_python_test.json')
    
    if os.path.exists(output_data_file):
        examples = json.load(open(output_data_file, 'r'))
    else:
        examples = json.load(open(input_data_file, 'r'))
    
    compute_count = 0
    for idx,p in enumerate(tqdm(examples)):
        if 'context' not in p:
            continue
        
        if not args.skip_prefiltering:
            if 'has_banned_keywords' in p and p['has_banned_keywords']:
                continue
        
        if 'simplified_code_round0' in p:
            continue
                
        prompt = [
            {
                "role": "user",
                "content": prompt_template.replace('__FUNCTION_NAME__', p['func_name']).replace('__CODE__', p['context'])
            }
        ]
        
        response = call_openai_completion(
            model=API_NAME, messages=prompt, max_tokens=4096,
        )
        
        try:
            response_text = response['choices'][0]['message']['content']
            examples[idx]['simplified_output_round0'] = response_text
            
            print('-'*30, f"func_name: {p['func_name']}", '-'*30)
            print('-'*30, 'Output', '-'*30)
            print(response_text)
            print('='*60)
            compute_count += 1
        except:
            print('Wrong format')
            pass 
        
        if compute_count % 10 == 0 and compute_count > 0:
            print(f'Saving {compute_count}/{(idx+1)} examples to file..')
            json.dump(examples, open(output_data_file, 'w'), indent=4)
            
    json.dump(examples, open(output_data_file, 'w'), indent=4)
            