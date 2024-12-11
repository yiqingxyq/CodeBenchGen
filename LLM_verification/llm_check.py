import json
from tqdm import tqdm 
import os
import re
import time
import argparse

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
final_dataset_DIR = os.environ.get("final_dataset_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from CodeBenchGen.utils import call_openai_completion
from LLM_verification.prompts import sandbox_check_prompt_template, test_check_prompt_template, instruction_check_prompt_template

API_NAME = "gpt-4o"


def get_index(text, string):
    if string not in text:
        return 

def extract_answer(text):
    if not text or "REASONING:" not in text or "ANSWER:" not in text:
        return None, None

    reasoning = text.split("ANSWER")[0].strip()
    answer_text = text.split("ANSWER")[-1].lower().strip()
    
    label2pos = {label:answer_text.find(label) for label in ["same", "yes", "no", "minor", "major"]} 
    # label2pos = {label:answer_text.find(label) for label in ["same", "minor", "major"]}
    label2pos = {k:v for k,v in label2pos.items() if v>0}
    answer = min(label2pos, key=lambda k:label2pos[k]) if label2pos else None
    
    return reasoning, answer


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["ExecCSN", "random50"])
    parser.add_argument("--input_file", type=str, default='test_set_final_round3.json')
    parser.add_argument("--mode", type=str, choices=["sandbox", "test", "instruction"], required=True)
    args = parser.parse_args()
    
    if args.dataset == "random50":
        input_file = os.path.join(dataset_generation_DIR, args.input_file)
        output_file = os.path.join(dataset_generation_DIR, f"{args.mode}_check_{args.input_file}")
    else:
        input_file = os.path.join(final_dataset_DIR, args.input_file)
        output_file = os.path.join(final_dataset_DIR, f"{args.mode}_check_{args.input_file}")
    
    examples = json.load(open(input_file, 'r'))
    
    compute_count = 0
    successful_count = 0
    almost_successful_count = 0
    same_count = 0
    for idx,func_dict in enumerate(tqdm(examples)):
        
        if args.dataset == "random50":
            code = func_dict["eval_script"]
        else:
            code = func_dict["code"]
            instruction = func_dict["revised_instruction"]
            
        orig_func = func_dict["orig_func"]
        func_name = func_dict["func_name"]
        test_func_name = "test_" + func_name.split(".")[-1]
        
        if args.mode == "sandbox":
            prompt = sandbox_check_prompt_template.replace('__FUNCTION_NAME__', func_name).replace('__ORIG_FUNC__', orig_func).replace('__NEW_CODE__', code)
            saved_key = "sandbox_functionality_check"
        elif args.mode == "test":
            prompt = test_check_prompt_template.replace('__FUNCTION_NAME__', func_name).replace('__TEST_FUNCTION_NAME__', test_func_name).replace('__CODE__', code)
            saved_key = "test_correctness_check"
        elif args.mode == "instruction":
            prompt = instruction_check_prompt_template.replace('__FUNCTION_NAME__', func_name).replace('__INSTRUCTION__', instruction).replace('__CODE__', code)
            saved_key = "instruction_clarity_check"
        
        response_text = call_openai_completion(
            model=API_NAME, messages=[{"role": "user", "content": prompt}], max_tokens=8192,
        )
        
        reasoning, answer = extract_answer(response_text)
        if answer:
            func_dict[saved_key] = {"reasoning": reasoning, "answer": answer}
            
            same_count += answer in ["same"]
            successful_count += answer in ["same", "yes"]
            almost_successful_count += answer in ["same", "yes", "minor"]
            compute_count += 1
        else:
            print("Generation error! Cannot extract answer")
            print(response_text)
        
        if compute_count % 10 == 0 and compute_count > 0:
            print(f'Saving {compute_count}/{(idx+1)} examples to file..')
            print(f"{same_count}/{successful_count}/{almost_successful_count}/{compute_count} examples are successful")
            json.dump(examples, open(output_file, 'w'), indent=4)
            
    print(f"{same_count}/{successful_count}/{almost_successful_count}/{compute_count} examples are successful")
    json.dump(examples, open(output_file, 'w'), indent=4)