import json
from tqdm import tqdm 
import os
import re
import time
import argparse
import numpy as np

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from CodeBenchGen.utils import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="cleaned_python_test_debug_round2_exec.json")
    parser.add_argument("--output_file", type=str, default="random50_debugged.json")
    parser.add_argument("--script_key", type=str, default="full_script_debug_round2")
    args = parser.parse_args()

    exec_result_key = f"{args.script_key}_exec_result"
    sh_key = f"{args.script_key}_sh"

    data = json.load(open( os.path.join(dataset_generation_DIR, args.input_file) ))
    data = [x for x in data if exec_result_key in x and x[exec_result_key][0] == "success"]

    clean_data = []
    for example in data:
        new_example = {k:v for k,v in example.items() 
            if k in ["func_name", "idx", "repo_name", "func_path"]
        }
        new_example["orig_func"] = example["func_code"]
        new_example["orig_context"] = example["context"]
        new_example["eval_script"] = example[args.script_key]
        if sh_key in example:
            new_example["eval_script_sh"] = example[sh_key]
        clean_data.append(new_example)
        
    print(f"Saving {len(clean_data)} examples..")
    json.dump(clean_data, open(os.path.join(dataset_generation_DIR, args.output_file), "w"), indent=4)