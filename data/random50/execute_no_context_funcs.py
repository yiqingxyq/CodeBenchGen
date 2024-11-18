import subprocess
import json
import re 
import os 
import argparse
import time

import numpy as np
from tqdm import tqdm 

CODE_DIR = os.environ.get("docker_CODE_DIR")
CACHE_DIR = os.environ.get("docker_CACHE_DIR")
EXEC_FILE_PREFIX = os.path.join(CACHE_DIR, "execution_test")

import sys 
sys.path.insert(0,CODE_DIR)
from CodeBenchGen.utils import *
from CodeBenchGen.exec_utils import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='cleaned_python_test.json')
    parser.add_argument("--auto_package_installation", default=False, action='store_true')
    parser.add_argument("--save_sh_commands", default=False, action='store_true')
    args = parser.parse_args()
    
    func_list = json.load(open(args.input_file))
    
    code_list = []
    for func_dict in func_list:
        if len(extract_code(func_dict["context"])) != 1 or "main_func" not in func_dict:
            code_list.append("")
            continue 
        
        context = extract_code(func_dict["context"])[0][1]
        context = context + "\n\n" + func_dict["main_func"]
        code_list.append(context)

    os.makedirs(EXEC_FILE_PREFIX, exist_ok=True)
    print(f"Running execution for {len(code_list)} funcitons' context.")


    sh_commands = []
    successful_count = 0
    execution_count = 0
    for code_idx, (code, func_dict) in enumerate(tqdm(zip(code_list, func_list))):
        if not code:
            continue 
        
        exec_file = f"{EXEC_FILE_PREFIX}_{code_idx}.py"
        with open(exec_file, "w") as fout:
            fout.write(code)
        
        # run python code
        res, idx = run_program(f"python {exec_file}", code_idx)
        
        # automatically install packages and re-execute the code
        if args.auto_package_installation:
            installed_list = automatic_package_install(res)
                
        print([code_idx])
        print(res)
        res = ( str(res[0]), res[1].replace('"',"'"), res[2].replace('"',"'"))
        
        execution_count += 1
        func_dict["exec_result"] = res
        
        if res[0] == "success":
            successful_count += 1
    
    print(f"Successfully execute {successful_count}/{execution_count}")
    json.dump(func_list, open(args.input_file.replace(".json", "_exec.json"), "w"), indent=4)
    
    if args.save_sh_commands:
        new_sh_commands = []
        for c in sh_commands:
            if c not in new_sh_commands:
                new_sh_commands.append(c)
        json.dump(new_sh_commands, open("sh_commands.json",'w'), indent=4)