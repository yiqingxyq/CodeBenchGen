import subprocess
import json
import re 
import os 
import argparse
import datetime
import time

import numpy as np
from tqdm import tqdm 

CODE_DIR = os.environ.get("docker_CODE_DIR")
CACHE_DIR = os.environ.get("docker_CACHE_DIR")
EXEC_FILE_PREFIX = os.path.join(CACHE_DIR, "execution_test")

import sys 
sys.path.insert(0,CODE_DIR)
from CodeBenchGen.utils import *
from CodeBenchGen.utils_exec import *

def insert_gt_implementation(script, full_func_name):
    class_name = full_func_name.split(".")[0] if "." in full_func_name else None
    func_name = full_func_name.split(".")[-1]
    new_implementation = extract_and_rename_new_implementation(script, func_name, class_name=class_name)
    new_script = insert_new_implementation(new_implementation, script, func_name, class_name=class_name)
    return new_script

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='cleaned_python_test_tests.json')
    parser.add_argument("--script_key", type=str, required=True) #e.g., full_script
    parser.add_argument("--run_sh_commands", default=False, action='store_true')
    args = parser.parse_args()
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    func_list = json.load(open(args.input_file))
    
    sh_commands = []
    successful_count = 0
    execution_count = 0
    extraction_error_count = 0
    for code_idx, func_dict in enumerate(tqdm(func_list)):
        if args.script_key not in func_dict:
            continue 
        
        code = insert_gt_implementation(func_dict[args.script_key], func_dict["func_name"])
        if not code:
            extraction_error_count += 1
            continue
        
        exec_file = f"{EXEC_FILE_PREFIX}_{code_idx}.py"
        with open(exec_file, "w") as fout:
            fout.write(code)
        
        # run python code
        res, idx = run_program(f"python {exec_file}", code_idx)
        
        # automatically install packages and re-execute the code
        if args.run_sh_commands:
            installed_list = automatic_package_install(f"python {exec_file}", code_idx, res)
            sh_commands.extend(installed_list)

            sh_script_key = f"{args.script_key}_sh"
            if sh_script_key in func_dict:
                for cmd in func_dict[sh_script_key].split("\n"):
                    if cmd:
                        run_program(cmd, code_idx)
                        sh_commands.append(cmd)
                        
                
        print([code_idx])
        print(res)
        res = ( str(res[0]), res[1].replace('"',"'"), res[2].replace('"',"'"))
        
        execution_count += 1
        func_dict[f"{args.script_key}_exec_result"] = res
        
        if res[0] == "success":
            successful_count += 1
    
    print(f"Successfully execute: {successful_count}/{execution_count} extractino error count: {extraction_error_count}")
    json.dump(func_list, open(args.input_file.replace(".json", "_exec.json"), "w"), indent=4)
    
    if args.run_sh_commands:
        new_sh_commands = []
        for c in sh_commands:
            if c not in new_sh_commands:
                new_sh_commands.append(c)
                
        sh_cmd_file = args.input_file.replace(".json", "sh_commands.json" + str(datetime.datetime.now()).replace(" ","."))
        json.dump(new_sh_commands, open("sh_commands.json",'w'), indent=4)