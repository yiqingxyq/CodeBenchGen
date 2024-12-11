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
docker_dataset_generation_DIR = os.environ.get("docker_dataset_generation_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from CodeBenchGen.utils import *
from CodeBenchGen.utils_exec import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='cleaned_python_test.json')
    parser.add_argument("--auto_package_installation", default=False, action='store_true')
    parser.add_argument("--save_sh_commands", default=False, action='store_true')
    args = parser.parse_args()
    
    # copy repos to CACHE_DIR
    repos_path = os.path.join(docker_dataset_generation_DIR, "repos")
    result = subprocess.run(f"cp -r {repos_path} {CACHE_DIR}", shell=True, capture_output=True, text=True)
    print(f"Finished copying all repos to {CACHE_DIR}")
    
    func_list = json.load(open(args.input_file))
    
    code_list = []
    for func_dict in func_list:
        file_name = func_dict["func_path"].split("/")[-1].replace(".py","")
        func_or_class_name = func_dict["func_name"].split(".")[0]
        import_code = f"from {file_name} import {func_or_class_name}"
        
        if "main_func" not in func_dict:
            code_list.append(import_code)
        else:
            code_list.append(import_code + "\n\n" + func_dict["main_func"])

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Running execution for {len(code_list)} funcitons' context.")


    successful_count = 0
    execution_count = 0
    for code_idx, (code, func_dict) in enumerate(tqdm(zip(code_list, func_list))):
        if not code:
            continue 
        
        repo_dir = os.path.join(os.path.join(CACHE_DIR, "repos"), func_dict["repo_name"])
        exec_file_dir = os.path.join(repo_dir, os.path.dirname(func_dict["func_path"]) )
        
        exec_file = f"codebenchgen_execution_test_{code_idx}.py"
        with open(os.path.join(exec_file_dir, exec_file), "w") as fout:
            fout.write(code)
        
        # run python code
        exec_cmd = f"cd {exec_file_dir} ; python {exec_file}"
        res, idx = run_program(exec_cmd, code_idx)
        
        # automatically install packages and re-execute the code
        if args.auto_package_installation:
            sh_commands = automatic_package_install(exec_file, code_idx, res)
                
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