import os
import json 
import numpy as np

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from CodeBenchGen.utils import *

def count_import(code):
    lines = []
    for line in code.split("\n"):
        if line[:6] == "import":
            lines.append(line)
        # else:
        #     if line.strip():
        #         break
            
    return len(lines)


data = json.load(open("data/random50/sandbox_check_random50_debugged.json"))
# data = [x for x in data if "full_script_debug_round1_exec_result" in x and x["full_script_debug_round1_exec_result"][0] == "success"]
data = [x for x in data if x["sandbox_functionality_check"]["answer"] in ["same", "yes"]]

for example in data:
    del example["sandbox_functionality_check"]

json.dump(data, open("data/random50/random50_checked.json", 'w'), indent=4)

token_nums = []
func_token_nums = []
package_counts = []
repos = set()
for example in data:
    # script = example["full_script_debug_round1"]
    script = example["eval_script"]
    token_nums.append( count_code_tokens(script) )
    
    func_name = example["func_name"].split(".")[-1]
    class_name = example["func_name"].split(".")[0] if "." in example["func_name"] else None
    func_start, func_end = get_function_line_idx(script, func_name, class_name=class_name)
    new_func = "\n".join(script.split("\n")[func_start:func_end+1])
    func_token_nums.append( count_code_tokens(new_func) )
    
    package_counts.append(count_import(script)) 
    
    repos.add(example["repo_name"])

print(f"num examples: {len(data)}")
print(max(token_nums), np.mean(token_nums), min(token_nums))
print(max(func_token_nums), np.mean(func_token_nums), min(func_token_nums))
print(f"Package_counts:", max(package_counts), np.mean(package_counts))
print(f"Repo count: {len(repos)}")