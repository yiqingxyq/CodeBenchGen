import json
from tqdm import tqdm 
import os
import re
import time
import argparse
import ast 

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
final_dataset_DIR = os.environ.get("final_dataset_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from CodeBenchGen.utils import *

def extract_docstrings(source_code: str):
    # DOSCTRING_PATTERN = r'(""".*?"""|\'\'\'.*?\'\'\')'
    DOSCTRING_PATTERN = r'(\n[\s\t]*""".*?"""|\n[\s\t]*\'\'\'.*?\'\'\')'
    match = re.findall(DOSCTRING_PATTERN, source_code, flags=re.DOTALL)
    return match


def remove_docstring(func_code):
    docstrings = extract_docstrings(func_code)
    clean_func_code = func_code
    for m in docstrings:
        clean_func_code = clean_func_code.replace(m, "")
    
    return clean_func_code


def are_functions_identical(func1: str, func2: str) -> bool:
    try:
        # Parse the strings into AST nodes
        tree1 = ast.parse(func1)
        tree2 = ast.parse(func2)

        # Get a normalized representation of the AST
        normalized_tree1 = ast.dump(tree1, annotate_fields=False, include_attributes=False)
        normalized_tree2 = ast.dump(tree2, annotate_fields=False, include_attributes=False)

        # Compare the normalized ASTs
        return normalized_tree1 == normalized_tree2
    except Exception as e:
        # Handle parsing errors
        # print(f"Error parsing functions: {e}")
        return None

def are_strings_identical(func1: str, func2: str):
    return "".join(remove_docstring(func1).split()) ==  "".join(remove_docstring(func2).split())

data = json.load(open("../ExecCSN_dataset/test_set_final_round3.json"))

count = 0
compute_count = 0
for example in data:
    func_name = example["func_name"].split(".")[-1]
    class_name = example["func_name"].split(".")[0] if "." in example["func_name"] else None
    func_start, func_end = get_function_line_idx(example["code"], func_name, class_name=class_name)
    if not func_start:
        continue 
    
    new_func = "\n".join(example["code"].split("\n")[func_start:func_end+1])
    orig_func = example["orig_func"]
    
    result = are_functions_identical(new_func, orig_func)
    if result is not None:
        count += result
    else:
        count += are_strings_identical(new_func, orig_func)
    compute_count += 1

print(f"{count}/{compute_count}={count/compute_count}")