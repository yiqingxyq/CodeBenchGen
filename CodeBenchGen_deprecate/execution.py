import subprocess
import json
import numpy as np
from tqdm import tqdm 
import re 
import os 
import argparse
import time

CACHE_DIR = os.environ.get("docker_CACHE_DIR")
dataset_generation_DIR = os.environ.get("docker_dataset_generation_DIR")
final_dataset_DIR = os.environ.get("docker_final_dataset_DIR")

MAX_VIRTUAL_MEMORY = 4 * 1024 * 1024 * 1024  # 4 GB

ERR_KEYWORDS = ['ERROR', 'Error', 'error']

def limit_virtual_memory(max_virtual_memory):
    # We do a soft limit in order to be able to change the limit later if needed
    return f"ulimit -S -v {max_virtual_memory}"

def direct_return(proc, proc_name, timeout=10):
    try:
        try:
            result, stderr = proc.communicate(timeout=timeout)
            has_err_keywords = False 
            for k in ERR_KEYWORDS:
                if k in stderr.decode("utf-8", errors="replace"):
                    has_err_keywords = True
            
            if proc.returncode or has_err_keywords:
                return "error", result.decode("utf8", errors="replace"), stderr.decode("utf-8", errors="replace")
            else:
                return "success", result.decode("utf8", errors="replace"), stderr.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            c = (
                    "kill `ps aux | grep '"
                    + proc_name
                    + "' | grep -v jupyter | grep -v grep | awk '{print($2)}'`"
            )
            subprocess.run(
                c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return "timeout", "", ""
    except:
        return "error", "", stderr.decode("utf-8", errors="replace")

def run_python_program(script_path, i, timeout=10):
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; python {script_path}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = direct_return(proc, f"python {script_path}", timeout=timeout)
    return res, i


if __name__ == "__main__":
    # example for debug: python execution.py --round 0 --mode "debug"
    # example for the final pass: python execution.py --round 3 --mode final; python execution.py --round 3 --mode final --skip_sh 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--mode", type=str, default='debug', choices=['debug', 'final'])
    parser.add_argument("--skip_sh", action="store_true")
    args = parser.parse_args()
    
    if args.mode == 'debug':
        data_file = os.path.join(dataset_generation_DIR, f"simplified_round{args.round}_python_test.json")
        output_file = os.path.join(dataset_generation_DIR, f"execution_round{args.round}_python_test.json")
        command_file = os.path.join(dataset_generation_DIR, f"execution_commands_round{args.round}_python_test.json")
        
        exec_file_prefix = os.path.join(CACHE_DIR, f"round{args.round}")
        
        code_key = f'simplified_code_w_test_round{args.round}'
        sh_code_key = f'simplified_sh_code_w_test_round{args.round}'
        output_key = f'result_round{args.round}'
        
    elif args.mode == 'final':
        data_file = os.path.join(dataset_generation_DIR, f"test_set_round{args.round}.json")
        output_file = os.path.join(dataset_generation_DIR, f"test_set_execution_round{args.round}.json")
        command_file = os.path.join(dataset_generation_DIR, f"test_set_execution_commands_round{args.round}.json")
        
        exec_file_prefix = os.path.join(CACHE_DIR, 'final')
        
        code_key, sh_code_key, output_key = 'code', 'sh_code', 'result'
        
        
    print(args.mode, code_key, sh_code_key, output_key)
        
    simplified_python = json.load(open(data_file))
    
    if not os.path.exists(os.path.dirname(exec_file_prefix)):
        os.mkdir(os.path.dirname(exec_file_prefix))
    
    sh_commands = []
    execution_count = 0
    for i,p in enumerate(tqdm(simplified_python)):

        if code_key not in p:
            continue 
        
        # run sh code
        if not args.skip_sh and args.mode != 'final' and sh_code_key in p:
            sh_code = p[sh_code_key]
            for line in sh_code.split('\n'):
                # only allow pip install code 
                if 'pip' not in line or line[:6] != 'python':
                    continue 
                print(f"Running sh code: {line}")
                sh_commands.append(line)
                proc = subprocess.Popen(line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable="/bin/bash")
                res = direct_return(proc, line, timeout=30)
                print(res)
                
        
        code = p[code_key]
        exec_file = f"{exec_file_prefix}_{p['idx']}.py"
        with open(exec_file, "w") as fout:
            fout.write(code)
        
        # run python code
        res, idx = run_python_program(exec_file, p['idx'])
        
        # automatically install packages and re-execute the code
        if not args.skip_sh:
            pip_count = 0
            installed_list = []
            while pip_count < 10:
                # install at most 10 packages 
                if res[0] != 'success' and 'No module named' in res[2]:
                    try:
                        pkg_name = res[2].split('No module named ')[1].split("'")[1].split('.')[0]
                        if pkg_name in installed_list:
                            break
                    except:
                        break 
                    
                    pip_count += 1
                    print(f'Installing {pkg_name}')
                    installed_list.append(pkg_name)
                    proc = subprocess.run(f'pip install {pkg_name}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    if proc.returncode != 0:
                        # error when installing packages
                        pip_err_msg = proc.stderr.decode("utf-8", errors="replace")
                        print('Installation failed with error msg:')
                        print(pip_err_msg)
                        break 
                    
                    # no error, run the program again
                    sh_commands.append(f'pip install {pkg_name}')
                    print(f'Running again after installing {pkg_name}')
                    res, idx = run_python_program(exec_file, p['idx'])
                else:
                    break
                
        print([i,idx])
        print(res)
        res = ( str(res[0]), res[1].replace('"',"'"), res[2].replace('"',"'"))
        simplified_python[i][output_key] = res
        
        execution_count += 1

                
    correct_count = 0
    for i,p in enumerate(tqdm(simplified_python)):
        correct_flag = False
        if output_key in p:
            if p[output_key][0] == "success":
                correct_flag = True 
        if correct_flag:
            correct_count += 1
            
    json.dump(simplified_python, open(output_file, 'w'), indent=4)
    
    if not args.skip_sh:
        json.dump(sh_commands, open(command_file, 'w'), indent=4)
    
    print(f"No runtime error: {correct_count}/{execution_count}")