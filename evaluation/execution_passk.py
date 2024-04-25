"""
Execution-based evaluation of LLM-generated code.

As this requires execution of arbitrary generated code, we HIGHLY recommend 
running this file from a secure, sandboxed environment (i.e. a container).

We use a highly parallel evaluation framework similar to that of
bigcode-evaluation-harness.
https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/custom_metrics/code_eval.py#L129

We note that executing codes in parallel may cause subtle bugs related
to stateful operations, such as filesystem reads and writes. However, these
cases constitute a small proportion of examples and executing in parallel
drastically increases the speed of execution evaluation.
"""

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from src.ts_utils import ts_parser, find_func_node

CACHE_DIR = os.environ.get("docker_CACHE_DIR")
EXEC_DIR = os.path.join(CACHE_DIR, 'eval')

MAX_VIRTUAL_MEMORY = 4 * 1024 * 1024 * 1024  # 4 GB

ERR_KEYWORDS = ["ERROR", "Error", "error"]


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
                return (
                    "error",
                    result.decode("utf8", errors="replace"),
                    stderr.decode("utf-8", errors="replace"),
                )
            else:
                return (
                    "success",
                    result.decode("utf8", errors="replace"),
                    stderr.decode("utf-8", errors="replace"),
                )
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


def run_python_program(script_path, logfile, i, timeout=10):
    thread = threading.currentThread()
    thread_name = thread.name
    logfile.write(f"{thread_name} | {script_path} | {i}\n")
    logfile.flush()
    executable = sys.executable
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; {executable} {script_path}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = direct_return(proc, f"{executable} {script_path}", timeout=timeout)
    return res, i


def get_function(code, func_name):
    """
    Extract function from surrounding code
    """
    try:
        # func_name = func_name.split('.')[-1]
        tree = ts_parser.parse(bytes(code, "utf-8"))
        fnode = find_func_node(tree.root_node, func_name, duplicate_ok=True)
        return fnode.text.decode("utf-8")
    except Exception as e:
        if "." in func_name:
            func_name = func_name.split(".")[-1]
            return get_function(code, func_name)
        # print(f"[{func_name}] Failed to parse function:", e)
        return ""


# Taken from https://github.com/openai/human-eval/blob/312c5e5532f0e0470bf47f77a6243e02a61da530/human_eval/evaluation.py#L13
def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)
    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


# May populate this with example indices as needed
BANNED_EXAMPLES = []

# A list of potentially dangerous commands. In the interest of safety, this list
# is a bit overly conservative.
DANGEROUS_COMMANDS = [
    "os.kill",
    "terminate",
    "subprocess.call(['kill',",
    "subprocess.call(['rm',",
    "subprocess.call(['rmdir',",
    'subprocess.call(["kill",',
    'subprocess.call(["rm",',
    'subprocess.call(["rmdir",',
    "sys.exit",
    "os.unlink",
    ".unlink",
    ".rmdir",
    "os.remove",
    "os.removedirs",
    "os.rmdir",
    "os.system",
    "rmtree",
    "send2trash",
    # I/O operations
    "open(",
    ".read",
    ".write",
    ".load",
    ".dump",
    "shutil.",
    "glob.",
    "os.path.",
    "os.remove(",
    "os.rename(",
    "os.rmdir(",
    "os.mkdir(",
    "os.makedirs(",
    "os.listdir(",
    ".readlines(",
    ".writelines(",
    ".seek(",
    ".tell(",
    "pickle.",
    "json.load(",
    "json.dump(",
    "csv.reader(",
    "csv.writer(",
    "tempfile.",
    ".flush(",
    "socket.",
    "requests.get(",
    "urllib.request.urlopen(",
    ".readinto(",
    "mmap.",
]


def get_passk_scores(examples, code_key, output_key):
    n_correct = []
    n_samples = []
    for i, p in enumerate(examples):
        if output_key not in p:
            n_correct.append(0)
            n_samples.append(len(p[code_key]))
            continue
        exec_results = p[output_key]
        correct = [r[0] == "success" for r in exec_results]
        n_correct.append(sum(correct))
        n_samples.append(len(correct))
    max_samples = max(n_samples)
    all_passk = []
    for k in range(1, max_samples + 1):
        passk_per_sample = estimate_pass_at_k(n_samples, n_correct, k)
        passk = sum(passk_per_sample) / len(passk_per_sample)
        all_passk.append(passk)
    return all_passk


def main(args):
    hypo_file = args.hypo_file
    code_key = args.code_key
    output_key = f"exec_{code_key}"
    full_code_key = f"full_{code_key}"

    output_file = args.output_file or hypo_file

    with open(hypo_file) as f:
        simplified_python = json.load(f)

    if args.end == -1:
        args.end = len(simplified_python)
    start = args.start
    end = args.end

    print(code_key, hypo_file, start, end)

    allowed_range = {i for i in range(start, end) if i not in BANNED_EXAMPLES}

    args.n_samples = len(simplified_python[0][code_key])
    print(f"Detected n_samples = {args.n_samples}")

    exec_file_prefix = os.path.join(EXEC_DIR, f"{args.code_gen_model}_{args.code_key}")

    print("Loading data from", args.data_file)
    with open(args.data_file) as f_data:
        original_data = json.load(f_data)

    results = []
    for idx, (example, output) in enumerate(zip(original_data, simplified_python)):
        if idx not in allowed_range:
            continue
        func_name = example["func_name"]
        code = []
        # if code_key not in output:
        #     output[code_key] = output['code_0']
        n_failed = 0
        for c in output[code_key]:
            c = get_function(c, func_name)
            if c == "":
                n_failed += 1
                code.append(c)
                continue
            if "\n" not in c:
                continue
            first_line, rest = c.split("\n", maxsplit=1)
            if not first_line.startswith("def") and not first_line.startswith("async"):
                first_line, rest = rest.split("\n", maxsplit=1)
                c = first_line + "\n" + rest
            if "." in func_name:
                if func_name in first_line:
                    first_line = first_line.replace(func_name, func_name.split(".")[-1])
                elif func_name.replace(".", "_") in first_line:
                    first_line = first_line.replace(
                        func_name.replace(".", "_"), func_name.split(".")[-1]
                    )
                c = first_line + "\n" + rest
            code.append(c)
        if n_failed > 0:
            print(
                f"Failed to extract function for {n_failed}/{args.n_samples} for {func_name}"
            )

        # TODO: check if spaces is ever not 4
        spaces = 4
        if "." in example["func_name"]:
            code = [
                "\n".join(spaces * " " + line for line in c.split("\n")).lstrip()
                for c in code
            ]

        full_generated_code = [
            example["masked_code"].replace("__MASK__", c) for c in code
        ]

        output[full_code_key] = full_generated_code

    if not os.path.exists(EXEC_DIR):
        os.mkdir(EXEC_DIR)

    logfile = open("logfile.txt", "a+")

    # Launch a thread pool to execute multiple codes at once
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        job_args = []

        pbar = tqdm(range(len(allowed_range) * args.n_samples), desc="Writing files")

        # first, write files and create list of jobs to run
        for example_id, example in enumerate(simplified_python):
            if example_id not in allowed_range:
                continue
            assert full_code_key in example
            code_samples = example[full_code_key]
            for code_idx, code in enumerate(code_samples):
                if code == "":
                    continue

                for dangerous_cmd in DANGEROUS_COMMANDS:
                    if dangerous_cmd in code:
                        continue

                exec_file = f"{exec_file_prefix}_{example_id}_{code_idx}.py"
                with open(exec_file, "w") as fout:
                    fout.write(code)

                job_args.append((exec_file, logfile, (example_id, code_idx)))

                pbar.update(1)

        pbar = tqdm(range(len(job_args)), desc="Submitting jobs")

        # optionally, shuffle jobs in case it helps prevent blockages from a single example
        # random.shuffle(job_args)

        # submit all jobs to the thread pool
        for func_args in job_args:
            future = executor.submit(run_python_program, *func_args)
            futures.append(future)
            pbar.update(1)

        pbar = tqdm(total=len(futures), desc="Getting results")

        # collect all execution results
        for future in as_completed(futures):
            sample_result, (example_id, code_idx) = future.result()
            example = simplified_python[example_id]
            if output_key not in example:
                example[output_key] = [
                    ("error", None, None) for _ in range(args.n_samples)
                ]

            example[output_key][code_idx] = (
                str(sample_result[0]),
                sample_result[1].replace('"', "'"),
                sample_result[2].replace('"', "'"),
            )
            pbar.update(1)

        for example_id, example in enumerate(simplified_python):
            if example_id not in allowed_range:
                continue
            if output_key not in example:
                example[output_key] = [
                    ("error", None, None) for _ in range(args.n_samples)
                ]

    logfile.close()

    for banned_idx in BANNED_EXAMPLES:
        simplified_python[banned_idx][output_key] = [
            ("error", None, None) for _ in range(args.n_samples)
        ]

    with open(output_file, "w+") as f:
        json.dump(simplified_python, f, indent=4)

    all_passk = get_passk_scores(
        [
            simplified_python[i]
            for i in allowed_range
            if 0 <= i < len(simplified_python)
        ],
        code_key,
        output_key,
    )

    print(f"Results ({args.start} - {args.end})")
    for k, passk in enumerate(all_passk):
        k += 1
        print(f"Pass@{k}: {passk * 100}")


if __name__ == "__main__":
    # use for debug: python execution.py --round 0 --mode "debug"
    # use for the final pass: python execution.py --round 3 --mode final; python execution.py --round 3 --mode final --skip_sh
    # use for code-gen testing: python execution.py --round 3 --mode test_gen --skip_sh --code_gen_mpdel gpt-4 --code_gen_mode ...

    parser = argparse.ArgumentParser()
    parser.add_argument("--hypo_file", type=str, required=True)
    parser.add_argument(
        "--data_file", type=str, default="data/test_set_final_round3.json"
    )
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)

    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument(
        "--code_gen_model", type=str, default="gpt-4"
    )  # only used when mode == 'test_gen'
    parser.add_argument("--code_key", type=str, default="code")

    args = parser.parse_args()

    main(args)
