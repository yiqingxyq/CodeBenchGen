"""Validate consistency of LLM-generated augmented tests by running them against the reference target code"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys

from tqdm import tqdm

from src.ts_utils import find_func_node, ts_parser
from test_augmentation.test_templates import (
    EXTEND_TEST_CODE,
    FULL_CODE,
    FUNCTION_ORACLE_TEST_CODE,
    METHOD_ORACLE_TEST_CODE,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CACHE_DIR = os.environ.get("docker_CACHE_DIR")

CACHE_DIR = os.path.join(CACHE_DIR, 'test_aug')
if not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)


def extract_code(completion: str) -> str:
    code_pattern = r"```(\w*)\n(.*?)\n```"

    match = re.findall(code_pattern, completion, flags=re.DOTALL)
    return [m[1] for m in match]


def get_oracle_test_functions(text):
    matches = extract_code(text)
    get_inputs_node = validate_output_node = None
    for match in matches:
        test_code_root = ts_parser.parse(bytes(match, "utf-8")).root_node
        if get_inputs_node is None:
            try:
                get_inputs_node = find_func_node(test_code_root, "get_inputs")
            except:
                pass
        if validate_output_node is None:
            try:
                validate_output_node = find_func_node(test_code_root, "validate_output")
            except:
                pass

    if get_inputs_node is None or validate_output_node is None:
        return None

    get_inputs = get_inputs_node.text.decode("utf-8")
    validate_output = validate_output_node.text.decode("utf-8")

    return get_inputs, validate_output


def get_extend_test_function(text, func_name):
    matches = extract_code(text)
    new_test_code = None
    new_test_func_name = f"test_hard_{func_name}"
    for match in matches:
        test_code_root = ts_parser.parse(bytes(match, "utf-8")).root_node
        try:
            new_test_node = find_func_node(test_code_root, new_test_func_name)
            new_test_code = new_test_node.text.decode("utf-8")
            break
        except:
            pass

    return new_test_code


MAX_VIRTUAL_MEMORY = 4 * 1024 * 1024 * 1024  # 4 GB

ERR_KEYWORDS = ["ERROR", "Error", "error"]


def limit_virtual_memory(max_virtual_memory):
    # We do a soft limit in order to be able to change the limit later if needed
    return f"ulimit -S -v {max_virtual_memory}"


def direct_return(proc, proc_name, timeout=10):
    try:
        try:
            result, stderr = proc.communicate(timeout=timeout)
            if proc.returncode:
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


def run_python_program(script_path, i, timeout=10):
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


def main(args):
    with open(args.test_file) as f:
        new_tests = json.load(f)

    id2test = {}
    for t in new_tests:
        id2test[t["idx"]] = t

    with open(args.data_file) as f:
        splitdata = json.load(f)

    execution_results = [("error", "", "placeholder") for _ in range(len(splitdata))]

    for example_idx, example in enumerate(tqdm(splitdata)):
        func_name = example["func_name"]
        idx = example["idx"]

        func_name_last = func_name.split(".")[-1]

        code = example["code"]

        root_node = ts_parser.parse(bytes(code, "utf-8")).root_node
        try:
            test_func_node = find_func_node(root_node, "test_" + func_name_last)
        except:
            logger.warning(f"Unable to get test function node for {func_name}")
            continue
        nontest_code = code[: test_func_node.start_byte].rstrip()

        focal_node = find_func_node(root_node, func_name, duplicate_ok=True)
        focal_code = focal_node.text.decode("utf-8").strip()
        oracle_code = focal_code.replace(
            f"def {func_name_last}", f"def {func_name_last}_oracle", 1
        )

        new_output = id2test[idx]
        if new_output.get("output") is None:
            continue
        for test_index, new_text in enumerate(new_output["output"]):
            if args.prompt == "oracle":
                oracle_code_extract = get_oracle_test_functions(new_text)
                if oracle_code_extract is None:
                    execution_results[example_idx] = (
                        "error",
                        "",
                        "couldn't find get_inputs_node and/or validate_output_node",
                    )
                    continue

                get_inputs, validate_output = oracle_code_extract

                if "." in func_name:  # class method
                    test_code_template = METHOD_ORACLE_TEST_CODE
                else:  # function
                    test_code_template = FUNCTION_ORACLE_TEST_CODE

                new_test_code = test_code_template.format(
                    oracle_code=oracle_code,
                    get_inputs=get_inputs,
                    validate_output=validate_output,
                    func_name_last=func_name_last,
                )
            else:
                new_test_code = get_extend_test_function(new_text, func_name_last)
                if new_test_code is None:
                    execution_results[example_idx] = (
                        "error",
                        "",
                        f"couldn't find node for function test_hard_{func_name_last}",
                    )
                    continue

                new_test_code = EXTEND_TEST_CODE.format(
                    test_code=new_test_code, func_name_last=func_name_last
                )

            full_code = FULL_CODE.format(
                nontest_code=nontest_code,
                new_test_code=new_test_code,
            )
            code_filepath = os.path.join(CACHE_DIR, f"tmp_{func_name}.py")
            with open(code_filepath, "w+") as f:
                f.write(full_code)

            result, _ = run_python_program(code_filepath, idx, timeout=5)

            if result[0] == "success":
                example["new_test_code"] = new_test_code
                example["new_test_index"] = test_index
                if "new_test_error" in example:
                    del example["new_test_error"]
                break
            else:
                example["new_test_code"] = None
                example["new_test_error"] = result

        execution_results[example_idx] = result

    outcome = [r[0] for r in execution_results]
    num_succ = sum(a == "success" for a in outcome)
    num_fail = sum(a == "error" for a in outcome)
    logger.info(f"% augmented       : {num_succ}/{len(outcome)}")
    logger.info(f"% not augmented   : {num_fail}/{len(outcome)}")

    logger.info(f"Writing execution outputs to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(splitdata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", type=str, default="data/test_set_final_round3.json"
    )
    parser.add_argument("--test_file", type=str, default="new_test_output.json")
    parser.add_argument("--output_file", type=str, default="exec_new_test_output.json")
    parser.add_argument(
        "--prompt", type=str, default="oracle", choices=["oracle", "extend"]
    )

    args = parser.parse_args()

    main(args)
