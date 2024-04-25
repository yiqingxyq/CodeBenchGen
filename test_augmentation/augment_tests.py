"""Augments test cases by using a code LLM to generate additional test functions"""

import argparse
import json
import logging
import os

import torch
from tqdm import tqdm

from src.decoders import build_decoder
from src.ts_utils import find_func_node, ts_parser
from test_augmentation.test_aug_prompts import (
    EXTEND_TEST_TEMPLATE,
    FUNCTION_ORACLE_TEST_TEMPLATE,
    METHOD_ORACLE_TEST_TEMPLATE,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(args):
    """
    Load LM decoder
    """
    decoder_args = {}

    if args.model_backend == "vllm":
        n_gpus = torch.cuda.device_count()
        decoder_args["tensor_parallel_size"] = n_gpus

    return build_decoder(args.model_name, args.model_backend, **decoder_args)


def main(args):
    n_samples = args.n_samples
    if "gpt-4" in args.model_name:
        n_samples = 1

    if args.prompt == "oracle":
        METHOD_TEMPLATE = METHOD_ORACLE_TEST_TEMPLATE
        FUNCTION_TEMPLATE = FUNCTION_ORACLE_TEST_TEMPLATE
    elif args.prompt == "extend":
        METHOD_TEMPLATE = FUNCTION_TEMPLATE = EXTEND_TEST_TEMPLATE

    model = load_model(args)

    with open(args.data_file) as f:
        data = json.load(f)

    if args.end == -1:
        args.end = len(data)

    data = data[args.start : args.end]

    if os.path.isfile(args.output_file):
        # Read saved partial outputs
        with open(args.output_file) as f:
            outputs = json.load(f)
    else:
        outputs = []

    pbar = tqdm(total=len(data))

    for example_idx, example in enumerate(data):
        if example_idx < len(outputs):
            continue
        idx = example["idx"]

        func_name = example["func_name"]
        code = example["code"]

        func_name_last = func_name.split(".")[-1]
        func_name_first = func_name.split(".")[0]
        root_node = ts_parser.parse(bytes(code, "utf-8")).root_node
        try:
            test_func_node = find_func_node(root_node, "test_" + func_name_last)
        except:
            import traceback

            exc_msg = traceback.format_exc()
            outputs.append({"output": None, "idx": idx, "error": exc_msg})
            continue

        nontest_code = code[: test_func_node.start_byte].rstrip()
        test_code = test_func_node.text.decode("utf-8")
        if "." in func_name:  # method
            test_gen_prompt = METHOD_TEMPLATE.format(
                func_name=func_name,
                code=nontest_code,
                test_func=test_code,
                func_name_first=func_name_first,
                func_name_last=func_name_last,
            )
        else:  # function
            test_gen_prompt = FUNCTION_TEMPLATE.format(
                func_name=func_name,
                code=nontest_code,
                test_func=test_code,
                func_name_last=func_name_last,
            )

        test_gen_prompt = [
            {"role": "user", "content": test_gen_prompt},
        ]

        tokenizer = model.get_tokenizer()
        if hasattr(tokenizer, "apply_chat_template"):
            from src.utils import set_chat_template

            set_chat_template(args.model_name, tokenizer)
            test_gen_prompt = tokenizer.apply_chat_template(
                test_gen_prompt, add_generation_prompt=True, tokenize=False
            )

        if example_idx == 0:
            logger.info(f"Example test generation prompt:\n\n{test_gen_prompt}")

        output, _ = model(
            test_gen_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=40,
            max_tokens=2048,
            n=n_samples,
        )

        outputs.append({"output": output, "idx": idx, "context": test_gen_prompt})

        logger.info(f"Saving partial output at #{example_idx}, func_name = {func_name}")
        with open(args.output_file, "w") as f:
            json.dump(outputs, f)

        pbar.update(1)

    with open(args.output_file, "w") as f:
        json.dump(outputs, f)

    logger.info("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", type=str, default="data/test_set_final_round3.json"
    )
    parser.add_argument("--output_file", type=str, default="new_test_output.json")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--model_backend", type=str, default="openai")
    parser.add_argument(
        "--prompt", type=str, default="oracle", choices=["oracle", "extend"]
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)

    args = parser.parse_args()

    main(args)
