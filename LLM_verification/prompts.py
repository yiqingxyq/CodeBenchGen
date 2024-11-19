sandbox_check_prompt_template = f"""\
Instructions:
- We revised a python function called __FUNCTION_NAME__ so it can be directly executed in an isolated environment.
- You are given the ORIGINAL FUNCTION and the CODE containing the REVISED FUNCTION.
- Your task is to check whether the functionality of the REVISED FUNCTION is the same as the ORIGINAL FUNCTION.
- If the REVISED FUNCTION is exactly the same as the ORINIGAL FUNCTION, output "same" as your answer.
- Otherwise, if the functionality of the REVISED FUNCTION is the same as the ORIGINAL FUNCTION, output "yes" as your answer.
- if the functionality of the REVISED FUNCTION is different, output "no".

- Provide your reasoning and the answer under "SOLUTION".

ORIGINAL FUNCTION:
__ORIG_FUNC__

CODE containing the REVISED FUNCTION:
__NEW_CODE__

Your answer should follow the format below:
```
REASONING: Your reasoning,
ANSWER: "same", "yes" or "no".
```

Do NOT include other formatting.

SOLUTION:
"""

test_check_prompt_template = f"""\
Instructions:
- You are given a piece of PYTHON CODE containing a function called __FUNCTION_NAME__ and its test function called __TEST_FUNCTION_NAME__.
- Your task is to judge whether the test function satisfies both CONDITIONS:
[CONDITION 1] The test function is verifying the correctness of the __FUNCTION_NAME__ function.
[CONDITION 2] At least one test case is non-trivial.

- If the test function satisfies both CONDITIONS, answer "yes". Otherwise, answer "no".
- Provide your reasoning and the answer under "SOLUTION".

PYTHON CODE:
__CODE__

Your answer should follow the format below:
```
REASONING: Your reasoning,
ANSWER: "yes" or "no".
```

Do NOT include other formatting.

SOLUTION:
"""


instruction_check_prompt_template = f"""\
Instructions:
- You are given a piece of PYTHON CODE containing a function called __FUNCTION_NAME__ and the INSTRUCTION to implement this function.
- Your task is to judge whether the INSTRUCTION is clear enough and well aligned with the __FUNCTION_NAME__ function.
- Answer with "yes" or "no". Provide your reasoning and the answer under "SOLUTION".

PYTHON CODE:
__CODE__

INSTRUCTION:
__INSTRUCTION__

Your answer should follow the format below:
```
REASONING: Your reasoning,
ANSWER: "yes" or "no".
```

Do NOT include other formatting.

SOLUTION:
"""
