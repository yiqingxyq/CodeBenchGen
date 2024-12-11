import os 
docker_CACHE_DIR = os.environ.get("docker_CACHE_DIR")

sandbox_prompt_template_ORIG = f"""\
Rewrite the following piece of python code so that we can test the implementation of the __FUNCTION_NAME__ function by directly executing this piece of code. You should keep the input code as much as possible and output every token of the content with no abbreviation. You can add code that is necessary. The code will be run in an isolated environment with no access to GPU or external API keys. 
Keep the function name and input arguments of the __FUNCTION_NAME__ function as the same. Output one complete piece of code in the format of ```python ```.
If necessary, output `pip` or `python -m` commands in another code block in the format of ```bash ```.
__CODE__
"""

sandbox_prompt_template = f"""\
Instructions:
- You're given a piece of PYTHON CODE containing a function called __FUNCTION_NAME__. Your goal is to revise the PYTHON CODE so that we can directly call the __FUNCTION_NAME__ function WITHOUT ANY MODIFICATIONS.
- You should edit the original PYTHON CODE as little as possible and you can add code only if necessary.
- DO NOT call any external API, database, etc. Instead, create a mock interface.
- Make sure that your code can be directly executed without any modification. For example, statements like `token = "your_auth_token_here"  # You need to replace this with a real token` is NOT allowed.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.

- Provide your reasoning and the revised PYTHON CODE below SOLUTION.

PYTHON CODE:
```python
__CODE__
```

Your answer should follow the format below:

Reasoning: ...
```python
# Your Code. 
```

Do NOT include other formatting. Output every token of the content with no omission or abbreviation.

SOLUTION:
"""

aggregate_and_sandbox_prompt_template = f"""\
Instructions:
- You're given a piece of PYTHON CODE containing a function called __FUNCTION_NAME__. We also provide you the CONTEXT of the PYTHON CODE. Your goal is to aggregate the PYTHON CODE and the CONTEXT into one script, so that we can directly call the __FUNCTION_NAME__ function WITHOUT ANY MODIFICATIONS.
- You should edit the original PYTHON CODE as little as possible and you can add code only if necessary.
- DO NOT call any external API, database, etc. Instead, create a mock interface.
- Make sure that your code can be directly executed without any modification. For example, statements like `token = "your_auth_token_here"  # You need to replace this with a real token` is NOT allowed.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.

- Provide your reasoning and the revised PYTHON CODE below SOLUTION.

PYTHON CODE:
```python
__CODE__
```

CONTEXT:
__CONTEXT__

Your answer should follow the format below:

Reasoning: ...
```python
# Your Code. 
```

Do NOT include other formatting. Output every token of the content with no omission or abbreviation. For example, abbreviation like `... # the code keeps unchanged` is NOT allowed.

SOLUTION:
"""

test_generation_prompt_template = f"""\
Instructions:
- You're given a piece of PYTHON CODE containing a function called __FUNCTION_NAME__. Assume we will later have another implentation of the __FUNCTION_NAME__ function called __FUNCTION_NAME___new_implementation.
- Your goal is to add (1) a test function called __TEST_FUNCTION_NAME__ to check whether __FUNCTION_NAME___new_implementation has the same functionality as the __FUNCTION_NAME__ function, and (2) a __main__ function that calls the test function.
- If the PYTHON CODE already contains a __main__ function, remove it and write a new __main__ function.
- The test function __TEST_FUNCTION_NAME__ should contain at least 3 assert statements. If __FUNCTION_NAME___new_implementation has different functionality as __FUNCTION_NAME__, an Assertion Error should be triggered.
- The test function __TEST_FUNCTION_NAME__ should cover all the major branches of the __FUNCTION_NAME__ function
- DO NOT test on error handling and DO NOT test on the print information in the function.
- The __main__ function should NOT contain a try-except structure. If the implementation is incorrect, the program should have a non-zero exit code.
- You should edit the original PYTHON CODE as little as possible.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.

- Provide your reasoning and the new PYTHON CODE containing your test function __TEST_FUNCTION_NAME__ and the __main__ function below SOLUTION.

PYTHON CODE:
```python
__CODE__
```

Your answer should follow the format below:

Reasoning: ...
```python
# The new PYTHON CODE containing your test function __TEST_FUNCTION_NAME__ and the __main__ function.
```

Do NOT include other formatting. Output every token of your edited PYTHON CODE with no omission or abbreviation.

SOLUTION:
"""

debug_prompt_template = f"""\
Instructions:
- You're given a piece of PYTHON CODE containing a function called __FUNCTION_NAME__ and its test function called __TEST_FUNCTION_NAME__. Assume we will later add another function called __FUNCTION_NAME___new_implementation, the test function aims to check whether __FUNCTION_NAME___new_implementation has the same functionality as __FUNCTION_NAME__.
- In our experiments, we implemented __FUNCTION_NAME___new_implementation exactly the same as __FUNCTION_NAME__, but the PYTHON CODE cannot be successfully executed. 
- Your task is to debug PYTHON CODE based on the ERROR MESSAGE.
- You should modify the code as little as possible, especially the test___FUNCTION_NAME__ function and the __FUNCTION_NAME__ function.
- Make sure that after debugging, the test function test___FUNCTION_NAME__ still have at least three assert statements and cover all the major branches of the __FUNCTION_NAME__ function.
- DO NOT test on error handling and DO NOT test on the print information in the function.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.

- Provide your reasoning and the debugged PYTHON CODE below SOLUTION. If necessary, output the bash scripts for Linux in another code block in the format of ```bash ... ```.

PYTHON CODE:
```python
__CODE__
```

ERROR MESSAGE:
```
__ERROR_MSG__
```

Your answer should follow the format below:

Reasoning: ...
```python
# The debugged PYTHON CODE in one piece.
```
```bash 
# the bash script, if necessary
```

Do NOT include other formatting. Output every token of your debugged PYTHON CODE with no omission or abbreviation.

SOLUTION:
"""