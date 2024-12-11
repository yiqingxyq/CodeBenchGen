import os
import re 
import ast
import tokenize
from io import StringIO
from copy import deepcopy

import openai

client = openai.OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

CODE_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"
def extract_code(text: str, pattern: str = CODE_BLOCK_PATTERN):
    match = re.findall(pattern, text, flags=re.DOTALL)
    return match if match else []


def call_openai_completion(**kwargs):
    retry_count = 0
    while retry_count <= 20:
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        
        except openai.RateLimitError as error:
            print(f"OpenAI API retry for {retry_count} times ({error})")
            time.sleep(5)
            retry_count += 1
            continue

        except Exception as error:
            print(error)
            return None
        
    print('Retry limit reached.')
    return None


def count_code_tokens(code):
    code_io = StringIO(code)
    tokens = list(tokenize.generate_tokens(code_io.readline))
    return len(tokens)


def check_func_body_match(script, func_code):
    return "".join(func_code.split()) in "".join(script.split())


def get_class_function_line_idx(script, func_name, class_name):
    """
    Finds the start and end line numbers of a specific class method in a Python script.
    """
    start_line, end_line = None, None
    try:
        tree = ast.parse(script)

        # Iterate over nodes in the AST
        for node in ast.walk(tree):
            # Find the specified class
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Search for the specified function in the class
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef) and sub_node.name == func_name:
                        if start_line is not None:
                            # error: multiple occurrence of the same function
                            print(f"Error: Multiple occurrence of function: {class_name}.{func_name}")
                            return None, None 
                        
                        # Return the start and end line numbers of the function
                        start_line = sub_node.lineno - 1
                        # Using end_lineno (introduced in Python 3.8+)
                        end_line = getattr(sub_node, "end_lineno", None) - 1
                    
    except Exception as e:
        print(f"Error occurred when search for funcion {class_name}.{func_name}: {e}")

    return start_line, end_line

def get_standalone_function_line_idx(script, func_name):
    """
    Finds the start and end line numbers of a standalone function in a Python script.
    """
    start_line, end_line = None, None
    try:
        tree = ast.parse(script)

        # Iterate over nodes in the AST to find standalone functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                if start_line is not None:
                    # error: multiple occurrence of the same function
                    print(f"Error: Multiple occurrence of function: {func_name}")
                    return None, None 
                
                # Return the start and end line numbers of the function
                start_line = node.lineno - 1
                # Using end_lineno (introduced in Python 3.8+)
                end_line = getattr(node, 'end_lineno', None) - 1
            
    except Exception as e:
        print(f"Error occurred when search for funcion {func_name}: {e}")

    return start_line, end_line


def get_function_line_idx(script, func_name, class_name=None):
    if class_name is not None:
        return get_class_function_line_idx(script, func_name, class_name)
    else:
        return get_standalone_function_line_idx(script, func_name)
    

MAIN_FUNC_STR = 'if __name__ == "__main__":'
MAIN_FUNC_STR2 = "if __name__ == '__main__':"
def remove_main(script):
    lines = script.split("\n")
    for lid, line in enumerate(lines):
        if MAIN_FUNC_STR in line or MAIN_FUNC_STR2 in line:
            return "\n".join(lines[:lid])
    return script


def extract_and_rename_new_implementation(script, func_name, class_name=None):
    lines = script.split("\n")
    start_line, end_line = get_function_line_idx(script, func_name, class_name)
    
    if start_line is None:
        return None
    
    # sth like @static
    if start_line > 0 and len( lines[start_line-1].strip() ) > 0:
        start_line = start_line - 1
    
    # rename function
    new_func_name = f"{func_name}_new_implementation"
    function_lines = deepcopy(lines[start_line:end_line+1])
    for lid,line in enumerate(function_lines):
        if f"def {func_name}" in line:
            function_lines[lid] = function_lines[lid].replace(f"def {func_name}", f"def {new_func_name}")
            return "\n".join(function_lines)
    
    return None


def insert_new_implementation(new_implementation, script, func_name, class_name=None):
    lines = script.split("\n")
    start_line, end_line = get_function_line_idx(script, func_name, class_name)
    
    if start_line is None or new_implementation is None:
        return None
    
    lines.insert(end_line+1, f"\n{new_implementation}\n")
    
    script = "\n".join(lines)
    script = re.sub(r'(\n){4,}', '\n\n\n', script)
    return script


def remove_function_if_exist(script, func_name, class_name=None):
    lines = script.split("\n")
    start_line, end_line = get_function_line_idx(script, func_name, class_name=class_name)
    if start_line is None:
        return script
    
    if start_line > 0 and len( lines[start_line-1].strip() ) > 0:
        start_line = start_line - 1

    func_content = "\n".join(script.split("\n")[start_line:end_line+1])
    script = script.replace(func_content, "\n")
    script = re.sub(r'(\n){4,}', '\n\n\n', script)
    
    return script