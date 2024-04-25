import os
import re
from typing import Optional

from tree_sitter import Language, Node, Parser

TS_LANG_PATH = os.environ.get("TS_LANG_PATH")

lang = Language(TS_LANG_PATH, "python")
ts_parser = Parser()
ts_parser.set_language(lang)


func_query = lang.query("""(function_definition) @param_func""")
class_query = lang.query("""(class_definition) @param_class""")


def make_tree(code):
    return ts_parser.parse(bytes(code, "utf-8")).root_node


def get_indent_level(node: Node):
    level = 0
    while node is not None:
        if node.type == "block":
            level += 1
        node = node.parent
    return level


def find_child(tree_root: Node, obj_type: str, obj_name: Optional[str] = None):
    """Search for object among tree_root's depth 1 children"""
    if tree_root is None:
        return None
    for child_node in tree_root.named_children:
        if child_node.type == obj_type:
            if obj_name is None:
                return child_node
            child_name = child_node.named_children[0].text.decode("utf-8")
            if child_name == obj_name:
                return child_node
    return None


def get_node_name(node):
    return node.named_children[0].text.decode("utf-8")


def in_class_scope(node):
    while node.parent is not None:
        if node.parent.type == "class_definition":
            return True
        node = node.parent
    return False


def get_text(node):
    return node.text.decode("utf-8")


def find_func_node(root_node: Node, func_name: str, duplicate_ok=False):
    if "." not in func_name:
        fnodes = [
            x for x, _ in func_query.captures(root_node) 
            if get_node_name(x) == func_name and not in_class_scope(x) and get_indent_level(x) == 0
        ]
    else:
        class_name, func_name = func_name
        cls_nodes = [
            x for x, _ in class_query.captures(root_node) if get_node_name(x) == class_name
        ]
        if len(cls_nodes) != 1:
            return None 
        
        class_node = cls_nodes[0]
        fnodes = [
            x for x, _ in func_query.captures(class_node) 
            if get_node_name(x) == func_name
        ]
        
    if len(fnodes) == 0 or (not duplicate_ok and len(fnodes) != 1):
        return None 
        
    func_node = max(fnodes, key=lambda n: len(n.text)) # assume longest func is the one we want
    return func_node


def find_main_node(root_node):
    for node in root_node.children:
        if node.type == 'if_statement':
            comparisons = [x for x in node.children if x.type=='comparison_operator']
            if len(comparisons) == 0:
                return None 
            comparison = comparisons[0]
            
            if len(comparison.children) != 3 or get_text(comparison.children[1]) != '==':
                return None 
            
            identifier, value = get_text(comparison.children[0]), get_text(comparison.children[-1])
            if "__name__" in identifier and "__main__" in value:
                return node
    return None


def is_target_function_call(node, target_func_name):
    """
    Checks if a given node represents a call to the target function.
    """
    if node.type == 'call':
        fname_node = node.child_by_field_name('function')
        if fname_node:
            call_text = fname_node.text.decode("utf-8")
            call_name = call_text.split('.')[-1]
            return call_name == target_func_name
    return False


def check_function_calls_by_name(node, target_func_name):
    """
    Recursively searches for and prints the location of calls to the target function.
    """
    if is_target_function_call(node, target_func_name):
        return True
    
    for child in node.children:
        ans = check_function_calls_by_name(child, target_func_name)
        if ans:
            return True 
    return False 
    
    
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


def get_comments_positions(node):
    if node.type == 'comment':
        start, end = node.start_point, node.end_point
        return [(start, end)]
    else:
        pos_list = []
        for child in node.children:
            child_pos_list = get_comments_positions(child)
            pos_list.extend(child_pos_list)
        return pos_list
    
    
def remove_comments(source_code):
    tree = ts_parser.parse(bytes(source_code, 'utf-8'))
    comments_list = get_comments_positions(tree.root_node)
    
    line2start = {start[0]:start[1] for start,_ in comments_list}
    new_lines = []
    for lid,line in enumerate(source_code.split('\n')):
        if lid not in line2start:
            new_lines.append(line)
        else:
            part = line[:line2start[lid]]
            if len(part.split()) == 0:
                continue 
            else:
                new_lines.append(part)
    source_code = '\n'.join(new_lines)
    return source_code