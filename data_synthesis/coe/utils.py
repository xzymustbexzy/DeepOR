import re
import json
import os


def extract_code_from_string(input_string):
    # Match code within ```python ... ``` or ``` ... ``` blocks
    # First try to match ```python\n ... ``` pattern
    pattern_python = r'```python\s*\n(.*?)\n```'
    code_blocks = re.findall(pattern_python, input_string, re.DOTALL)
    
    # If no python blocks found, try general ``` ... ``` pattern
    if not code_blocks:
        pattern_general = r'```\s*\n?(.*?)\n?```'
        code_blocks = re.findall(pattern_general, input_string, re.DOTALL)

    if len(code_blocks) == 0:
        # print(f'Parse code error! {input_string}')
        return input_string
    elif len(code_blocks) == 1:
        return code_blocks[0]

    code_blocks = [code for code in code_blocks if 'pip' not in code]
    return '\n'.join(code_blocks)


def read_problem(dataset, problem_name):
    base_dir = 'dataset'
    with open(os.path.join(base_dir, dataset, problem_name, 'description.txt'), 'r', encoding='utf8') as f:
        description = f.read()

    with open(os.path.join(base_dir, dataset, problem_name, 'code_example.py'), 'r', encoding='utf8') as f:
        code_example = f.read()

    return {
        'description': description,
        'code_example': code_example
    }
