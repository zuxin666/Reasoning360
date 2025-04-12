import re
import random
import ast
import operator



def validate_format(response):
    if not response:
        return False
    
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return True
    return False

def extract_solution(solution_str):
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    # for llama chat template
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        solution_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    # qwen chat template
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)

    if matches:
        final_answer = matches[-1].group(1).strip()
        if re.search(r'^[A-Za-z]+$', final_answer):
            return final_answer
    else:
        return None

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)

    target = ground_truth.lower()
    solution = extract_solution(solution_str)
    if solution:
        solution = solution.lower()
    do_print = random.randint(1, 64) == 1
    # delete this
    do_print = True
    if do_print:
        print(f"target: {target}, solution: {solution}")
    if not solution:
        return 0.0
    
    if not validate_format(solution_str):
        if do_print:
            print(f"Format not followed properly")
        return format_score

    try:
        if target == solution:
            return score
        else:
            if do_print:
                print(f"Solution not correct")
            return format_score
    except Exception as e:
        if do_print:
            print(f"Error in computing score: {e}")
        return format_score

    



