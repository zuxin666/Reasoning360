import re
import random
import ast
import operator

def extract_solution(solution_str):

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)

    if matches:
        final_answer = matches[-1].group(1).strip()
        if re.search(r'^[A-Za-z]+$', final_answer):
            return final_answer
    else:
        return None

def compute_score(solution_str, ground_truth):
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)

    target = ground_truth.lower()
    solution = extract_solution(solution_str)
    
    if solution:
        solution = solution.lower()
    else:
        return 0.0

    try:
        if target == solution:
            return 1.0
        else:
            return 0.0

    except Exception as e:
        return 0.0

    



