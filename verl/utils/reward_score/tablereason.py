from verl.utils.reward_score import prime_math
from verl.utils.reward_score import math

def compute_score(model_output: str, ground_truth: str) -> bool:
    model_output = str(model_output)
    ground_truth = str(ground_truth)
    _, response = prime_math.match_answer(model_output)
    if "|" not in ground_truth:
        # Single answer
        score = math.is_equiv(response, ground_truth)
    else:
        # Multiple answers, in format "ans1|ans2|ans3"
        try:
            ground_truth = sorted([ans.strip() for ans in ground_truth.split("|")])
            response = sorted([ans.strip() for ans in response.split("|")])
            if len(ground_truth) != len(response):
                score = 0
            else:
                score = 1
                for gt, res in zip(ground_truth, response):
                    if not math.is_equiv(gt, res):
                        score = 0
                        break
        except Exception as e:
            print(f"Error postprocessing ground truth or response in tablereason.py: {e}")
            print(ground_truth)
            print(response)
            score = 0

    return {"score": score, "acc": score}
