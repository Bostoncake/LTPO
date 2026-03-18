"""
Prompt templates for Vision-Language LTPO.
"""


def vl_cot_prompt(q, prompt_idx=0):
    """
    CoT prompts for Vision-Language models.

    Args:
        q (str): The question text (image is passed separately to the processor).
        prompt_idx (int): 0 = step-by-step CoT, 1 = detailed CoT, 2 = direct answer.

    Returns:
        str: Formatted prompt text.
    """
    prompts = [
        # idx 0: Baseline CoT (default)
        (
            "Please analyze the image carefully and solve this problem step by step.\n"
            "Show your reasoning process clearly, then put your final answer within \\boxed{}.\n\n"
            f"Question: {q}"
        ),
        # idx 1: Detailed CoT
        (
            "Let's solve this visual math problem step by step:\n"
            "1. First, carefully observe the image and identify all relevant information.\n"
            "2. Break down the problem into smaller steps.\n"
            "3. Show your calculations and reasoning for each step.\n"
            "4. Finally, provide your answer within \\boxed{}.\n\n"
            f"Question: {q}"
        ),
        # idx 2: No CoT (direct answer)
        (
            "Please provide your final answer within \\boxed{}.\n\n"
            f"Question: {q}"
        ),
    ]
    return prompts[prompt_idx]


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think>"
    "<answer> answer here </answer>"
)
