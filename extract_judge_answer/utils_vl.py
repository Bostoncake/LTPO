"""
Answer extraction and judging for MLLM (Vision-Language) datasets.

Answer verification uses an LLM judge via an OpenAI-compatible API.
Required environment variables:
    OPENAI_API_KEY          – API key
    OPENAI_API_BASE_URL     – base URL  (default: https://api.openai.com/v1)
    MODEL_TYPE              – judge model name (default: gpt-4o)
"""
import os
import re

from openai import OpenAI

MLLM_DATASET_NAMES = ['hallusion', 'math_vision', 'math_vista', 'mm_math', 'mmstar', 'mmvp', 'scienceqa']

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _judge_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ.get('OPENAI_API_BASE_URL', 'https://api.openai.com/v1'),
    )


# _JUDGE_SYSTEM = (
#     "You are a strict answer evaluator for vision-language benchmark tasks. "
#     "Your only job is to decide whether a model's answer conveys the same meaning "
#     "as the ground-truth answer. Be lenient about formatting differences "
#     "(e.g. 'A', 'A.', '(A)', 'Option A' are all equivalent option labels). "
#     "Respond with exactly one word: 'correct' or 'wrong'."
# )

_JUDGE_SYSTEM = (
    "You are an answer equivalence checker. "
    "Decide if the candidate answer and the ground truth express the same final result. "
    "Treat option labels as equivalent regardless of formatting: "
    "'A', 'A.', '(A)', '(a)', 'a)', 'option A', 'A:' all refer to option A. "
    "Reply with exactly one word: yes or no."
)

_JUDGE_USER_TMPL = """\
Candidate answer: {output}
Ground truth: {label}

Same result? Reply yes or no."""


def _normalize_option(text: str) -> str | None:
    """
    Try to reduce text to a single uppercase option letter (A-F).
    Handles: A, (a), a), A., A:, A - ..., (A) Open, etc.
    Returns None if the text cannot be reduced to a single letter.
    """
    if not text:
        return None
    # Strip common wrappers and take the first letter found
    m = re.search(r'\(([A-Fa-f])\)|^([A-Fa-f])[^a-zA-Z]|^([A-Fa-f])$', text.strip())
    if m:
        letter = next(g for g in m.groups() if g is not None)
        return letter.upper()
    return None


def llm_judge(output: str, combined_output: str, label: str, data_name: str, question: str = '') -> bool:
    """
    Judge correctness.  For option-letter answers both sides are first normalised
    to a bare letter (A-F); if both reduce cleanly the comparison is done locally
    without an API call.  Otherwise the LLM judge is invoked.
    """
    
    # print(f"Output: {output}, Label: {label}.")

    # Fast path: both sides are just an option letter
    # NOTE: currently we don't use this path
    norm_out = _normalize_option(output)
    norm_lbl = _normalize_option(label)
    if norm_out is not None and norm_lbl is not None:
        print(f"[No LLM judge] Normalized option: {norm_out}, {norm_lbl}.")
        return norm_out == norm_lbl
    client = _judge_client()
    model = os.environ.get('MODEL_TYPE', 'gpt-4o')

    user_msg = _JUDGE_USER_TMPL.format(output=combined_output, label=label)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': _JUDGE_SYSTEM},
                {'role': 'user', 'content': user_msg},
            ],
            max_tokens=8,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().lower()
        return verdict.startswith('yes')
    except Exception as e:
        print(f"[LLM judge ERROR] {e}")
        return False


# ---------------------------------------------------------------------------
# extract_true_answer_vl
# ---------------------------------------------------------------------------

def extract_true_answer_vl(text: str, name: str = '') -> str:
    """Return the ground-truth solution string as-is for MLLM datasets."""
    return text


# ---------------------------------------------------------------------------
# extract_answer_vl  (from model output)
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> str | None:
    """Extract the innermost \\boxed{...} content."""
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return None
    # Take the last match and find its closing brace
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return text[start:].strip()


def _extract_option_letter(text: str) -> str | None:
    """
    Extract a single option letter (A-F or a-f) from text.
    Handles formats like: A, (A), A., A:, Option A, the answer is A.
    Returns uppercase letter.
    """
    # Explicit boxed option
    boxed = _extract_boxed(text)
    if boxed:
        m = re.search(r'\b([A-Fa-f])\b', boxed)
        if m:
            return m.group(1).upper()

    # Common patterns, searched from the end of the text
    patterns = [
        r'(?:answer\s+is|answer:)\s*\(?([A-Fa-f])\)?',
        r'\b([A-Fa-f])\s*[.:\)]\s',
        r'\(([A-Fa-f])\)',
        r'\b([A-Fa-f])\b',
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()
    return None


def _extract_number(text: str) -> str | None:
    """Extract the last numeric value from text (int or float)."""
    boxed = _extract_boxed(text)
    if boxed:
        m = re.search(r'-?[\d,]+(?:\.\d+)?', boxed)
        if m:
            return m.group(0).replace(',', '')

    matches = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    return matches[-1].replace(',', '') if matches else None


def extract_answer_vl(text: str, data_name: str = '') -> str | None:
    """
    Extract a canonical answer from a model's response for MLLM datasets.

    Strategy (in order):
      1. Extract \\boxed{} content
      2. For datasets with option-letter answers, extract the letter
      3. Fall back to the last numeric value
    """
    if text is None:
        return None

    # NOTE: Currently only boxed answers count.
    # # Datasets that are primarily option-letter based
    # option_datasets = ['math_vision', 'mmstar', 'mmvp', 'scienceqa', 'math_vista']
    # is_option = any(d in data_name.lower() for d in option_datasets)

    # # Datasets that are primarily yes/no
    # yesno_datasets = ['hallusion']
    # is_yesno = any(d in data_name.lower() for d in yesno_datasets)

    # if is_yesno:
    #     boxed = _extract_boxed(text)
    #     candidate = boxed if boxed else text
    #     m = re.search(r'\b(yes|no)\b', candidate, re.IGNORECASE)
    #     return m.group(1).capitalize() if m else None

    # if is_option:
    #     letter = _extract_option_letter(text)
    #     if letter:
    #         return letter
    #     # Fall through to numeric for mixed datasets like math_vista

    boxed = _extract_boxed(text)
    if boxed:
        return boxed.strip()

    return _extract_number(text)


# ---------------------------------------------------------------------------
# judge_answer_vl
# ---------------------------------------------------------------------------

def judge_answer_vl(
    output: str,
    label: str,
    data_name: str = '',
    question: str = '',
) -> bool:
    """
    Judge whether the model output matches the ground-truth label.

    Uses the LLM judge for all MLLM datasets.  The extracted answer from the
    model output is also passed so the judge sees a cleaner signal, but the
    full output is provided for context.
    """
    extracted = extract_answer_vl(output, data_name=data_name)
    # Pass both the extracted answer and raw output to the judge for robustness
    combined_output = (
        f"[Extracted answer]: {extracted}\n[Full response]: {output}"
        if extracted
        else output
    )
    return llm_judge(
        output=extracted,
        combined_output=combined_output,
        label=label,
        data_name=data_name,
        question=question,
    )
