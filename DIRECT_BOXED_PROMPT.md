Goal:
Implement a minimal diagnostic variant of the DMLR LTPO pipeline that forces the assistant response to start with "\boxed{" and directly generate only the final answer plus "}", without visible CoT. This is for testing whether LTPO can improve under an direct reward on the answer token, instead of the current thought-token confidence reward that is often dominated by <|im_end|> / newline boundary tokens.

Important constraints:
1. Do NOT modify existing files unless absolutely necessary.
2. Create new files by copying/adapting the current DMLR files:
   - Create ltpo_vl_dmlr_direct_boxed.py from ltpo_vl_dmlr.py.
   - Create main_vl_dmlr_direct_boxed.py from main_vl_dmlr.py.
   - Create scripts/run_ltpo_vl_dmlr_direct_boxed_oracle_dev.sh.
3. Keep the code changes as small and local as possible.
4. Preserve the existing model loading, processor setup, visual-token pre-merging, dataset loading, answer extraction, and LLM verifier logic as much as possible.
5. Use the string "\\boxed{" in Python source. Do NOT use "\bboxed{".

Background:
Current ltpo_vl_dmlr.py uses a DMLR-style prompt. For multiple-choice questions it still says:
"- First, solve the problem step by step."
and then asks for the final answer in \boxed{}.
It also inserts latent thought tokens in the user prompt and then calls apply_chat_template(..., add_generation_prompt=True). The current generation can therefore produce visible CoT. I want a new variant where the assistant side is forced to begin with "\\boxed{", so the model should only generate the answer content and a closing "}".

Implementation details:

A. New file: ltpo_vl_dmlr_direct_boxed\.py

Start by copying ltpo_vl_dmlr.py.

Modify/add the following:

1. Add a constant:
   ASSISTANT_BOXED_PREFIX = "\\boxed{"

2. Modify build_inputs_vl(...) so that:
   - Keep the latent thought tokens in the prompt exactly as before, because we still want to optimize them.
   - After calling processor.apply_chat_template(..., add_generation_prompt=True), append ASSISTANT_BOXED_PREFIX to the text before tokenization:
      text = text + ASSISTANT_BOXED_PREFIX

3. Keep thought_idx detection unchanged. The thought tokens should still be inside the prompt before the assistant prefix.

4. For the reward for the direct boxed version, we use LTPO-style reward but on the first generated token (the prediction from the last token in the current input sequence), so that by maximizing this reward, we maximize the confidence for the first generated token (which is part of the answer).

B. New file: main_vl_dmlr_direct_boxed.py

Start by copying main_vl_dmlr.py.

This file should have the same behaviour with its source. Remember that even when we use eval_baseline, the model still generates answer with ASSISTANT_BOXED_PREFIX added to the input prompt.

C. New script: scripts/run_ltpo_vl_dmlr_direct_boxed_dev.sh

Create a shell script with a MODE argument:
   MODE=${1:-ltpo}

If MODE=baseline:
   we run the eval_baseline.

If MODE=ltpo:
   run LTPO with our new reward with a hyperparameter searching range (you can refer to `scripts/run_ltpo_vl_dmlr_paired_grid_dev.sh`).

After coding, summarize:
1. Files created.
2. Functions added.
3. How direct "\\boxed{" prefix is injected.
4. How current reward is computed.
5. Commands to run baseline / ltpo.