"""
Reward model for Vision-Language LTPO.

Mirrors the text-only ``RewardModel`` in reward.py but works with a VL
processor instead of a plain tokenizer, and uses ``<|endoftext|>`` tokens
(Qwen family) for the latent thought token placeholder.
"""
import re

from termcolor import colored


VERA_ANSWER_SYMBOL = "THE SCORE IS:"


class VLRewardModel:
    def __init__(self, model, processor, num_thought_tokens, device="cuda"):
        self.model = model
        self.processor = processor
        self.num_thought_tokens = num_thought_tokens
        self.device = device

    def get_reward(self, question, specil_tokens_embeds):
        """Score the usefulness of latent thought-token embeddings."""
        system_instruct = (
            f"You are a critical scorer tasked with scoring some special tokens for visual reasoning problems. "
            f"You will be provided a question and {self.num_thought_tokens} special tokens. "
            f"These {self.num_thought_tokens} special tokens will be used for helping another language model to "
            f"think and reason in order to solve the given question.\n"
            f"Your job is to carefully go over the question, then give a score that can "
            f"precisely represent how useful these special tokens are in terms of helping thinking "
            f"and reasoning for solving the reasoning problem.\n"
            f"NOTE: the score MUST be between 0 and 1 (both are inclusive), where 0 means totally useless, "
            f"while 1 means extremely useful.\n"
        )
        latent_tokens = "<|endoftext|>" * self.num_thought_tokens
        vera_prompt = (
            f"{system_instruct}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"SPECIAL TOKENS:\n{latent_tokens}\n\n"
            f"INSTRUCTIONS:\n"
            f'You MUST output the score in following format: "{VERA_ANSWER_SYMBOL}[your_score]".\n'
            f'For example: if the score is 0.75, then your output SHOULD BE "{VERA_ANSWER_SYMBOL}0.75"\n\n'
        )
        message = [{"role": "user", "content": vera_prompt}]
        inputs = self.processor.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(self.device)
        inputs_embeds = self.model.get_input_embeddings()(inputs["input_ids"])

        # Locate the endoftext token span
        eos_id = self.processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        ids_list = inputs["input_ids"][0].tolist()
        start_idx = ids_list.index(eos_id)
        end_idx = start_idx + self.num_thought_tokens

        inputs_embeds[0, start_idx:end_idx] = specil_tokens_embeds
        inputs["inputs_embeds"] = inputs_embeds
        inputs.pop("input_ids")

        answer = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)[0]
        response = self.processor.tokenizer.decode(answer, skip_special_tokens=True)
        print(colored(f"Response from Reward Model:\n{response}", "light_blue"))
        return self.extract_score(response)

    @staticmethod
    def extract_score(response):
        pattern = r"(?:THE SCORE IS:?|THE SCORES ARE:?|\*\*Score:?\*\*|\*Score:?\*|Score:?)\s*(\d+(?:\.\d+)?)"
        matches = re.findall(pattern, response, re.IGNORECASE)
        if not matches:
            print(colored(
                f"WARNING: no score found in verifier response:\n"
                f"{'-'*30}\n{response}\n{'-'*30}", "yellow",
            ))
            return 0.0
        score = float(matches[-1])
        print(colored(f"Verifier score: {score}", "green"))
        return score
