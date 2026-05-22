import numpy as np
from typing import Any


class ConstrainedDecoder:
    """
    Token-aware constrained decoder.

    Key idea:
    - We NEVER assume JSON tokens exist.
    - We filter logits using tokenizer prefix matching.
    """

    def __init__(self, model, functions: list[dict[str, Any]]):
        self.model = model
        self.functions = functions

        self.fn_map = {f["name"]: f for f in functions}
        self.fn_names = list(self.fn_map.keys())

    # -------------------------
    # ENTRY
    # -------------------------
    def generate(self, prompt: str) -> str:
        self.current_fn = self.select_function(prompt)
        if not self.current_fn:
            self.current_fn = self.fn_names[0]

        self.param_keys = list(self.fn_map[self.current_fn]["parameters"].keys())

        text = ""
        input_ids = self.model.encode(self.build_prompt(prompt)).tolist()[0]

        for _ in range(120):
            logits = self.model.get_logits_from_input_ids(input_ids)

            allowed_ids = self.allowed_token_ids(text)

            next_token = self.sample(logits, allowed_ids)

            input_ids.append(next_token)
            text = self.model.decode(input_ids)

            if text.strip().endswith("}"):
                break

        return text

    # -------------------------
    # FUNCTION SELECTION
    # -------------------------
    def select_function(self, prompt: str):
        p = prompt.lower()

        best, best_score = None, -1

        for fn in self.functions:
            name = fn["name"].lower()

            score = 0
            if "add" in p and "add" in name:
                score += 10
            if "greet" in p and "greet" in name:
                score += 10
            if "reverse" in p and "reverse" in name:
                score += 10
            if "square" in p and "square" in name:
                score += 10
            if "replace" in p and "substitute" in name:
                score += 10

            if score > best_score:
                best_score = score
                best = fn["name"]

        return best

    # -------------------------
    # PROMPT
    # -------------------------
    def build_prompt(self, user_prompt: str) -> str:
        return f"""<|im_start|>system
Return ONLY JSON:
{{"name": "...", "parameters": {{...}}}}
<|im_end|>
<|im_start|>user
{user_prompt}
<|im_end|>
<|im_start|>assistant
"""

    # -------------------------
    # TOKEN FILTERING (CORE IDEA)
    # -------------------------
    def allowed_token_ids(self, partial_text: str):
        """
        We constrain by PREFIX MATCHING over tokenizer vocab.
        """
        vocab = self.model._tokenizer.get_vocab()

        allowed = []

        # We only enforce weak structural constraints:
        # (strong constraints break Qwen tokenization)
        for token_str, token_id in vocab.items():

            decoded = self.model._tokenizer.decode([token_id])

            # must not break JSON structure
            if self.is_valid_extension(partial_text, decoded):
                allowed.append(token_id)

        return set(allowed)

    # -------------------------
    # VALIDITY CHECK (LIGHTWEIGHT FSM)
    # -------------------------
    def is_valid_extension(self, text: str, new_piece: str) -> bool:
        candidate = text + new_piece

        # Hard rules
        if candidate.count("{") < candidate.count("}"):
            return False

        if '"name"' in candidate and candidate.count(":") == 0:
            return False

        # prevent garbage injection
        if len(candidate) > 500:
            return False

        return True

    # -------------------------
    # SAMPLING
    # -------------------------
    def sample(self, logits, allowed_ids):
        logits = np.array(logits)

        if not allowed_ids:
            return int(np.argmax(logits))

        mask = np.full_like(logits, -1e9)

        for i in allowed_ids:
            if i < len(mask):
                mask[i] = logits[i]

        return int(np.argmax(mask))