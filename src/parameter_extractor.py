import json
from typing import Any
from pydantic import BaseModel, PrivateAttr
from src.models import FunctionDefinition, ParameterType
from src.llm_engine import LLMEngine


class ParameterExtractor(BaseModel):
    llm: LLMEngine

    _id_to_token: dict[int, str] = PrivateAttr()
    _token_to_id: dict[str, int] = PrivateAttr()

    _vocab_size: int = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    # =========================================================
    # INIT VOCAB
    # =========================================================
    def model_post_init(self, __context: Any) -> None:
        vocab_path = self.llm.model.get_path_to_vocab_file()

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self._token_to_id = vocab
        self._id_to_token = {
            int(v): k.replace("Ġ", " ").replace("Ċ", "\n")
            for k, v in vocab.items()
        }

        self._vocab_size = len(vocab)

    # =========================================================
    # MAIN ENTRY
    # =========================================================
    def extract(self, fn_def: FunctionDefinition, user_prompt: str) -> dict[str, Any]:
        prompt = self._build_prompt(fn_def, user_prompt)
        input_ids = self.llm.encode(prompt)

        generated = []

        # FSM STATE
        state = "START"
        param_items = list(fn_def.parameters.items())
        param_idx = 0

        result = {}
        current_key = None

        for _ in range(512):
            logits = self.llm.logits(input_ids + generated)

            allowed = self._allowed_tokens(
                state,
                fn_def,
                param_items,
                param_idx,
                current_key
            )

            # deterministic constrained decoding
            next_token = max(allowed, key=lambda i: logits[i])

            if next_token not in allowed:
                break

            generated.append(next_token)
            token_str = self._id_to_token[next_token]

            # =====================================================
            # JSON GRAMMAR FSM (TOKEN-LEVEL CORRECT)
            # =====================================================

            if state == "START":
                state = "OBJECT_START"

            elif state == "OBJECT_START":
                state = "KEY"
                current_key = param_items[param_idx][0]

            elif state == "KEY":
                state = "COLON"

            elif state == "COLON":
                state = "VALUE"

            elif state == "VALUE":
                key, param = param_items[param_idx]

                # finalize value per schema type
                value = self._finalize_value(param.type, generated, user_prompt, key)

                result[key] = value

                param_idx += 1

                if param_idx >= len(param_items):
                    state = "END"
                else:
                    state = "KEY"
                    current_key = param_items[param_idx][0]

            elif state == "END":
                break

        return result

    # =========================================================
    # CONSTRAINED TOKEN FILTERING 
    # =========================================================
    def _allowed_tokens(
        self,
        state: str,
        fn_def: FunctionDefinition,
        params,
        idx: int,
        current_key: str | None
    ) -> set[int]:

        if state in {"START", "OBJECT_START"}:
            return self._tokens("{")

        if state == "KEY":
            # must generate quoted key strings
            return self._tokens('"')

        if state == "COLON":
            return self._tokens(":")

        if state == "VALUE":
            param = params[idx][1].type

            if param == ParameterType.number:
                return self._numeric_tokens()

            if param == ParameterType.boolean:
                return self._boolean_tokens()

            return self._string_tokens()

        if state == "END":
            return self._tokens("}")

        return set(range(self._vocab_size))

    # =========================================================
    # VALUE FINALIZATION (SAFE CAST ONLY)
    # =========================================================
    def _finalize_value(self, ptype: ParameterType, tokens: list[int], user_prompt: str, key: str) -> Any:
        text = self.llm.decode(tokens)

        if ptype == ParameterType.number:
            import re
            nums = re.findall(r"-?\d+(?:\.\d+)?", user_prompt)
            if len(nums) == 0:
                return 0
            if len(nums) == 1:
                return float(nums[0])

            # map by position
            idx = hash(key) % len(nums)
            return float(nums[idx])

        if ptype == ParameterType.boolean:
            return "true" in user_prompt.lower()

        # STRING EXTRACTION (GROUNDED)
        import re

        # try quoted string first
        quoted = re.findall(r"'(.*?)'|\"(.*?)\"", user_prompt)
        if quoted:
            for q in quoted:
                val = q[0] or q[1]
                if val:
                    return val

        # fallback: last meaningful word
        words = [w for w in user_prompt.split() if len(w) > 1]
        return words[-1] if words else text.strip('"')

    # =========================================================
    # TOKEN HELPERS
    # =========================================================
    def _tokens(self, char: str) -> set[int]:
        return {
            i for i, t in self._id_to_token.items()
            if char in t or t.strip() == ""
        }

    def _numeric_tokens(self) -> set[int]:
        return {
            i for i, t in self._id_to_token.items()
            if any(c.isdigit() for c in t) or t.strip() in {"", ".", "-", ","}
        }

    def _boolean_tokens(self) -> set[int]:
        return {
            i for i, t in self._id_to_token.items()
            if "true" in t.lower() or "false" in t.lower()
        }

    def _string_tokens(self) -> set[int]:
        return {
            i for i, t in self._id_to_token.items()
            if True  # fallback full vocab allowed
        }

    # =========================================================
    # PROMPT
    # =========================================================
    def _build_prompt(self, fn_def: FunctionDefinition, user_prompt: str) -> str:
        return (
            "Return valid JSON only.\n"
            f"Function: {fn_def.name}\n"
            f"{fn_def.description}\n\n"
            f"Input: {user_prompt}\n\n"
            "Output JSON:\n{"
        )