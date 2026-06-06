import json
from typing import Any
from pydantic import BaseModel, PrivateAttr
from src.models import FunctionDefinition, ParameterType
from src.llm_engine import LLMEngine


class JSONConstrainedDecoder(BaseModel):
    llm: LLMEngine
    _id_to_token: dict[int, str] = PrivateAttr()
    _all_tokens: set[int] = PrivateAttr()
    _lbrace: set[int] = PrivateAttr()
    _rbrace: set[int] = PrivateAttr()
    _quote: set[int] = PrivateAttr()
    _colon: set[int] = PrivateAttr()
    _comma: set[int] = PrivateAttr()
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        vocab_path = self.llm.model.get_path_to_vocab_file()

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self._id_to_token = {
            int(v): k.replace("Ġ", " ").replace("Ċ", "\n")
            for k, v in vocab.items()}

        self._all_tokens = set(self._id_to_token.keys())
        self._lbrace = {i for i, t in self._id_to_token.items() if "{" in t}
        self._rbrace = {i for i, t in self._id_to_token.items() if "}" in t}
        self._quote = {i for i, t in self._id_to_token.items() if '"' in t}
        self._colon = {i for i, t in self._id_to_token.items() if ":" in t}
        self._comma = {i for i, t in self._id_to_token.items() if "," in t}

    def decode(self, fn_def: FunctionDefinition, user_prompt: str) -> dict[str, Any]:
        prompt = self._build_prompt(fn_def, user_prompt)
        input_ids = self.llm.encode(prompt)

        generated = []

        brace_balance = 0
        started = False
        finished = False

        max_steps = 128

        for _ in range(max_steps):

            logits = self.llm.logits(input_ids + generated)

            allowed = self._allowed_tokens(fn_def, generated)

            if finished:
                break

            next_token = max(allowed, key=lambda i: logits[i])

            if next_token not in allowed:
                break

            generated.append(next_token)

            token_str = self._id_to_token[next_token]

            brace_balance += token_str.count("{")
            brace_balance -= token_str.count("}")

            if "{" in token_str:
                started = True

            if started and brace_balance == 0:
                finished = True

        text = self.llm.decode(generated)

        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON start found")

        json_str = text[start:].strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # fallback: recover only first valid JSON object
            import re
            match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if not match:
                raise

            return json.loads(match.group(0))

    def _allowed_tokens(self, fn_def: FunctionDefinition, generated: list[int]) -> set[int]:

        if not generated:
            return self._lbrace

        # always allow completion
        return self._all_tokens

    def _build_prompt(self, fn_def: FunctionDefinition, user_prompt: str) -> str:
        return (
            "Return ONLY valid JSON.\n"
            f"Function: {fn_def.name}\n"
            f"{fn_def.description}\n"
            f"Input: {user_prompt}\n"
            "JSON:\n{"
        )