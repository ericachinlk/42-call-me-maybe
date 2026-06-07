from src.models import FunctionDefinition, TestPrompt
from src.llm_engine import LLMEngine
from typing import Any
from pydantic import BaseModel, PrivateAttr


class FunctionSelector(BaseModel):
    llm: LLMEngine
    functions: list[FunctionDefinition]
    _function_tokens: dict[str, list[int]] = PrivateAttr()
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        self._function_tokens = {
            fn.name: self.llm.encode(fn.name)
            for fn in self.functions
        }

    def build_prompt(self, user_prompt: TestPrompt) -> str:
        lines = []
        lines.append("You are a function selection system.")
        lines.append(
            "Select EXACTLY ONE function that best matches the user request.")
        lines.append("Do NOT execute the function.")
        lines.append("Only output the function name.")
        lines.append("")
        lines.append("Available functions:")
        lines.append("")

        for fn in self.functions:
            lines.append(f"Name: {fn.name}")
            lines.append(f"Description: {fn.description}")
            lines.append("Parameters:")
            for param_name, param in fn.parameters.items():
                lines.append(f"  - {param_name}: {param.type}")
            lines.append("")

        lines.append("User request:")
        lines.append(user_prompt)
        lines.append("")
        lines.append("Return ONLY the function name:")
        lines.append("")
        return "\n".join(lines)

    def _allowed_tokens(self, generated: list[int]) -> set[int]:
        allowed = set()
        for seq in self._function_tokens.values():
            if len(seq) <= len(generated):
                continue
            if seq[: len(generated)] == generated:
                allowed.add(seq[len(generated)])
        return allowed

    def select(self, user_prompt: TestPrompt) -> Any:
        prompt = self.build_prompt(user_prompt)
        input_ids = self.llm.encode(prompt)
        generated: list[int] = []

        while True:
            logits = self.llm.logits(input_ids + generated)
            allowed = self._allowed_tokens(generated)
            for i in range(len(logits)):
                if i not in allowed:
                    logits[i] = float("-inf")
            next_token = max(range(len(logits)), key=lambda x: logits[x])
            generated.append(next_token)

            for fn_name, seq in self._function_tokens.items():
                if generated == seq:
                    return fn_name
