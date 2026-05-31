from src.models import FunctionDefinition
from src.llm_engine import LLMEngine


class FunctionSelector:
    def __init__(
        self,
        llm: LLMEngine,
        functions: list[FunctionDefinition],
    ):
        self.llm = llm
        self.functions = functions

        self.function_tokens = {
            fn.name: self.llm.encode(fn.name)
            for fn in functions
        }

    def build_prompt(self, user_prompt: str) -> str:
        lines = ["Available functions:\n"]

        for fn in self.functions:
            lines.append(f"Function: {fn.name}")
            lines.append(f"Description: {fn.description}")
            lines.append("")

        lines.append(f"User request: {user_prompt}")
        lines.append("")
        lines.append("Function:")

        return "\n".join(lines)

    def _allowed_tokens(
        self,
        generated: list[int],
    ) -> set[int]:
        allowed = set()

        for seq in self.function_tokens.values():

            if len(seq) <= len(generated):
                continue

            if seq[: len(generated)] == generated:
                allowed.add(seq[len(generated)])

        return allowed

    def select_function(
        self,
        user_prompt: str,
    ) -> str:

        prompt = self.build_prompt(user_prompt)

        input_ids = self.llm.encode(prompt)

        generated = []

        while True:

            logits = self.llm.logits(
                input_ids + generated
            )

            allowed = self._allowed_tokens(generated)

            for i in range(len(logits)):
                if i not in allowed:
                    logits[i] = float("-inf")

            next_token = max(
                range(len(logits)),
                key=lambda x: logits[x],
            )

            generated.append(next_token)

            for fn_name, seq in self.function_tokens.items():
                if generated == seq:
                    return fn_name