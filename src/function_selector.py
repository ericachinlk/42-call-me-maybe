from src.models import FunctionDefinition
from src.llm_engine import LLMEngine
import numpy as np


# class TrieNode:
#     def __init__(self):
#         self.children = {}
#         self.function_name = None


# class FunctionSelector:
#     def __init__(self, llm, functions):
#         self.llm = llm
#         self.functions = functions

#         self.func_tokens = {
#             f.name: self._norm(llm.encode(f.name))
#             for f in functions
#         }

#         self.trie = self._build_trie(self.func_tokens)

#     def _norm(self, x):
#         return x if isinstance(x, list) else x.tolist()

#     def _build_trie(self, func_tokens):
#         root = TrieNode()

#         for name, tokens in func_tokens.items():
#             node = root
#             for t in tokens:
#                 if t not in node.children:
#                     node.children[t] = TrieNode()
#                 node = node.children[t]
#             node.function_name = name

#         return root
    
#     def select(self, prompt: str):

#         input_ids = self._norm(self.llm.encode(prompt))

#         node = self.trie
#         generated = []

#         while True:

#             # if leaf reached → return immediately
#             if node.function_name is not None:
#                 return node.function_name

#             # only valid next tokens exist in trie
#             allowed_tokens = list(node.children.keys())

#             # we STILL use logits only as tie-breaker (optional)
#             logits = self.llm.logits(input_ids + generated)

#             # HARD MASK invalid tokens
#             masked = {t: logits[t] for t in allowed_tokens}

#             # deterministic pick (NO ambiguity)
#             next_token = max(masked, key=masked.get)

#             generated.append(next_token)

#             # move in trie
#             node = node.children[next_token]


# class FunctionSelector:
#     def __init__(self, llm, functions):
#         self.llm = llm
#         self.functions = functions

#         # tokenized function names
#         self.func_tokens = {
#             f.name: llm.encode(f.name)
#             for f in functions
#         }

#     # -------------------------------------------------
#     # MAIN ENTRY
#     # -------------------------------------------------
#     def select(self, prompt: str):

#         input_ids = self.llm.encode(prompt)
#         generated = []

#         candidates = list(self.func_tokens.keys())

#         while True:

#             # 1. prune FIRST (not after generation)
#             candidates = [
#                 name for name, seq in self.func_tokens.items()
#                 if seq[:len(generated)] == generated
#             ]

#             if len(candidates) == 1:
#                 return candidates[0]

#             if len(candidates) == 0:
#                 raise RuntimeError("No valid function match")

#             # 2. compute allowed tokens ONLY from valid candidates
#             allowed = set()

#             for name in candidates:
#                 seq = self.func_tokens[name]
#                 if len(seq) > len(generated):
#                     allowed.add(seq[len(generated)])

#             # 3. logits
#             logits = self.llm.logits(input_ids + generated)

#             masked = [-float("inf")] * len(logits)

#             for i in allowed:
#                 masked[i] = logits[i]

#             # 4. pick token
#             next_token = max(range(len(logits)), key=lambda i: masked[i])

#             generated.append(next_token)

#     # -------------------------------------------------
#     # PREFIX MATCHING
#     # -------------------------------------------------
#     def _matches_prefix(self, seq, generated):
#         if len(generated) > len(seq):
#             return False
#         return seq[:len(generated)] == generated

#     # -------------------------------------------------
#     # ALLOWED TOKENS BUILDER
#     # -------------------------------------------------
#     def _allowed_next_tokens(self, candidates, generated):

#         allowed = None

#         for name in candidates:
#             seq = self.func_tokens[name]

#             if len(seq) <= len(generated):
#                 continue

#             next_token = seq[len(generated)]

#             if allowed is None:
#                 allowed = {next_token}
#             else:
#                 allowed &= {next_token}

#         return allowed if allowed else set()


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

    # def build_prompt(self, user_prompt: str) -> str:
    #     lines = ["Available functions:\n"]

    #     for fn in self.functions:
    #         lines.append(f"Function: {fn.name}")
    #         lines.append(f"Description: {fn.description}")
    #         lines.append("")

    #     lines.append(f"User request: {user_prompt}")
    #     lines.append("")
    #     lines.append("Function:")

    #     return "\n".join(lines)
    
    def build_prompt(self, user_prompt: str) -> str:
        lines = []

        lines.append("You are a function selection system.")
        lines.append("Select EXACTLY ONE function that best matches the user request.")
        lines.append("Do NOT execute the function.")
        lines.append("Only output the function name.")
        lines.append("")

        lines.append("Available functions:")
        lines.append("")

        for fn in self.functions:
            lines.append(f"Name: {fn.name}")
            lines.append(f"Description: {fn.description}")

            # include parameters (VERY IMPORTANT)
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

    def select(
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


