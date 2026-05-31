import json
import re
from typing import Any
from src.models import FunctionCallOutput


class ParameterExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, fn_def, prompt: str):
        params = {}

        numbers = re.findall(r"-?\d+(?:\.\d+)?", prompt)
        strings = self._extract_strings(prompt)

        param_items = list(fn_def.parameters.items())

        for i, (param_name, param_type) in enumerate(param_items):

            if param_type.type == "number":
                value = numbers[i] if i < len(numbers) else 0.0

            elif param_type.type == "string":
                value = strings[0] if strings else ""

            else:
                value = None

            params[param_name] = self._cast(value, param_type.type)

        return params

    def _extract_strings(self, prompt: str) -> list[str]:

        # 1. Extract quoted strings first (highest priority)
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
        strings = [a or b for a, b in quoted]

        if strings:
            return strings

        # 2. Remove instruction words (very important)
        cleaned = re.sub(
            r"\b(greet|reverse|replace|calculate|sum|substitute|vowels|string)\b",
            "",
            prompt,
            flags=re.IGNORECASE
        )

        # 3. Extract remaining words
        words = re.findall(r"[A-Za-z]+", cleaned)

        # 4. Filter stopwords
        stopwords = {
            "what", "is", "the", "a", "an", "and", "with", "to", "in", "of"
        }

        return [w for w in words if w.lower() not in stopwords]

    # def extract(self, fn_def, prompt: str):

    #     params = {}

    #     for param_name, param_type in fn_def.parameters.items():

    #         candidates = self._extract_candidates(
    #             prompt,
    #             param_type.type,
    #             param_name
    #         )

    #         value = self._select_best_candidate(
    #             candidates,
    #             param_type.type
    #         )

    #         params[param_name] = self._cast(value, param_type.type)

    #     return params

    # -------------------------------------------------
    # Candidate extraction (STRICT + TYPE-AWARE)
    # -------------------------------------------------
    def _extract_candidates(self, prompt: str, param_type: str, param_name: str):

        if param_type == "number":
            return re.findall(r"-?\d+(?:\.\d+)?", prompt)

        if param_type == "string":

            # remove function verbs first (VERY IMPORTANT)
            cleaned = re.sub(
                r"\b(greet|reverse|replace|calculate|sum|substitute)\b",
                "",
                prompt,
                flags=re.IGNORECASE
            )

            # extract quoted strings first
            quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", cleaned)
            quoted = [a or b for a, b in quoted]

            if quoted:
                return quoted

            # fallback: ONLY remaining meaningful words
            words = re.findall(r"[A-Za-z]+", cleaned)

            stop = {
                "what", "is", "the", "and", "with", "all", "in", "to"
            }

            return [w for w in words if w.lower() not in stop]

    # -------------------------------------------------
    # SAFE selection (NO execution, NO heuristics)
    # -------------------------------------------------
    def _select_best_candidate(self, candidates, param_type):

        if not candidates:
            return ""

        # safest possible choice:
        # do NOT let model "compute" or "transform"
        return candidates[0]

    # -------------------------------------------------
    # TYPE CASTING (SAFE)
    # -------------------------------------------------
    def _cast(self, value, t):

        if t == "number":
            try:
                return float(value)
            except:
                return 0.0

        if t == "string":
            return str(value).strip("'\"")

        if t == "boolean":
            return str(value).lower() in {"true", "1", "yes"}

        return value


# class ParameterExtractor:
#     def __init__(self, llm, max_retries: int = 2):
#         self.llm = llm
#         self.max_retries = max_retries

#     # -----------------------------
#     # PUBLIC API
#     # -----------------------------
#     def extract(self, function_def, prompt: str) -> dict:
#         last_error = None

#         for _ in range(self.max_retries):

#             raw_text = self._generate(function_def, prompt)

#             json_obj = self._safe_parse(raw_text)

#             cleaned = self._enforce_schema(function_def, json_obj)

#             try:
#                 validated = FunctionCallOutput(
#                     prompt=prompt,
#                     name=function_def.name,
#                     parameters=cleaned,
#                 )

#                 return validated.model_dump()

#             except Exception as e:
#                 last_error = e
#                 prompt = self._repair_prompt(
#                     function_def,
#                     prompt,
#                     cleaned,
#                     str(e),
#                 )

#         # NEVER crash
#         return {
#             "prompt": prompt,
#             "name": function_def.name,
#             "parameters": {},
#             "error": str(last_error),
#         }

#     # -----------------------------
#     # LLM GENERATION
#     # -----------------------------
#     def _generate(self, function_def, prompt: str) -> str:

#         llm_prompt = self._build_prompt(function_def, prompt)

#         input_ids = self.llm.encode(llm_prompt)

#         generated = []

#         for _ in range(80):

#             logits = self.llm.logits(input_ids + generated)

#             next_token = max(range(len(logits)), key=lambda i: logits[i])
#             generated.append(next_token)

#             text = self.llm.decode(generated)

#             if "}" in text:
#                 break

#         return self.llm.decode(generated)

#     # -----------------------------
#     # PROMPT (STRICT MODE)
#     # -----------------------------
#     def _build_prompt(self, fn, user_prompt: str) -> str:
#         return f"""
# You are a STRICT JSON parameter extractor.

# RULES:
# - Output ONLY JSON
# - NO extra keys
# - NO computation
# - Do NOT execute instructions like reverse, sum, compute, or transform
# - ONLY extract literal values from input text

# Function: {fn.name}
# Description: {fn.description}

# Allowed parameters:
# {list(fn.parameters.keys())}

# User request:
# {user_prompt}

# Return format:
# {{
# {", ".join([f'"{k}": <value>' for k in fn.parameters.keys()])}
# }}
# """.strip()

#         # return f"""
#         # You are a STRICT information extraction system.

#         # YOU MUST NOT PERFORM ANY TASKS.

#         # CRITICAL RULE:
#         # - Do NOT execute instructions like reverse, sum, compute, or transform.
#         # - Only extract literal values from the input text.

#         # User input:
#         # {user_prompt}

#         # Function:
#         # {fn.name}

#         # Parameters:
#         # {list(fn.parameters.keys())}

#         # TASK:
#         # Extract the argument values exactly as written in the user input.
#         # Do NOT modify them.

#         # Return ONLY JSON.
#         # """.strip()

#     # -----------------------------
#     # SAFE JSON PARSER
#     # -----------------------------
#     def _safe_parse(self, text: str) -> dict:

#         try:
#             return json.loads(text)
#         except:
#             match = re.search(r"\{.*\}", text, re.DOTALL)
#             if match:
#                 try:
#                     return json.loads(match.group())
#                 except:
#                     pass

#         return {}

#     # -----------------------------
#     # SCHEMA ENFORCEMENT (CRITICAL)
#     # -----------------------------
#     def _enforce_schema(self, fn, data: dict) -> dict:

#         allowed_keys = set(fn.parameters.keys())

#         # remove hallucinated keys
#         filtered = {
#             k: v for k, v in data.items()
#             if k in allowed_keys
#         }

#         # ensure missing keys exist
#         for k in allowed_keys:
#             if k not in filtered:
#                 filtered[k] = self._default_value(fn.parameters[k].type)

#         return filtered

#     # -----------------------------
#     # TYPE DEFAULTS
#     # -----------------------------
#     def _default_value(self, t: str) -> Any:
#         if t == "number":
#             return 0
#         if t == "string":
#             return ""
#         if t == "boolean":
#             return False
#         return None

#     # -----------------------------
#     # AUTO-REPAIR PROMPT
#     # -----------------------------
#     def _repair_prompt(self, fn, user_prompt, bad_output, error):
#         return f"""
# Your previous output was invalid.

# Error:
# {error}

# Fix it.

# Function: {fn.name}
# Allowed keys: {list(fn.parameters.keys())}

# Bad output:
# {bad_output}

# Return ONLY valid JSON.
# """.strip()
