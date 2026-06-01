import json
import re
from typing import Any
from src.models import FunctionCallOutput


class ParameterExtractor:
    def __init__(self, llm):
        self.llm = llm

    # -------------------------------------------------
    # MAIN ENTRY
    # -------------------------------------------------
    def extract(self, fn_def, prompt: str):
        params = {}

        numbers = self._get_numbers(prompt)
        strings = self._extract_strings(prompt)

        num_i = 0

        for param_name, param_type in fn_def.parameters.items():

            value = None
            name = param_name.lower()

            # -------------------------
            # NUMBER
            # -------------------------
            if param_type.type == "number":
                value = numbers[num_i] if num_i < len(numbers) else 0.0
                num_i += 1

            # -------------------------
            # STRING
            # -------------------------
            elif param_type.type == "string":

                if "source" in name or "input" in name:
                    value = self._get_source_string(prompt, strings)

                elif "replacement" in name:
                    value = self._extract_replacement(prompt)

                elif "regex" in name:
                    value = self._infer_regex(prompt)

                else:
                    value = strings[0] if strings else ""

            # fallback safety
            if value is None:
                value = self._default_value(name, param_type.type)

            params[param_name] = self._cast(value, param_type.type)

        return params

    # -------------------------------------------------
    # SOURCE STRING (robust)
    # -------------------------------------------------
    def _get_source_string(self, prompt, strings):
        if strings:
            return max(strings, key=len)

        return prompt

    # -------------------------------------------------
    # REPLACEMENT (FIXED - no leakage)
    # -------------------------------------------------
    def _extract_replacement(self, prompt: str):

        match = re.search(
            r"\bwith\s+['\"]?(.*?)['\"]?\s*(?:$|in\b)",
            prompt,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip(" '\"")

        return ""

    # -------------------------------------------------
    # REGEX INFERENCE (SAFE + COMPLETE)
    # -------------------------------------------------
    # def _infer_regex(self, prompt: str):
    #     p = prompt.lower()

    #     # -------------------------
    #     # 1. LITERAL WORD (HIGHEST PRIORITY)
    #     # -------------------------
    #     quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
    #     quoted = [a or b for a, b in quoted]

    #     if quoted:
    #         # assume first quoted token is target word
    #         return quoted[0]

    #     # -------------------------
    #     # 2. KEYWORD PATTERNS
    #     # -------------------------
    #     if "vowel" in p:
    #         return r"[aeiouAEIOU]"

    #     if "number" in p or "digit" in p:
    #         return r"\d+"

    #     if "space" in p:
    #         return r"\s+"

    #     # -------------------------
    #     # 3. FALLBACK (ONLY LAST RESORT)
    #     # -------------------------
    #     return r"\w+"
    
    def _infer_regex(self, prompt: str):

        p = prompt.lower()

        if "vowel" in p:
            return r"[aeiouAEIOU]"

        if "number" in p or "digit" in p:
            return r"\d+"

        if "space" in p:
            return r"\s+"

        if "word" in p:
            quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
            quoted = [a or b for a, b in quoted]

            if quoted:
                return quoted[0]
            # return r"\w+"

        return ""

    # -------------------------------------------------
    # NUMBER EXTRACTION
    # -------------------------------------------------
    def _get_numbers(self, prompt):
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", prompt)]

    # -------------------------------------------------
    # STRING EXTRACTION
    # -------------------------------------------------
    def _extract_strings(self, prompt: str):

        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
        strings = [a or b for a, b in quoted]

        if strings:
            return strings

        cleaned = re.sub(
            r"\b(greet|reverse|replace|calculate|sum|substitute)\b",
            "",
            prompt,
            flags=re.IGNORECASE
        )

        words = re.findall(r"[A-Za-z]+", cleaned)

        stop = {"what", "is", "the", "and", "with", "in", "to", "of"}

        return [w for w in words if w.lower() not in stop]

    # -------------------------------------------------
    # DEFAULTS
    # -------------------------------------------------
    def _default_value(self, name, t):

        if t == "number":
            return 0.0

        if t == "string":
            if "regex" in name:
                return ".*"
            return ""

        return None

    # -------------------------------------------------
    # CASTING (SAFE)
    # -------------------------------------------------
    def _cast(self, value, t):

        if value is None:
            return None

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
#     def __init__(self, llm):
#         self.llm = llm

#     def extract(self, fn_def, prompt: str):
#         params = {}

#         # extract raw signals once
#         # numbers = re.findall(r"-?\d+(?:\.\d+)?", prompt)
#         numbers = self._get_numbers(prompt)
#         num_i = 0

#         strings = self._extract_strings(prompt)

#         words = re.findall(r"[A-Za-z]+", prompt)

#         for param_name, param_type in fn_def.parameters.items():

#             value = None

#             # -------------------------
#             # NUMBER PARAMETERS
#             # -------------------------
#             # if param_type.type == "number":
#             #     value = numbers[0] if numbers else 0
#             if param_type.type == "number":
#                 value = numbers[num_i] if num_i < len(numbers) else 0
#                 num_i += 1

#             # -------------------------
#             # STRING PARAMETERS (ROLE-AWARE)
#             # -------------------------
#             elif param_type.type == "string":

#                 name = param_name.lower()

#                 # SOURCE STRING (main input text)
#                 if "source" in name or "input" in name:
#                     value = self._find_longest_phrase(strings, prompt)

#                 # REPLACEMENT TEXT
#                 elif "replacement" in name:
#                     # value = self._find_literal_after_keywords(prompt, ["with", "as"])
#                     value = self._extract_replacement(prompt)

#                 # REGEX / PATTERN
#                 elif "regex" in name:
#                     value = self._infer_regex(prompt)

#                 else:
#                     # value = strings[0] if strings else ""
#                     if value is None or value == "":
#                         value = self._default_value(param_name, param_type.type)
            
#             params[param_name] = self._cast(value, param_type.type)

#         print("RAW PROMPT:", prompt)
#         print("REPLACEMENT RAW SLICE:", prompt[prompt.lower().find("with"):])
#         print(repr(prompt))
#         print(self._extract_replacement(prompt))

#         return params
    
#     def _find_longest_phrase(self, strings, prompt):
#         if not strings:
#             return ""

#         # prefer longest quoted string
#         return max(strings, key=len)
    
#     def _find_literal_after_keywords(self, prompt, keywords):
#         for kw in keywords:
#             if kw in prompt.lower():
#                 parts = prompt.lower().split(kw)
#                 if len(parts) > 1:
#                     return parts[1].strip(" '\"")
#         return ""
    
#     def _infer_regex(self, prompt: str):
#         p = prompt.lower()

#         if "vowel" in p:
#             return r"[aeiouAEIOU]"

#         if "digit" in p:
#             return r"\d+"

#         if "space" in p:
#             return r"\s+"

#         return None
    
#     def _get_numbers(self, prompt):
#         return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", prompt)]
    
#     def _extract_replacement(self, prompt: str):
#         # match: "... with XYZ"
#         match = re.search(
#             r"\bwith\s+['\"]?(.*?)['\"]?\s*$",
#             prompt,
#             re.IGNORECASE
#         )

#         if match:
#             return match.group(1).strip()

#         return ""
    
#     def _default_value(self, name, t):
#         if t == "number":
#             return 0.0

#         if t == "string":

#             if "regex" in name:
#                 return ".*"   # safest neutral regex

#             return ""

#         return None

#     def _extract_strings(self, prompt: str) -> list[str]:

#         # 1. Extract quoted strings first (highest priority)
#         quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
#         strings = [a or b for a, b in quoted]

#         if strings:
#             return strings

#         # 2. Remove instruction words (very important)
#         cleaned = re.sub(
#             r"\b(greet|reverse|replace|calculate|sum|substitute|vowels|string)\b",
#             "",
#             prompt,
#             flags=re.IGNORECASE
#         )

#         # 3. Extract remaining words
#         words = re.findall(r"[A-Za-z]+", cleaned)

#         # 4. Filter stopwords
#         stopwords = {
#             "what", "is", "the", "a", "an", "and", "with", "to", "in", "of"
#         }

#         return [w for w in words if w.lower() not in stopwords]

#     # -------------------------------------------------
#     # Candidate extraction (STRICT + TYPE-AWARE)
#     # -------------------------------------------------
#     def _extract_candidates(self, prompt: str, param_type: str, param_name: str):

#         if param_type == "number":
#             return re.findall(r"-?\d+(?:\.\d+)?", prompt)

#         if param_type == "string":

#             # remove function verbs first (VERY IMPORTANT)
#             cleaned = re.sub(
#                 r"\b(greet|reverse|replace|calculate|sum|substitute)\b",
#                 "",
#                 prompt,
#                 flags=re.IGNORECASE
#             )

#             # extract quoted strings first
#             quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", cleaned)
#             quoted = [a or b for a, b in quoted]

#             if quoted:
#                 return quoted

#             # fallback: ONLY remaining meaningful words
#             words = re.findall(r"[A-Za-z]+", cleaned)

#             stop = {
#                 "what", "is", "the", "and", "with", "all", "in", "to"
#             }

#             return [w for w in words if w.lower() not in stop]

#     # -------------------------------------------------
#     # SAFE selection (NO execution, NO heuristics)
#     # -------------------------------------------------
#     def _select_best_candidate(self, candidates, param_type):

#         if not candidates:
#             return ""

#         # safest possible choice:
#         # do NOT let model "compute" or "transform"
#         return candidates[0]

#     # -------------------------------------------------
#     # TYPE CASTING (SAFE)
#     # -------------------------------------------------
#     def _cast(self, value, t):
#         if value is None:
#             return None

#         if t == "number":
#             try:
#                 return float(value)
#             except:
#                 return 0.0

#         if t == "string":
#             return str(value).strip("'\"")

#         if t == "boolean":
#             return str(value).lower() in {"true", "1", "yes"}

#         return value


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
