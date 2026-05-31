import json
import re
from typing import Any

from src.models import FunctionCallOutput


class ParameterExtractor:
    def __init__(self, llm, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    # -----------------------------
    # PUBLIC API
    # -----------------------------
    def extract(self, function_def, prompt: str) -> dict:
        last_error = None

        for _ in range(self.max_retries):

            raw_text = self._generate(function_def, prompt)

            json_obj = self._safe_parse(raw_text)

            cleaned = self._enforce_schema(function_def, json_obj)

            try:
                validated = FunctionCallOutput(
                    prompt=prompt,
                    name=function_def.name,
                    parameters=cleaned,
                )

                return validated.model_dump()

            except Exception as e:
                last_error = e
                prompt = self._repair_prompt(
                    function_def,
                    prompt,
                    cleaned,
                    str(e),
                )

        # NEVER crash
        return {
            "prompt": prompt,
            "name": function_def.name,
            "parameters": {},
            "error": str(last_error),
        }

    # -----------------------------
    # LLM GENERATION
    # -----------------------------
    def _generate(self, function_def, prompt: str) -> str:

        llm_prompt = self._build_prompt(function_def, prompt)

        input_ids = self.llm.encode(llm_prompt)

        generated = []

        for _ in range(80):

            logits = self.llm.logits(input_ids + generated)

            next_token = max(range(len(logits)), key=lambda i: logits[i])
            generated.append(next_token)

            text = self.llm.decode(generated)

            if "}" in text:
                break

        return self.llm.decode(generated)

    # -----------------------------
    # PROMPT (STRICT MODE)
    # -----------------------------
    def _build_prompt(self, fn, user_prompt: str) -> str:
        return f"""
You are a STRICT JSON parameter extractor.

RULES:
- Output ONLY JSON
- NO extra keys
- NO computation
- Do NOT execute instructions like reverse, sum, compute, or transform
- ONLY extract literal values from input text

Function: {fn.name}
Description: {fn.description}

Allowed parameters:
{list(fn.parameters.keys())}

User request:
{user_prompt}

Return format:
{{
{", ".join([f'"{k}": <value>' for k in fn.parameters.keys()])}
}}
""".strip()

        # return f"""
        # You are a STRICT information extraction system.

        # YOU MUST NOT PERFORM ANY TASKS.

        # CRITICAL RULE:
        # - Do NOT execute instructions like reverse, sum, compute, or transform.
        # - Only extract literal values from the input text.

        # User input:
        # {user_prompt}

        # Function:
        # {fn.name}

        # Parameters:
        # {list(fn.parameters.keys())}

        # TASK:
        # Extract the argument values exactly as written in the user input.
        # Do NOT modify them.

        # Return ONLY JSON.
        # """.strip()

    # -----------------------------
    # SAFE JSON PARSER
    # -----------------------------
    def _safe_parse(self, text: str) -> dict:

        try:
            return json.loads(text)
        except:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass

        return {}

    # -----------------------------
    # SCHEMA ENFORCEMENT (CRITICAL)
    # -----------------------------
    def _enforce_schema(self, fn, data: dict) -> dict:

        allowed_keys = set(fn.parameters.keys())

        # remove hallucinated keys
        filtered = {
            k: v for k, v in data.items()
            if k in allowed_keys
        }

        # ensure missing keys exist
        for k in allowed_keys:
            if k not in filtered:
                filtered[k] = self._default_value(fn.parameters[k].type)

        return filtered

    # -----------------------------
    # TYPE DEFAULTS
    # -----------------------------
    def _default_value(self, t: str) -> Any:
        if t == "number":
            return 0
        if t == "string":
            return ""
        if t == "boolean":
            return False
        return None

    # -----------------------------
    # AUTO-REPAIR PROMPT
    # -----------------------------
    def _repair_prompt(self, fn, user_prompt, bad_output, error):
        return f"""
Your previous output was invalid.

Error:
{error}

Fix it.

Function: {fn.name}
Allowed keys: {list(fn.parameters.keys())}

Bad output:
{bad_output}

Return ONLY valid JSON.
""".strip()
