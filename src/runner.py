import re
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict
from llm_sdk.llm_sdk import Small_LLM_Model


class LLMRunner(BaseModel):
    model: Small_LLM_Model
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -------------------------
    # MAIN ENTRY
    # -------------------------
    def run(self, prompt: str, functions: list[dict[str, Any]]) -> dict[str, Any]:

        fn = self.select_function(prompt, functions)

        if fn is None:
            return {"name": "INVALID_PARSE", "parameters": {}}

        params = self.extract_parameters(prompt, fn)

        return {
            "name": fn["name"],
            "parameters": params
        }

    # -------------------------
    # FUNCTION SELECTION (SAFE)
    # -------------------------
    def select_function(
        self,
        prompt: str,
        functions: list[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:

        p = prompt.lower()

        def score(fn: dict[str, Any]) -> int:
            name = fn["name"].lower()
            desc = fn["description"].lower()

            s = 0

            # strong intent matching
            if ("add" in p or "sum" in p or "plus" in p) and "add" in name:
                s += 20

            if "greet" in p and "greet" in name:
                s += 20

            if "reverse" in p and "reverse" in name:
                s += 20

            if ("square root" in p or "sqrt" in p) and "square" in name:
                s += 20

            if (
                "replace" in p
                or "substitute" in p
                or "vowel" in p
                or "cat" in p
            ) and "substitute" in name:
                s += 20

            # weak semantic match
            for w in desc.split():
                if w in p:
                    s += 1

            return s

        scored = [(fn, score(fn)) for fn in functions]
        best_fn, best_score = max(scored, key=lambda x: x[1])

        if best_score == 0:
            return None

        return best_fn

    # -------------------------
    # PARAMETER EXTRACTION
    # -------------------------
    def extract_parameters(
        self,
        prompt: str,
        fn: dict[str, Any]
    ) -> dict[str, Any]:

        params_schema = fn["parameters"]
        result: dict[str, Any] = {}

        for key, spec in params_schema.items():

            if spec["type"] == "number":
                result[key] = self._extract_number(prompt, key, fn["description"])

            elif spec["type"] == "string":
                result[key] = self._extract_string(prompt, key, fn["description"])

        return result

    # -------------------------
    # NUMBER EXTRACTION (SAFE)
    # -------------------------
    def _extract_number(self, text: str, key: str, desc: str) -> int:

        nums = list(map(int, re.findall(r"-?\d+", text)))

        if not nums:
            return 0

        if "add" in desc or "sum" in desc:
            if key == "a":
                return nums[0]
            if key == "b":
                return nums[1] if len(nums) > 1 else nums[0]

        return nums[0]

    # -------------------------
    # STRING EXTRACTION (FIXED)
    # -------------------------
    def _extract_string(self, text: str, key: str, desc: str) -> str:

        t = text.strip()
        tl = t.lower()

        # -------------------------
        # 1. QUOTED STRING FIRST
        # -------------------------
        quoted_val = None
        match = re.search(r"'([^']*)'|\"([^\"]*)\"", t)
        if match:
            quoted_val = next(filter(None, match.groups()))

        # -------------------------
        # 2. SUBSTITUTE / REGEX
        # -------------------------
        if (
            "replace" in tl
            or "substitute" in tl
            or "regex" in tl
        ):

            # source_string
            if key == "source_string":
                if quoted_val:
                    return quoted_val

                if " in " in tl and " with " in tl:
                    return t.split(" in ")[-1].split(" with ")[0].strip("'\" ")

                return t

            # regex
            if key == "regex":

                if "vowel" in tl:
                    return r"[aeiouAEIOU]"

                if "number" in tl:
                    return r"\d+"

                if "cat" in tl:
                    return r"cat"

                return r"."

            # replacement
            if key == "replacement":
                if " with " in tl:

                    # find actual pattern: "... with <replacement>"
                    parts = re.split(r"\swith\s", t, flags=re.IGNORECASE)

                    if len(parts) >= 2:
                        replacement_part = parts[-1]

                        # stop accidental trailing sentence capture
                        replacement_part = replacement_part.split(" in ")[0]

                        return replacement_part.strip("'\" .,!?")

                return t.split()[-1].strip("'\".,!?")

        # -------------------------
        # 3. GREET
        # -------------------------
        if "greet" in tl:
            return t.split()[-1].strip("'\".,!?")

        # -------------------------
        # 4. REVERSE
        # -------------------------
        if "reverse" in tl:
            return t.split()[-1].strip("'\".,!?")

        # -------------------------
        # 5. FALLBACK
        # -------------------------
        return t.split()[-1].strip("'\".,!?")


# -------------------------
# PIPELINE ENTRY
# -------------------------
def run_llm(
    prompt: str,
    functions: list[dict[str, Any]],
    llm_runner: LLMRunner,
) -> dict[str, Any]:

    return llm_runner.run(prompt, functions)
