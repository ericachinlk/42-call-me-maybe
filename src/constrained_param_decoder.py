import json
from pydantic import ValidationError
from src.models import FunctionCallOutput


class ConstrainedParamDecoder:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, function_def, prompt):
        llm_prompt = self._build_prompt(function_def, prompt)

        input_ids = self.llm.encode(llm_prompt)

        # generate output tokens (normal greedy decoding)
        output_tokens = self._generate(input_ids)

        text = self.llm.decode(output_tokens)

        # try parse JSON safely
        params = self._safe_json(text)

        return params
    
    def _build_prompt(self, fn, user_prompt):
        return f"""
        You are a parameter extraction system.

        CRITICAL RULES:
        - Do NOT perform any computation
        - Do NOT transform or modify values
        - Do NOT reverse, calculate, or interpret meaning
        - Only extract raw values from the input text

        Function: {fn.name}
        Description: {fn.description}

        Parameters:
        {fn.parameters}

        User request:
        {user_prompt}

        Return ONLY JSON with extracted values.
        """.strip()

    def _generate(self, input_ids, max_tokens=80):
        generated = []

        for _ in range(max_tokens):
            logits = self.llm.logits(input_ids + generated)

            next_token = max(range(len(logits)), key=lambda i: logits[i])
            generated.append(next_token)

            text = self.llm.decode(generated)

            if "}" in text:
                break

        return generated

    def _safe_json(self, text):
        try:
            return json.loads(text)
        except:
            # fallback: extract JSON block
            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end+1])
                except:
                    pass

        return {}
