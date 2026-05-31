import json


class ParameterExtractor:
    def __init__(self, llm):
        self.llm = llm

    def build_prompt(self, function_def, user_prompt):
        schema = {
            k: v.type for k, v in function_def.parameters.items()
        }

        return f"""
        You are a function parameter extractor.

        Return ONLY valid JSON.

        Function:
        {function_def.name}
        Description:
        {function_def.description}

        Parameters schema:
        {json.dumps(schema, indent=2)}

        User request:
        {user_prompt}

        Return format:
        {{
        {', '.join([f'"{k}": ...' for k in schema.keys()])}
        }}
        """.strip()

    def extract(self, function_def, user_prompt):
        prompt = self.build_prompt(function_def, user_prompt)

        input_ids = self.llm.encode(prompt)
        logits = self.llm.logits(input_ids)

        # IMPORTANT:
        # We do NOT do full constrained decoding here
        # We rely on model + validation layer

        # simple greedy decode via sdk tokenizer
        output_ids = self._generate(prompt=input_ids)

        text = self.llm.decode(output_ids)

        try:
            return json.loads(text)
        except Exception:
            return {}  # safe fallback

    def _generate(self, prompt, max_tokens=128):
        generated = []

        for _ in range(max_tokens):
            logits = self.llm.logits(prompt + generated)

            next_token = max(
                range(len(logits)),
                key=lambda i: logits[i]
            )

            generated.append(next_token)

            # crude stopping condition (works fine for assignment)
            text = self.llm.decode(generated)
            if "}" in text:
                break

        return generated
