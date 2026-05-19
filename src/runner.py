import json
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model


class LLMRunner:
    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model

    def build_prompt(
            self,
            user_prompt: str,
            functions: list[dict[str, str]]
    ) -> str:
        fn_list = "\n".join(
            [f"- {f['name']}: {f['description']}" for f in functions]
        )

        return f"""<|im_start|>system
You are a function-calling system that outputs ONLY valid JSON.
Return strictly in this format:
{{"name": "...", "parameters": {{...}}}}
<|im_end|>
<|im_start|>user
Available functions:
{fn_list}

User request:
{user_prompt}
<|im_end|>
<|im_start|>assistant
"""

    def generate(self, prompt: str) -> str:
        import numpy as np

        output_ids = self.model.encode(prompt).tolist()[0]
        for step in range(25):
            logits = self.model.get_logits_from_input_ids(output_ids)

            next_token = int(np.argmax(logits))
            output_ids.append(next_token)

            if step % 5 == 0:
                text = self.model.decode(output_ids)

                if "}" in text and len(text) > 50:
                    break

        return self.model.decode(output_ids)

    def extract_json(self, text: str) -> dict[str, Any] | None:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == -1:
                return None

            return json.loads(text[start:end])
        except Exception:
            return None

    def safe_parse(self, parsed: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            return {"name": "INVALID_PARSE", "parameters": {}}

        name = parsed.get("name")
        params = parsed.get("parameters")

        if not isinstance(name, str) or not isinstance(params, dict):
            return {"name": "INVALID_PARSE", "parameters": {}}

        return {"name": name, "parameters": params}


def run_llm(
    prompt: str,
    functions: list[dict[str, str]],
    llm_runner: LLMRunner,
) -> dict[str, Any]:

    full_prompt = llm_runner.build_prompt(prompt, functions)
    raw_output = llm_runner.generate(full_prompt)

    print("MODEL OUTPUT:\n", raw_output)

    parsed = llm_runner.extract_json(raw_output)
    result = llm_runner.safe_parse(parsed)

    return result
