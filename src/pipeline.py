import json
from src.loader import load_json
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constrained_decoder import ConstrainedDecoder


def run_pipeline(
    functions_path: str,
    input_path: str,
    output_path: str
) -> None:

    functions = load_json(functions_path)
    inputs = load_json(input_path)

    if not functions:
        raise RuntimeError("Failed to load functions definition file")

    if not inputs:
        raise RuntimeError("Failed to load input file")

    model = Small_LLM_Model()
    decoder = ConstrainedDecoder(model, functions)

    results = []

    for item in inputs:
        prompt = item.get("prompt", "")

        # ✅ constrained decoding per prompt
        result_text = decoder.generate(prompt)

        # safe parse (now guaranteed valid JSON)
        try:
            result = json.loads(result_text)
        except Exception:
            result = {"name": "INVALID_PARSE", "parameters": {}}

        results.append({
            "prompt": prompt,
            "name": result.get("name", "INVALID_PARSE"),
            "parameters": result.get("parameters", {})
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
