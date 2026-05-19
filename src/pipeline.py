import json
from src.loader import load_json
from src.runner import LLMRunner, run_llm
from llm_sdk.llm_sdk import Small_LLM_Model


def run_pipeline(
    functions_path: str,
    input_path: str,
    output_path: str
) -> None:

    try:
        functions = load_json(functions_path)
        inputs = load_json(input_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load input files: {e}")

    model = Small_LLM_Model()
    llm_runner = LLMRunner(model)

    results = []

    if not functions or not inputs:
        raise RuntimeError("Empty functions or input file")

    for item in inputs:
        prompt = item.get("prompt", "")
        result = run_llm(prompt, functions, llm_runner)
        results.append({
            "prompt": prompt,
            "name": result.get("name", "INVALID_PARSE"),
            "parameters": result.get("parameters", {})
        })

    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write output file: {e}")
