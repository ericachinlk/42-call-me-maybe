from src.file_handler import load_functions, load_input, save_output
from src.llm_engine import LLMEngine
from src.function_selector import FunctionSelector
from src.parameter_extractor import ParameterExtractor
from src.validator import validate_parameters, PipelineError
from pathlib import Path
from typing import Any
import argparse
import time
import os

DEBUG = os.getenv("DEBUG") == "1"


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json")
    parser.add_argument(
        "--input",
        default="data/input/functions_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        pipeline_start = time.perf_counter()

        functions = load_functions(args.functions_definition)
        prompts = load_input(args.input)
        llm = LLMEngine()

        fn_map = {f.name: f for f in functions}
        extractor = ParameterExtractor()
        selector = FunctionSelector(llm=llm, functions=functions)

        results = []
        for prompt in prompts:
            fn_name = selector.select(prompt)

            if fn_name not in fn_map:
                raise PipelineError(
                    f"LLM hallucinated an invalid function name: '{fn_name}'. "
                    f"Available choices are: {list(fn_map.keys())}"
                )

            fn_def = fn_map[fn_name]
            params = extractor.extract(fn_def, prompt)
            validate_parameters(fn_def, params)
            results.append(
                {
                    "prompt": prompt,
                    "name": fn_name,
                    "parameters": params,
                }
            )

            if DEBUG:
                print("=" * 60)
                print("Prompt:", prompt)
                print("Selected Function:", fn_name)
                print("Parameters:", params)
                breakpoint()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_output(args.output, results)

        total_duration = (time.perf_counter() - pipeline_start) / 60
        print("\nPipeline Completed successfully!")
        print(f"Total Execution Time: {total_duration:.2f} minutes")

    except PipelineError as e:
        print("Pipeline Error:", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
