from src.file_handler import load_functions, load_input, save_output
from src.llm_engine import LLMEngine
from src.validator import PipelineError
from src.prompt_processor import PromptProcessor
from pathlib import Path
from typing import Any
import argparse
import time


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json")
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        pipeline_start = time.perf_counter()

        functions = load_functions(args.functions_definition)
        prompts = load_input(args.input)
        llm = LLMEngine()

        processor = PromptProcessor(prompts=prompts, functions_definition=functions, llm=llm)
        results = processor.process()

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
