from src.file_handler import load_functions, load_input, save_output
from src.llm_engine import LLMEngine
from src.function_selector import FunctionSelector
from src.parameter_extractor import ParameterExtractor
from pathlib import Path
from typing import Any
import argparse


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.input).exists():
        raise FileNotFoundError(args.input)
    functions = load_functions(args.functions_definition)
    prompts = load_input(args.input)
    llm = LLMEngine()

    fn_map = {f.name: f for f in functions}
    extractor = ParameterExtractor(llm)
    selector = FunctionSelector(llm, functions)

    results = []
    for prompt in prompts:
        print("=" * 60)
        print("Prompt:", prompt)

        fn_name = selector.select(prompt)
        print("Selected Function:", fn_name)

        fn_def = fn_map[fn_name]
        params = extractor.extract(fn_def, prompt)
        print("Parameters:", params)

        results.append(
            {
                "prompt": prompt,
                "name": fn_name,
                "parameters": params,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_output(args.output, results)


if __name__ == "__main__":
    main()
