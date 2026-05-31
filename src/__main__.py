import json

from src.file_handler import load_functions
from src.llm_engine import LLMEngine
from src.function_selector import FunctionSelector
from src.parameter_extractor import ParameterExtractor
from src.models import FunctionCallOutput
from src.validator import Validator


def main():

    llm = LLMEngine()

    functions = load_functions(
        "src/data/input/functions_definition.json"
    )

    fn_map = {f.name: f for f in functions}

    selector = FunctionSelector(llm, functions)
    extractor = ParameterExtractor(llm)
    validator = Validator()

    results = []

    with open("src/data/input/function_calling_tests.json") as f:
        tests = json.load(f)

    for item in tests:

        prompt = item["prompt"]

        # -------------------------
        # STEP 1: FUNCTION SELECTION
        # -------------------------
        fn_name = selector.select_function(prompt)
        fn_def = fn_map[fn_name]

        # -------------------------
        # STEP 2: PARAMETER EXTRACTION
        # -------------------------
        params = extractor.extract(fn_def, prompt)

        # -------------------------
        # STEP 3: VALIDATION (PYDANTIC)
        # -------------------------
        validated = validator.validate(
            prompt,
            fn_name,
            params
        )

        results.append(validated)

    # -------------------------
    # OUTPUT FILE
    # -------------------------
    with open(
        "src/data/output/function_calls.json",
        "w"
    ) as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()


# import json
# from src.llm_engine import LLMEngine
# from src.functions import load_functions, build_function_map
# from src.schema import Schema
# from src.decoder import ConstrainedDecoder


# def main():
#     llm = LLMEngine()

#     functions = load_functions("src/data/input/functions_definition.json")
#     fn_map = build_function_map(functions)

#     schema = Schema(fn_map)
#     decoder = ConstrainedDecoder(llm, schema)

#     results = []

#     for item in load_functions("src/data/input/function_calling_tests.json"):

#         prompt = item["prompt"]
#         input_ids = llm.encode(prompt)

#         generated = []

#         # -------------------------
#         # STEP 1: FUNCTION NAME
#         # -------------------------
#         while True:
#             logits = llm.logits(input_ids + generated)

#             allowed = schema.allowed_function_tokens(llm, generated)

#             decoder.mask(logits, allowed)

#             next_token = decoder.pick(logits)

#             generated.append(next_token)

#             # stop when full function name matched
#             for fn in fn_map:
#                 if generated == llm.encode(fn):
#                     function_name = fn
#                     break
#             else:
#                 continue
#             break

#         # -------------------------
#         # STEP 2: BUILD OUTPUT JSON (simplified)
#         # -------------------------
#         result = {
#             "prompt": prompt,
#             "name": function_name,
#             "parameters": {}
#         }

#         fn_def = fn_map[function_name]

#         # naive parameter extraction (simplified for assignment base version)
#         for param in fn_def["parameters"]:
#             if fn_def["parameters"][param]["type"] == "number":
#                 result["parameters"][param] = 0
#             else:
#                 result["parameters"][param] = ""

#         results.append(result)

#     # -------------------------
#     # WRITE OUTPUT
#     # -------------------------
#     with open("src/data/output/function_calls.json", "w") as f:
#         json.dump(results, f, indent=2)


# if __name__ == "__main__":
#     main()


# import argparse
# from pathlib import Path
# from src.pipeline import run_pipeline


# def main() -> None:
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--functions_definition", required=True)
#     parser.add_argument("--input", required=True)
#     parser.add_argument("--output", required=True)

#     args = parser.parse_args()

#     # create parent directory for output file
#     output_path = Path(args.output)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     run_pipeline(
#         functions_path=args.functions_definition,
#         input_path=args.input,
#         output_path=args.output
#     )


# if __name__ == "__main__":
#     main()
