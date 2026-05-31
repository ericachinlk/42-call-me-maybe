from src.llm_engine import LLMEngine
from src.file_handler import load_functions
from src.function_selector import FunctionSelector
from src.parameter_extractor import ParameterExtractor


def main():

    llm = LLMEngine()
    functions = load_functions("src/data/input/functions_definition.json")

    fn_map = {f.name: f for f in functions}

    decoder = ParameterExtractor(llm)
    selector = FunctionSelector(llm, functions,)

    tests = [
        ("What is the sum of 265 and 345?", "fn_add_numbers"),
        ("Greet shrek", "fn_greet"),
        ("Reverse the string 'hello'", "fn_reverse_string"),
    ]

    # for prompt, fn_name in tests:

    #     print("=" * 60)
    #     print("Prompt:", prompt)
    #     print("Function:", fn_name)

    #     params = decoder.extract(fn_map[fn_name], prompt)

    #     print("Parameters:", params)
    
    for prompt, expected_fn in tests:
        print("=" * 60)
        print("Prompt:", prompt)

        # STEP 1: predicted function (REALISTIC TEST)
        fn_name = selector.select_function(prompt)

        print("Predicted Function:", fn_name)

        fn_def = fn_map[fn_name]

        # STEP 2: parameters
        params = decoder.extract(fn_def, prompt)

        print("Parameters:", params['parameters'])


if __name__ == "__main__":
    main()


# from src.llm_engine import LLMEngine
# from src.file_handler import load_functions
# from src.parameter_extractor import ParameterExtractor
# from src.models import FunctionDefinition


# def main():

#     print("Loading model...")
#     llm = LLMEngine()
#     print("Model loaded.\n")

#     # -------------------------
#     # Load function definitions
#     # -------------------------
#     functions = load_functions(
#         "src/data/input/functions_definition.json"
#     )

#     fn_map = {f.name: f for f in functions}

#     extractor = ParameterExtractor(llm)

#     # -------------------------
#     # TEST CASES
#     # -------------------------
#     tests = [
#         {
#             "prompt": "What is the sum of 2 and 3?",
#             "function": "fn_add_numbers",
#         },
#         {
#             "prompt": "What is the sum of 265 and 345?",
#             "function": "fn_add_numbers",
#         },
#         {
#             "prompt": "Greet shrek",
#             "function": "fn_greet",
#         },
#         {
#             "prompt": "Reverse the string 'hello'",
#             "function": "fn_reverse_string",
#         },
#         {
#             "prompt": "Calculate the square root of 144",
#             "function": "fn_get_square_root",
#         },
#         {
#             "prompt": "Replace vowels in Programming is fun with *",
#             "function": "fn_substitute_string_with_regex",
#         },
#     ]

#     # -------------------------
#     # RUN TESTS
#     # -------------------------
#     for test in tests:

#         prompt = test["prompt"]
#         fn_name = test["function"]
#         fn_def = fn_map[fn_name]

#         print("=" * 60)
#         print(f"Prompt: {prompt}")
#         print(f"Function: {fn_name}")

#         params = extractor.extract(fn_def, prompt)

#         print("Extracted parameters:")
#         print(params)

#     print("\nDone.")


# if __name__ == "__main__":
#     main()