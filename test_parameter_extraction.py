from src.llm_engine import LLMEngine
from src.file_handler import load_functions
from src.function_selector import FunctionSelector
from src.parameter_extractor import ParameterExtractor


def main():

    llm = LLMEngine()

    functions = load_functions("src/data/input/functions_definition.json")

    fn_map = {f.name: f for f in functions}

    extractor = ParameterExtractor(llm)
    selector = FunctionSelector(llm, functions,)

    # tests = [
    #     ("What is the sum of 265 and 345?", "fn_add_numbers"),
    #     ("Greet shrek", "fn_greet"),
    #     ("Reverse the string 'hello'", "fn_reverse_string"),
    #     ("Calculate the square root of 144", "fn_get_square_root"),
    #     ("Replace all vowels in 'Programming is fun' with asterisks", "fn_substitute_string_with_regex")
    # ]

    tests = [
        # ("What is the sum of 2 and 3?", "fn_add_numbers"),
        # ("Greet john", "fn_greet"),
        # ("Reverse the string 'world'", "fn_reverse_string"),
        # ("What is the square root of 16?", "fn_get_square_root"),
        ("Replace all vowels in 'Programming is fun' with asterisks", "fn_substitute_string_with_regex"),
        ("Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'", "fn_substitute_string_with_regex"),
        ("Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS", "fn_substitute_string_with_regex")
    ]
    
    for prompt, expected_fn in tests:
        print("=" * 60)
        print("Prompt:", prompt)

        fn_name = selector.select(prompt)

        print("Predicted Function:", fn_name)
        print("Expected Function:", expected_fn)

        # assert fn_name == expected_fn, f"Mismatch: {fn_name} != {expected_fn}"

        fn_def = fn_map[fn_name]

        params = extractor.extract(fn_def, prompt)

        print("Parameters:", params)


def run(prompt, selector, extractor, fn_map):

    fn_name = selector.select(prompt)
    fn_def = fn_map[fn_name]

    params = extractor.extract(fn_def, prompt)

    return {
        "prompt": prompt,
        "name": fn_name,
        "parameters": params
    }


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