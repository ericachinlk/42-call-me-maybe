from src.file_handler import load_functions
from src.llm_engine import LLMEngine
from src.function_selector import FunctionSelector


def main():

    print("Loading model...")
    llm = LLMEngine()
    print("Model loaded")

    functions = load_functions(
        "src/data/input/functions_definition.json"
    )

    selector = FunctionSelector(
        llm,
        functions,
    )

    tests = [
        "What is the sum of 2 and 3?",
        "What is the sum of 265 and 345?",
        "Greet shrek",
        "Greet john",
        "Reverse the string 'hello'",
        "Reverse the string 'world'",
        "What is the square root of 16?",
        "Calculate the square root of 144",
        "Replace all numbers in 'Hello 34 I'm 233 years old' with NUMBERS",
        "Replace all vowels in 'Programming is fun' with asterisks",
    ]

    for prompt in tests:
        print(prompt)
        print(selector.select_function(prompt))
        print()

    # prompt = "What is the sum of 265 and 345?"

    # print()
    # print("Prompt:")
    # print(prompt)

    # print()
    # print("Selecting function...")

    # result = selector.select_function(prompt)

    # print()
    # print("Selected function:")
    # print(result)


if __name__ == "__main__":
    main()