from src.llm_engine import LLMEngine
from src.file_handler import load_functions
from src.function_selector import FunctionSelector
from src.parameter_extractor import ParameterExtractor


# def main():
#     llm = LLMEngine()

#     tests = ["fn_greet", "fn_add_numbers", "fn_reverse_string"]

#     for t in tests:
#         tokens = llm.encode(t)

#         print("\nTEXT:", t)
#         print("TYPE:", type(tokens))
#         print("VALUE:", tokens)

#         if hasattr(tokens, "tolist"):
#             tokens = tokens.tolist()

#         print("NORMALIZED:", tokens)

def main():
    llm = LLMEngine()
    functions = load_functions("src/data/input/functions_definition.json")
    # extractor = ParameterExtractor(llm)
    selector = FunctionSelector(llm, functions)
    prompt = selector.build_prompt(
        "Replace all vowels in 'Programming is fun' with asterisks"
    )

    tokens = llm.encode(prompt)
    logits = llm.logits(tokens)

    for fn_name, seq in selector.function_tokens.items():
        first_token = seq[1]

        print(
            fn_name,
            first_token,
            logits[first_token]
        )


if __name__ == "__main__":
    main()