from src.llm_engine import LLMEngine


def main():
    llm = LLMEngine()

    tests = ["fn_greet", "fn_add_numbers", "fn_reverse_string"]

    for t in tests:
        tokens = llm.encode(t)

        print("\nTEXT:", t)
        print("TYPE:", type(tokens))
        print("VALUE:", tokens)

        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        print("NORMALIZED:", tokens)


if __name__ == "__main__":
    main()