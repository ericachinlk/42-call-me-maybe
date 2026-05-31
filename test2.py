from llm_sdk.llm_sdk import Small_LLM_Model


def main():
    llm = Small_LLM_Model()

    # -------------------------------------------------
    # 1. Define allowed function names
    # -------------------------------------------------
    valid_functions = ["fn_greet", "fn_add_numbers"]

    # encode each function into token sequences
    valid_sequences = {
        fn: llm.encode(fn)[0].tolist()
        for fn in valid_functions
    }

    print("Valid sequences:")
    for k, v in valid_sequences.items():
        print(k, "→", v)

    # -------------------------------------------------
    # 2. Start decoding process
    # -------------------------------------------------
    generated = []

    # we don't really need a prompt here for demo
    input_ids = llm.encode("Greet John")[0].tolist()

    print("\nInput tokens:", input_ids)

    # -------------------------------------------------
    # 3. Constrained decoding loop
    # -------------------------------------------------
    while True:

        logits = llm.get_logits_from_input_ids(
            input_ids + generated
        )

        # -------------------------------------------------
        # 4. Find valid next tokens based on prefix
        # -------------------------------------------------
        allowed_next_tokens = set()

        for seq in valid_sequences.values():
            if len(seq) > len(generated):
                if seq[:len(generated)] == generated:
                    allowed_next_tokens.add(seq[len(generated)])

            # also allow starting tokens if nothing generated yet
            if len(generated) == 0:
                allowed_next_tokens.add(seq[0])

        # -------------------------------------------------
        # 5. MASK LOGITS (this is the key step)
        # -------------------------------------------------
        for i in range(len(logits)):
            if i not in allowed_next_tokens:
                logits[i] = float("-inf")

        # -------------------------------------------------
        # 6. Pick best valid token
        # -------------------------------------------------
        next_token = max(
            range(len(logits)),
            key=lambda i: logits[i]
        )

        generated.append(next_token)

        print("Generated so far:", generated)

        # -------------------------------------------------
        # 7. Stop when full sequence matches a function
        # -------------------------------------------------
        for fn, seq in valid_sequences.items():
            if generated == seq:
                print("\nFINAL FUNCTION:", fn)
                return


if __name__ == "__main__":
    main()