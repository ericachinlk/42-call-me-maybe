from llm_sdk.llm_sdk import Small_LLM_Model
import json


def main():
    llm = Small_LLM_Model()

    text = "fn_add_numbers"
    ids = llm.encode(text)[0].tolist()

    print("encoded:", ids)

    print("decoded:", llm.decode(ids))


# def main():

#     llm = Small_LLM_Model()

#     # 1. get vocab file path
#     vocab_path = llm.get_path_to_vocab_file()
#     print("Vocab file:", vocab_path)

#     # 2. load vocab
#     with open(vocab_path, "r", encoding="utf-8") as f:
#         vocab = json.load(f)

#     # 3. build reverse mapping
#     id_to_token = {
#         token_id: token
#         for token, token_id in vocab.items()
#     }

#     # 4. test lookup
#     print("\nExample lookups:")

#     for token, token_id in list(vocab.items())[:5]:
#         print(f"token='{token}' → id={token_id}")
#         print(f"id back → token='{id_to_token[token_id]}'")


# def main():
#     print("loading model...", flush=True)

#     llm = Small_LLM_Model()

#     print("model loaded", flush=True)

#     ids = llm.encode("Hello")[0].tolist()
#     print("encoding...")
#     print(ids)

#     print("logits...")
#     logits = llm.get_logits_from_input_ids(ids)
#     print(logits[:10])  # print only first 10 values


if __name__ == "__main__":
    main()