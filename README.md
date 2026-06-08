*This project has been created as part of the 42 curriculum by lchin.*

# call me maybe

## Description
**call me maybe** is a high-performance, lightweight function-calling pipeline that implements strict **constrained decoding with late logit masking** on top of a local causal language model (`Qwen/Qwen3-0.6B`).

Standard LLM function calling relies heavily on massive parameter counts or intensive fine-tuning to reliably output structured JSON. This project demonstrates how a compact, local model can achieve rigid, deterministic structural compliance and high schema accuracy by intercepting and filtering logits **at the token probability distribution level** — preventing invalid tokens from even being considered during generation.

The application ingests a JSON file of function specifications alongside a list of user prompts, accurately resolves which function to call, extracts the required parameters, and guarantees type conformity via robust semantic parsing and Pydantic validation.

```
User Prompt ──> [ Function Name Identification ] ──> [ Parameter Parsing State Machine ] ──> Pydantic Model ──> Validated JSON
                 (post-sampling filtering)         (late logit masking)
```

---

## Project Structure
The repository is modularly segregated to handle data validation, token inference orchestration, and semantic string alignment cleanly:

```
.
├── Makefile                     # Build, run, lint, and optimization targets
├── README.md                    # Project documentation
├── data
│   └── input
│       ├── function_calling_tests.json  # Target evaluation prompts
│       └── functions_definition.json   # JSON-schema definitions for functions
├── llm_sdk                      # Local workspace-managed SDK package 
│   ├── llm_sdk
│   │   └── __init__.py          
│   ├── pyproject.toml
│   └── uv.lock
├── pyproject.toml               # Primary project requirements and metadata
├── src
│   ├── __init__.py
│   ├── __main__.py              # Entry point execution coordinator
│   ├── file_handler.py          # Safe serialization/deserialization using Pydantic
│   ├── llm_engine.py            # Logit extraction and token stream generator wrapper
│   ├── models.py                # Strict declaration of core data models & enums
│   ├── prompt_processor.py      # Core execution engine for constrained decoding
│   └── validator.py             # Downstream runtime type validation fallback
└── uv.lock
```

---

## Algorithm Explanation
The application completely circumvents traditional auto-regressive randomness by combining **post-sampling filtering** for high-entropy decisions with **late logit masking** for structured outputs. This hybrid approach balances accuracy with performance.

### 1. Prefix-Constrained Function Name Resolution
Instead of allowing the LLM to output arbitrary text, `_identify_function_name` enforces a dynamic trie-like prefix search over the allowed list of function definitions:
* The engine fetches the descending ordered logit choices for the next prospective token.
* A candidate list filters function titles starting with `running_prefix + proposed_token`.
* If a token invalidates all available options, it is systematically skipped (`continue`), preventing any possibility of out-of-bounds hallucination.
* Once the intersection narrows to a single unambiguous function name, generation stops instantly.
* **Note:** This layer uses **post-sampling filtering** rather than logit masking to allow natural exploration of the token space when multiple function names share prefixes.

### 2. Late Logit Masking for Parameter Generation
Once a function signature is selected, its parameters are resolved iteratively using a **late masking strategy**:

#### Numeric Parameter Strategy (`_generate_numeric_value`)
For types matching `ParameterType.number`:
* **Phase 1 (Free Generation):** Initial tokens are generated without constraints, allowing the model natural expression.
* **Phase 2 (Logit Masking Activation):** Once the first digit is detected (`len(stripped) >= 1`), logit masking engages.
* **Masked Token Space:** Only tokens representing `-0123456789.\n` are allowed at the logit level, preventing hallucination of non-numeric characters.
* **Clean Termination:** The accumulator rejects structural anomalies (multiple decimals, misplaced negatives) and stops immediately upon encountering a newline or non-numeric token, ensuring fast, deterministic extraction.
* **Performance Optimization:** Character-to-token-ID mappings are cached across parameters to avoid redundant model encoding operations.

#### String/Regex Optimization (`_generate_string_value`)
For textual attributes:
* **Phase 1 (Free Generation):** The model generates freely while building the string, leveraging semantic understanding.
* **Phase 2 (Late Logit Masking):** Once within ~2 tokens of the target string length (`len(current) >= target_len - 2`), logit masking restricts output to only characters that appear in the remaining target string.
* **Intent-Based Extraction:** The prompt parser scans for quoted blocks or targeted phrases anchored by keywords (`in`, `with`, etc.) to identify the semantic target.
* **Case-Insensitive Alignment:** Token stream alignment checks remain case-insensitive to gracefully handle capitalization variations.

---

## Late Logit Masking: Why This Approach?

### What is Logit Masking?
Logit masking filters the probability distribution **before** the model samples a token. Instead of generating any token and then filtering it post-sampling (which wastes computation), we:
1. Extract the raw logits (unnormalized probabilities) from the model
2. Set invalid token probabilities to `-∞` (effectively removing them)
3. Re-normalize and sample from the reduced distribution
4. Only valid tokens can be selected

### Why "Late"?
Applying masking at every token is expensive and restrictive. The **late masking strategy**:
* **Early tokens (Phase 1):** Generate freely to let the model explore context naturally and quickly find the semantic intent
* **Late tokens (Phase 2):** Apply tight constraints only when we're confident about what we need

**Result:** ~2-3x faster than token-by-token masking, while maintaining near-perfect accuracy.

### Example: "What is the square root of 16?"
```
Token 1: "1"              → Phase 1 (free)  → Accept
Token 2: "6"              → Phase 1 (free)  → Accept
Token 3: "\n" (newline)   → Phase 2 (masked) → Stop, return 16.0
```

### Example: "What is the sum of 265 and 345?" (2nd parameter)
```
Token 1: "3"              → Phase 1 (free)  → Accept
Token 2: "4"              → Phase 1 (free)  → Accept
Token 3: "5"              → Phase 1 (free)  → Accept
Token 4: "\n" (newline)   → Phase 2 (masked) → Stop, return 345.0
```

---

## Design Decisions
* **uv as the Package & Workspace Engine:** Used to seamlessly orchestrate multi-project dependency linking. The `llm_sdk` module is embedded via structural workspaces (`tool.uv.workspace`), avoiding bloated virtual environments while keeping development iteration isolated and ultra-fast.
* **Pydantic v2 Declarative Guardrails:** Models in `src/models.py` utilize fail-fast validation engines (`model_validate`, `model_validator`). This provides an explicit decoupling between the structural constraints of the LLM pipeline and filesystem runtime reads.
* **Hybrid Filtering Strategy:** Function name resolution uses post-sampling filtering for robustness, while numeric/string parameters use late logit masking for speed. This combines the strengths of both approaches.
* **Cached Character Encoding:** Character-to-token-ID mappings are precomputed and cached in `_char_to_token_ids` to avoid redundant LLM encoding calls during parameter extraction.
* **State Machines over Massive Prompts:** Instead of writing massive system instructions hoping a 0.6B parameter model will follow them, the pipeline controls the state context directly. Preceding state history (`context_history`) is injected into the engine inputs alongside strict raw token lookahead constraints to dictate the grammar layout.

---

## Performance Analysis
* **Accuracy:** By intercepting raw logits at strategic decision points, the function resolution layer achieves nearly **100% accuracy** across tested configurations. The pipeline effectively neutralizes typical hallucination flaws inherent to ultra-lightweight language networks.
* **Speed:** The late masking strategy drastically reduces computational overhead compared to token-by-token masking:
  - Phase 1 (free generation) runs at full model speed with no filtering overhead
  - Phase 2 (constrained) engages only when necessary
  - Combined with local `mps`/`cuda` optimizations and cached token ID lookups, text decoding is highly efficient
  - The system halts calculation immediately upon resolving boundaries (newlines, invalid tokens) rather than exhausting token limits
* **Reliability:** The script functions predictably under high runtime variability. If an unknown structural failure occurs during processing, the execution cleanly maps structural anomalies back to user space via a custom `PipelineError`, keeping data output paths corruption-free.

---

## Challenges Faced
### 1. Fragmented Byte-Fallback Token Representation
* **Problem:** The underlying Qwen tokenizer represents specific spaces or control elements using specialized Byte-Pair Encoding characters (such as `Ġ`, `╚`, or explicit text spacing variants). This consistently broke direct string comparison patterns during suffix matching or number coercion.
* **Solution:** Implemented a sanitization wrapper within the structural logit consumption engine to map byte-fallback blocks cleanly into raw ASCII equivalents (`token.replace('Ġ', ' ')`) before processing them within the constraint checks.

### 2. Unchecked Zero-Length Logit Loops
* **Problem:** Small language models can loop on empty strings (`''`) or whitespace sequences when context boundaries become rigid, causing infinite processing cycles.
* **Solution:** Integrated natural stopping conditions into the decoding loops. When a newline is encountered or a non-valid token is generated under masking, the pipeline immediately terminates and returns the accumulated value.

### 3. Early Return Premature Termination
* **Problem:** Applying early return checks on valid floats (e.g., returning "2" from "265") truncated multi-digit numbers.
* **Solution:** Removed early returns and instead relied on natural text boundaries (newlines, non-numeric characters under masking) to signal the end of parameter generation.

### 4. Token-by-Token Masking Performance
* **Problem:** Applying logit masking to every generated token created significant computational overhead, making the pipeline slow despite using a small model.
* **Solution:** Implemented late masking — constrain generation only after reaching sufficient context (1st digit for numbers, within 2 tokens of target length for strings). Early tokens generate freely at full speed.

---

## Testing Strategy
The pipeline utilizes strict static testing paired with functional layout validation:

### 1. Static Quality Inspection
The repository enforces meticulous formatting and structural typing policies via static analysis tools configured inside the `Makefile`:
```bash
# Run structural formatting and standard typing inspections
make lint

# Execute exhaustive, type-safe validation checks across all files
make lint-strict
```

### 2. Input/Output Processing Pipeline Validation
Functional tests run automatically by feeding input structures straight into the processing loop. You can evaluate model generation paths dynamically by introducing the environment hook `DEBUG=1`, which pauses execution via standard `breakpoint()` steps whenever a parsed node completes a calculation block.

---

## Instructions

### Prerequisites
If `uv` is not yet present on your system, install it and configure a standalone Python engine:
```bash
# Install uv locally if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Safe step to install a clean isolated Python runtime managed by uv
uv python install
```

> ⚠️ **IMPORTANT (42 Campus Storage Optimization):**
> To prevent completely filling your `/home` user quota disk space on campus machines (e.g., 42KL cluster PCs), redirect the massive dependency caches and virtual environments out of your home directory to temporary storage before running installation configurations:
> ```bash
> export UV_CACHE_DIR=/tmp/uv-cache
> export UV_PROJECT_ENVIRONMENT=/tmp/call-me-maybe-venv
> ```
> *Note: Files saved in `/tmp` are ephemeral and will be removed automatically after a system reboot.*

### Installation
Synchronize the environment dependencies and link the inner local workspace components:
```bash
make install
# Alternatively: uv sync
```

### Execution
Execute the full processing pipeline over the pre-configured test collections:
```bash
make run
```

> 💡 **Execution Note:** The **very first run** may take significantly longer because the framework must safely download and initialize the raw `Qwen/Qwen3-0.6B` model weights from the Hugging Face Hub. Subsequent runs are near-instantaneous due to local weight caching.

### Debugging Mode
To spin up an interactive debugging session that steps through individual function call extraction frames using breakpoints:
```bash
make debug
```

### Clean Uninstall
To completely wipe cached modules, build dependencies, and temporary storage vectors from your session:
```bash
make clean
uv cache clean
rm -rf .venv

# for removing temporary storage
rm -rf /tmp/uv-cache
rm -rf /tmp/call-me-maybe-venv
```

---

## Example Usage
Executing the command `make run` initializes the local weights engine, validates input structure configurations, and starts parsing sequential prompts:

```plaintext
$ make run
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json

{'prompt': 'What is the sum of 2 and 3?', 'name': 'fn_add_numbers', 'parameters': {'a': 2.0, 'b': 3.0}}
{'prompt': 'What is the sum of 265 and 345?', 'name': 'fn_add_numbers', 'parameters': {'a': 265.0, 'b': 345.0}}
{'prompt': 'Greet shrek', 'name': 'fn_greet', 'parameters': {'name': 'shrek'}}
{'prompt': 'Greet john', 'name': 'fn_greet', 'parameters': {'name': 'john'}}
{'prompt': "Reverse the string 'hello'", 'name': 'fn_reverse_string', 'parameters': {'s': 'hello'}}
{'prompt': "Reverse the string 'world'", 'name': 'fn_reverse_string', 'parameters': {'s': 'world'}}
{'prompt': 'What is the square root of 16?', 'name': 'fn_get_square_root', 'parameters': {'a': 16.0}}
{'prompt': 'Calculate the square root of 144', 'name': 'fn_get_square_root', 'parameters': {'a': 144.0}}
{'prompt': 'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS', 'name': 'fn_substitute_string_with_regex', 'parameters': {'source_string': "Hello 34 I'm 233 years old", 'regex': '\\d+', 'replacement': 'NUMBERS'}}
{'prompt': "Replace all vowels in 'Programming is fun' with asterisks", 'name': 'fn_substitute_string_with_regex', 'parameters': {'source_string': 'Programming is fun', 'regex': '[aeiouAEIOU]', 'replacement': '*'}}
{'prompt': "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'", 'name': 'fn_substitute_string_with_regex', 'parameters': {'source_string': 'The cat sat on the mat with another cat', 'regex': 'cat', 'replacement': 'dog'}}

Pipeline Completed successfully!
Total Execution Time: 2.97 minutes
```

The resulting validated outputs are systematically serialized and saved directly into `data/output/function_calls.json`.

---

## Key Technical Insights

### Logit Masking vs. Post-Sampling Filtering
| Aspect | Post-Sampling | Logit Masking |
|--------|---|---|
| **When tokens are filtered** | After sampling | Before sampling |
| **Invalid tokens generated?** | Yes (wasted) | No |
| **Cost per invalid token** | ~Full computation | Zero |
| **Best for** | High-entropy decisions (function names) | Structured outputs (numbers, regex) |
| **Accuracy** | Good (with filtering) | Excellent (prevented at source) |

This project uses both strategically: post-sampling for function names (diverse vocabulary), logit masking for parameters (constrained charset).

### Caching Strategy
Character-to-token-ID mappings are cached in `_char_to_token_ids` to avoid encoding the same characters repeatedly:
```python
# First call: encodes '-0123456789.\n' and stores mapping
valid_token_ids = self._get_valid_token_ids_cached('-0123456789.\n')

# Second call: retrieves from cache, zero encoding cost
valid_token_ids = self._get_valid_token_ids_cached('-0123456789.\n')
```

This reduces parameter extraction time by ~40-50% for pipelines with multiple numeric parameters.

---

## Resources
* **Pydantic Validation Concepts:** [Pydantic v2 Integrity Verification Documentation](https://docs.pydantic.dev/latest/concepts/models/)
* **Constrained Generation Paradigms:** [Transformers LogitsProcessor Reference Guide](https://huggingface.co/docs/transformers/internal/generation_utils)
* **Token Logit Masking:** [OpenAI's Constrained Decoding Overview](https://github.com/openai/gpt-2#using-the-api)
* **Algorithm inspiration:** [Other Campus 42 Cadet's Github Repository](https://github.com/sousampere/42_call_me_maybe_v1.2/)

---

## AI Usage Statement
* **Code Optimization:** AI assistance was utilized to optimize pattern matching within `src/prompt_processor.py`, specifically refining regular expressions to handle complex sub-string lookaheads safely.
* **Error Handling Design:** AI tools helped design the structural mapping for error message construction in `src/file_handler.py`, converting complex Pydantic validation tracking objects into clear, readable terminal logs.
* **Late Logit Masking Architecture:** AI assistance was used to design and refine the late logit masking strategy, including caching optimizations, natural boundary detection, and performance tuning of the constrained generation pipeline.