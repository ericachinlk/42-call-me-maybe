*This project has been created as part of the 42 curriculum by lchin.*

# call me maybe

## Description
**call me maybe** is a high-performance, lightweight function-calling pipeline that implements strict **constrained decoding** on top of a local causal language model (`Qwen/Qwen3-0.6B`).

Standard LLM function calling relies heavily on massive parameter counts or intensive fine-tuning to reliably output structured JSON. This project demonstrates how a compact, local model can achieve rigid, deterministic structural compliance and high schema accuracy by intercepting, filtering, and guiding token generation at the logit level. 

The application ingests a JSON file of function specifications alongside a list of user prompts, accurately resolves which function to call, extracts the required parameters, and guarantees type conformity via robust semantic parsing and Pydantic validation.

```
User Prompt ──> [ Function Name Identification ] ──> [ Parameter Parsing State Machine ] ──> Pydantic Model ──> Validated JSON
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
The application completely circumvents traditional auto-regressive randomness by converting open-ended token generation into deterministic paths using two key sub-systems:

### 1. Prefix-Constrained Function Name Resolution
Instead of allowing the LLM to output arbitrary text, `_identify_function_name` enforces a dynamic trie-like prefix search over the allowed list of function definitions:
* The engine fetches the descending ordered logit choices for the next prospective token.
* A candidate list filters function titles starting with `running_prefix + proposed_token`.
* If a token invalidates all available options, it is systematically skipped (`continue`), preventing any possibility of out-of-bounds hallucination.
* Once the intersection narrows to a single unambiguous function name, generation stops instantly, avoiding unnecessary processing overhead.

### 2. High-Speed Constrained Parameter Generation
Once a function signature is selected, its parameters are resolved iteratively:
* **Numeric Parameter Strategy (`_generate_numeric_value`):** For types matching `ParameterType.number`, the token generator locks out non-numeric characters. The accumulator loop rejects tokens containing structural anomalies like multiple decimals (`.`) or out-of-place negative signs (`-`). It safely forces the engine to step sequentially through a valid floating-point representation, executing clean stream termination on line breaks (`\n`).
* **String/Regex Optimization (`_generate_string_value`):** For textual attributes, the architecture blends **Intent-Based Semantic Extraction** with case-insensitive token alignment. The prompt parser scans for quoted blocks or targeted phrases anchored by keywords (like `in` or `with`). It dynamically guides candidate search trajectories toward structural alignment boundaries, guaranteeing zero out-of-bounds hallucinations.

---

## Design Decisions
* **`uv` as the Package & Workspace Engine:** Used to seamlessly orchestrate multi-project dependency linking. The `llm_sdk` module is embedded via structural workspaces (`tool.uv.workspace`), avoiding bloated virtual environments while keeping development iteration isolated and ultra-fast.
* **Pydantic v2 Declarative Guardrails:** Models in `src/models.py` utilize fail-fast validation engines (`model_validate`, `model_validator`). This provides an explicit decoupling between the structural constraints of the LLM pipeline and filesystem runtime reads.
* **State Machines over Massive Prompts:** Instead of writing massive system instructions hoping a 0.6B parameter model will follow them, the pipeline controls the state context directly. Preceding state history (`context_history`) is injected into the engine inputs alongside strict raw token lookahead arrays to dictate the grammar layout.

---

## Performance Analysis
* **Accuracy:** By intercepting raw logits, the function resolution layer achieves nearly **100% accuracy** across tested configurations. The pipeline effectively neutralizes typical hallucination flaws inherent to ultra-lightweight language networks.
* **Speed:** By combining target-scoped token interception with local `mps`/`cuda` optimizations within the underlying PyTorch code, text decoding overhead is drastically minimized. The system halts calculation immediately upon resolving boundaries (e.g., matching known target candidates or identifying a trailing newline), rather than exhausting the model's total token ceiling (`max_tokens`).
* **Reliability:** The script functions predictably under high runtime variability. If an unknown structural failure occurs during processing, the execution cleanly maps structural anomalies back to user space via a custom `PipelineError`, keeping data output paths corruption-free.

---

## Challenges Faced
### 1. Fragmented Byte-Fallback Token Representation
* **Problem:** The underlying Qwen tokenizer represents specific spaces or control elements using specialized Byte-Pair Encoding characters (such as `Ġ`, `╚`, or explicit text spacing variants). This consistently broke direct string comparison patterns during suffix matching or number coercion.
* **Solution:** Implemented a sanitization wrapper within the structural logit consumption engine to map byte-fallback blocks cleanly into raw ASCII equivalents (`token.replace('Ġ', ' ')`) before processing them within the constraint checks.

### 2. Unchecked Zero-Length Logit Loops
* **Problem:** Small language models can loop on empty strings (`''`) or whitespace sequences when context boundaries become rigid, causing infinite processing cycles.
* **Solution:** Integrated an automated escape hatch into the decoding loops. If consecutive tokens yield zero contextual progression or collapse structural validation checks, the pipeline falls back to semantic regex parameters inferred directly from the host prompt structures.

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
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

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
{'prompt': 'Greet shrek', 'name': 'fn_greet', 'parameters': {'name': 'shrek'}}
{'prompt': "Reverse the string 'hello'", 'name': 'fn_reverse_string', 'parameters': {'s': 'hello'}}
{'prompt': 'Calculate the square root of 144', 'name': 'fn_get_square_root', 'parameters': {'a': 144.0}}
{'prompt': 'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS', 'name': 'fn_substitute_string_with_regex', 'parameters': {'source_string': "Hello 34 I'm 233 years old", 'regex': '\\d+', 'replacement': 'NUMBERS'}}

Pipeline Completed successfully!
Total Execution Time: 1.57 minutes
```

The resulting validated outputs are systematically serialized and saved directly into `data/output/function_calls.json`.

---

## Resources
* **Pydantic Validation Concepts:** [Pydantic v2 Integrity Verification Documentation](https://docs.pydantic.dev/latest/concepts/models/)
* **Constrained Generation Paradigms:** [Transformers LogitsProcessor Reference Guide](https://huggingface.co/docs/transformers/internal/generation_utils)

---

## AI Usage Statement
* **Code Optimization:** AI assistance was utilized to optimize pattern matching within `src/prompt_processor.py`, specifically refining regular expressions to handle complex sub-string lookaheads safely.
* **Error Handling Design:** AI tools helped design the structural mapping for error message construction in `src/file_handler.py`, converting complex Pydantic validation tracking objects into clear, readable terminal logs.
