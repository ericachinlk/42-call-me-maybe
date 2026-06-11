FUNCTION_PATH = data/input/functions_definition.json
INPUT_PATH = data/input/function_calling_tests.json
OUTPUT_PATH = data/output/function_calling_results.json

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition $(FUNCTION_PATH) \
		--input $(INPUT_PATH) \
		--output $(OUTPUT_PATH)

debug:
	DEBUG=1 uv run python -m src \
		--functions_definition $(FUNCTION_PATH) \
		--input $(INPUT_PATH) \
		--output $(OUTPUT_PATH)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	uv run flake8 . \
	--exclude=.venv,__pycache__,.mypy_cache,llm_sdk,tests
	uv run mypy . --warn-return-any --warn-unused-ignores \
	--disallow-untyped-defs --check-untyped-defs --exclude '(.venv|llm_sdk|tests)'

lint-strict:
	uv run flake8 . \
	--exclude=.venv,__pycache__,.mypy_cache,llm_sdk,tests
	uv run mypy . --strict --exclude '(.venv|llm_sdk|tests)'

.PHONY: install run debug clean lint lint-strict
