import argparse
from pathlib import Path
from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--functions_definition", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    # create parent directory for output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        functions_path=args.functions_definition,
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
