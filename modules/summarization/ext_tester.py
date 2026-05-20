import argparse
import json
import sys
from pathlib import Path

from medical_summarizer import MedicalSummarizer

def load_segments_from_json(file_path: str) -> list[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must contain a list of segments.")

    #for index, item in enumerate(data):
    #    if not isinstance(item, list):
    #        raise ValueError(f"Segment at index {index} is not a JSON object.")

    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run medical summarizer on input JSON file."
    )

    parser.add_argument(
        "input_file",
        help="Path to JSON file containing list of transcript segments."
    )

    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name. Default: gpt-4.1-mini"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save extractor response as JSON."
    )

    args = parser.parse_args()

    try:
        raw_segments = load_segments_from_json(args.input_file)
        
        summarizer = MedicalSummarizer()
        #response = extractor.extract(utterances)
        print(summarizer.summarize(raw_segments))

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())