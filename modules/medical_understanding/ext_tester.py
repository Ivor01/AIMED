import argparse
import json
import sys
from pathlib import Path

from extractor import MedicalExtractor
from schemas import InputUtterance


def load_segments_from_json(file_path: str) -> list[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of segments.")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Segment at index {index} is not a JSON object.")

    return data


def convert_to_utterances(raw_segments: list[dict]) -> list[InputUtterance]:
    utterances = []

    for index, segment in enumerate(raw_segments):
        try:
            utterance = InputUtterance.model_validate(segment)
        except Exception as exc:
            raise ValueError(
                f"Invalid segment at index {index}: {exc}"
            ) from exc

        if not utterance.text.strip():
            continue

        utterances.append(utterance)

    if not utterances:
        raise ValueError("No valid utterances found in input file.")

    return utterances


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run medical understanding extractor on input JSON file."
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
        utterances = convert_to_utterances(raw_segments)

        extractor = MedicalExtractor(model=args.model)
        response = extractor.extract(utterances)

        response_dict = response# = response.model_dump()

        response_json = json.dumps(
            response_dict,
            indent=2,
            ensure_ascii=False
        )

        if args.output:
            output_path = Path(args.output)
            output_path.write_text(response_json, encoding="utf-8")
            print(f"Response saved to: {output_path}")
        else:
            print(type(response_dict))

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())