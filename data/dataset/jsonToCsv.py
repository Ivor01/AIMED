import json
import csv
from pathlib import Path

# Files are all in the same root folder
JSON_PATH = Path("transcript.json")
OUT_CSV = Path("metadata.csv")

def main():
    items = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    if not isinstance(items, list):
        raise ValueError("JSON must be a list of objects with 'audio' and 'text'.")

    rows = []
    missing = 0
    skipped = 0

    for entry in items:
        audio_path = Path(entry.get("audio", "").strip())
        text = entry.get("text", "").strip()

        if not audio_path or not text:
            skipped += 1
            continue

        if not audio_path.exists():
            missing += 1
            continue

        rows.append({
            "path": str(audio_path.resolve()),
            "text": text
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Wrote {len(rows)} valid entries to {OUT_CSV}")
    if missing:
        print(f"⚠ {missing} audio files not found.")
    if skipped:
        print(f"⚠ {skipped} entries skipped due to missing audio/text.")

if __name__ == "__main__":
    main()