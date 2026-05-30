import json
from pathlib import Path

from modules.electronicRecords.schemas import InternalEHRRecord


class EHRJsonExporter:
    def export_to_dict(self, ehr_record: InternalEHRRecord) -> dict:
        return ehr_record.model_dump(mode="json")

    def export_to_json_string(self, ehr_record: InternalEHRRecord) -> str:
        return ehr_record.model_dump_json(indent=2)

    def save_to_file(
        self,
        ehr_record: InternalEHRRecord,
        output_path: str,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                ehr_record.model_dump(mode="json"),
                file,
                ensure_ascii=False,
                indent=2,
            )

    
    def save_dict_to_file(
        self,
        data: dict,
        output_path: str,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )