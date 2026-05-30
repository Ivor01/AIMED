from modules.electronicRecords.schemas import InternalEHRRecord


class FHIRMapper:
    def to_minimal_bundle(self, ehr_record: InternalEHRRecord) -> dict:
        return {
            "resourceType": "Bundle",
            "type": "document",
            "entry": [
                {
                    "resource": self._build_composition(ehr_record)
                }
            ]
        }

    def _build_composition(self, ehr_record: InternalEHRRecord) -> dict:
        return {
            "resourceType": "Composition",
            "status": "preliminary",
            "title": "AI generirani klinički zapis",
            "date": ehr_record.record_metadata.created_at,
            "subject": {
                "reference": f"Patient/{ehr_record.patient.patient_id}"
            },
            "encounter": {
                "display": ehr_record.encounter.type
            },
            "author": [
                {
                    "display": (
                        ehr_record.practitioner.name
                        if ehr_record.practitioner
                        else "Nepoznati liječnik"
                    )
                }
            ],
            "section": [
                {
                    "title": "Subjective",
                    "text": {
                        "status": "generated",
                        "div": self._section_to_html(
                            ehr_record.clinical_note.subjective
                        )
                    }
                },
                {
                    "title": "Objective",
                    "text": {
                        "status": "generated",
                        "div": self._section_to_html(
                            ehr_record.clinical_note.objective
                        )
                    }
                },
                {
                    "title": "Assessment",
                    "text": {
                        "status": "generated",
                        "div": self._section_to_html(
                            ehr_record.clinical_note.assessment
                        )
                    }
                },
                {
                    "title": "Plan",
                    "text": {
                        "status": "generated",
                        "div": self._plan_to_html(
                            ehr_record.clinical_note.plan
                        )
                    }
                }
            ]
        }

    def _section_to_html(self, items) -> str:
        if not items:
            return "<div>Nema podataka.</div>"

        lines = []

        for item in items:
            entity = item.data

            lines.append(
                f"<li>{entity.normalized_name} "
                f"({entity.status}) - {entity.evidence}</li>"
            )

        return "<div><ul>" + "".join(lines) + "</ul></div>"

    def _plan_to_html(self, plans) -> str:
        if not plans:
            return "<div>Nema plana.</div>"

        lines = []

        for plan in plans:
            data = plan.data

            if hasattr(data, "model_dump"):
                content = data.model_dump(mode="json")
            else:
                content = data

            lines.append(f"<li>{content}</li>")

        return "<div><ul>" + "".join(lines) + "</ul></div>"