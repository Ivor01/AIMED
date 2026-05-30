from datetime import datetime
from uuid import uuid4
from typing import Any, Optional

from pydantic import BaseModel

from modules.electronicRecords.schemas import (
    PatientInfo,
    EncounterInfo,
    PractitionerInfo,
    RecordMetadata,
    ClinicalNote,
    InternalEHRRecord,
    ExtractedEntity,
    ExtractedPlan,
    ExtractedMainReason,
    ExtractedWarning,
)

from modules.medical_understanding.schemas import (
    MedicalEntity,
    TipPlana,
    MainReason,
    WarningItem,
)


class EHRBuilder:

    DEFAULT_REVIEW_STATUS = "čeka_potvrdu"

    def build(
        self,
        frontend_input: dict[str, Any],
        medical_understanding: Any,
        medical_summary: Optional[Any] = None,
    ) -> InternalEHRRecord:
        
        record_metadata = self._build_record_metadata()
        patient = self._build_patient(frontend_input)
        encounter = self._build_encounter(frontend_input)
        practitioner = self._build_practitioner(frontend_input)
        
        clinical_note = self._build_clinical_note(medical_understanding, medical_summary)

        return InternalEHRRecord(
            record_metadata=record_metadata,
            patient=patient,
            encounter=encounter,
            practitioner=practitioner,
            clinical_note=clinical_note,
        )

    def _build_record_metadata(self) -> RecordMetadata:
        return RecordMetadata(
            record_id=f"ehr-{uuid4()}",
            created_at=datetime.now().isoformat(timespec="seconds"),
            created_by="system",
            schema_version="1.0",
        )

    def _build_patient(self, frontend_input: dict[str, Any]) -> PatientInfo:
        patient_data = frontend_input["patient"]

        return PatientInfo(
            patient_id=patient_data["patient_id"],
            age=patient_data["age"],
            sex=patient_data["sex"],
            source="doctor_input",
        )

    def _build_encounter(self, frontend_input: dict[str, Any]) -> EncounterInfo:
        encounter_data = frontend_input.get("encounter", {})

        return EncounterInfo(
            date=encounter_data.get("date", datetime.now().date().isoformat()),
            type=encounter_data.get("type", "opće"),
            department=encounter_data.get("department", "Opća"),
            reason_for_visit=encounter_data.get("reason_for_visit"),
            source="doctor_input",
        )

    def _build_practitioner(self, frontend_input: dict[str, Any],) -> Optional[PractitionerInfo]:
        practitioner_data = frontend_input.get("practitioner")

        return PractitionerInfo(
            practitioner_id=practitioner_data.get("practitioner_id"),
            name=practitioner_data.get("name"),
            department=practitioner_data.get("department"),
            source="doctor_input",
        )

    def _build_clinical_note(
        self,
        medical_understanding, medical_summary) -> ClinicalNote:
     
        subjective = []
        objective = []
        assessment = []
        plan = []
        
        for entity in medical_understanding["entities"]:
            if entity["speaker"] == "Pacijent":
                subjective.append(ExtractedEntity(data=entity,source="ai_extracted",review=self.DEFAULT_REVIEW_STATUS))
            elif entity["speaker"] == "Liječnik" and entity["kategorija"] in ("nalaz", "lijekovi"):
                objective.append(ExtractedEntity(data=entity,source="ai_extracted",review=self.DEFAULT_REVIEW_STATUS))
            elif entity["kategorija"] == "dijagnoze":
                assessment.append(ExtractedEntity(data=entity,source="ai_extracted",review=self.DEFAULT_REVIEW_STATUS))    
        for plan_ent in medical_understanding["plan"]:
            plan.append(ExtractedPlan(data=plan_ent,source="ai_extracted",review=self.DEFAULT_REVIEW_STATUS))
        return ClinicalNote(
            format="SOAP",
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan,
            summary=medical_summary
        )
