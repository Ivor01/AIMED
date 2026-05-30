from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

from modules.medical_understanding.schemas import MainReason, MedicalEntity, TipPlana, WarningItem


ReviewStatus = Literal[
    "čeka_potvrdu",
    "potvrđeno",
    "ispravljano",
    "odbijeno"
]

SourceType = Literal[
    "doctor_input",
    "ai_extracted",
    "ai_generated_summary"
]

EncounterType = Literal[
    "ambulantne_konzultacije",
    "medicinske_bilješke",
    "opće"
]

class PatientInfo(BaseModel):
    patient_id: str
    age: Optional[int] = None
    sex: Optional[Literal["M","Ž"]]
    source: SourceType = "doctor_input"

class EncounterInfo(BaseModel):
    date: str
    type: EncounterType = "opće"
    department: Optional[str] = "Opća"
    reason_for_visit: Optional[str] = None
    source: SourceType = "doctor_input"

class PractitionerInfo(BaseModel):
    practitioner_id: Optional[str] = None
    name: Optional[str] = None
    department: Optional[str] = None
    source: SourceType = "doctor_input"

class ExtractedEntity(BaseModel):
    data: MedicalEntity = Field(description="Originalni entitet ekstraktiran u medicinskom razumijevanju.")
    source: SourceType = "ai_extracted"
    review: ReviewStatus

class ExtractedPlan(BaseModel):
    data: TipPlana = Field(description="Iz originalne ekstrakcije.")
    source: SourceType = "ai_extracted"
    review: ReviewStatus

class ExtractedMainReason(BaseModel):
    data: MainReason = Field(description="Iz originalne ekstrakcije.")
    source: SourceType = "ai_extracted"
    review: ReviewStatus

class ExtractedWarning(BaseModel):
    data: WarningItem = Field(description="Iz originalne ekstrakcije.")
    source: SourceType = "ai_extracted"
    review: ReviewStatus

class ClinicalNote(BaseModel):
    format: Optional[str] = "SOAP"
    subjective: List[ExtractedEntity] = None
    objective: List[ExtractedEntity] = None
    assessment: List[ExtractedEntity] = None
    plan: List[ExtractedPlan] = None
    summary: Optional[str]

class RecordMetadata(BaseModel):
    record_id: str
    created_at: str
    created_by: str = "system"
    schema_version: str = "1.0"

class InternalEHRRecord(BaseModel):
    record_metadata: RecordMetadata
    patient: PatientInfo
    encounter: EncounterInfo
    practitioner: Optional[PractitionerInfo] = None
    clinical_note: ClinicalNote
    
