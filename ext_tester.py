from pydantic import ValidationError

from modules.electronicRecords.internalEHR import EHRBuilder

from modules.medical_understanding.schemas import (
    MedicalEntity,
    TipPlana,
)
from modules.electronicRecords.eRecorder import ElectronicRecorder

def create_example_input_metadata() -> dict:
    """
    Simulates data entered by doctor in frontend before recording/transcription.
    """

    return {
        "patient": {
            "patient_id": "demo-patient-001",
            "age": 52,
            "sex": "M",
        },
        "encounter": {
            "date": "2026-05-30",
            "type": "ambulantne_konzultacije",
            "department": "Opća medicina",
            "reason_for_visit": "Kašalj i povišena temperatura",
        },
        "practitioner": {
            "practitioner_id": "doctor-001",
            "name": "dr. Demo Liječnik",
            "department": "Opća medicina",
        },
    }


def create_example_med_entities() -> list[MedicalEntity]:
    """
    Simulates output from medical_understanding module.

    Important:
    The exact string values for kategorija, status, sigurnost, temporalnost and speaker
    must match the values allowed in your medical_understanding.schemas.
    If your Literals/Enums use different values, rename the values below.
    """
    med_ents = [
        MedicalEntity(
            text= "boli me koljeno",
            normalized_name="bol u koljenu",
            kategorija="simptom",
            status="sadašnje",
            sigurnost="sigurno",
            temporalnost="sadašnje",
            speaker="Pacijent",
            duration="3 dana",
            severity="umjereno",
            body_location=None,
            value=None,
            unit=None,
            evidence="boli me koljeno",
            utterance_id="utt-001",
        ),
        MedicalEntity(
            text="Ibuprofen",
            normalized_name="ibuprofen",
            kategorija="lijekovi",
            status="sadašnje",
            sigurnost="sigurno",
            temporalnost="sadašnje",
            speaker="Pacijent",
            duration="2 dana",
            severity=None,
            body_location=None,
            value="38.2",
            unit="°C",
            evidence="Pacijent navodi temperaturu do 38.2 °C zadnja dva dana.",
            utterance_id="utt-002",
        ),
        MedicalEntity(
            text="uho vam je crveno",
            normalized_name="crveno uho",
            kategorija="simptom",
            status="negated",
            sigurnost="sigurno",
            temporalnost="sadašnje",
            speaker="Liječnik",
            duration=None,
            severity=None,
            body_location="prsa",
            value=None,
            unit=None,
            evidence="Pacijent negira otežano disanje.",
            utterance_id="utt-003",
        ),
        TipPlana(type= "TEMP",
                name="popisati terapiju",
                status= "planirano",
                evidence= "sada ćemo vam popisati terapiju",
                utterance_id= "utt_10")
        
    ]

    med_entities_json = {
    "entities": [
        entity.model_dump(mode="json")
        for entity in med_ents[:-1]
    ],
    "plan": [med_ents[-1].model_dump(mode="json")]
    }
    return med_entities_json


def create_example_medical_summary() -> dict:
    return """Doktor izvršio pregled."""


def main() -> None:
    input_metadata = create_example_input_metadata()
    med_entities = create_example_med_entities()
    medical_summary = create_example_medical_summary()

    recorder = ElectronicRecorder()
    recorder.record(input_metadata,med_entities,medical_summary)
    


if __name__ == "__main__":
    main()