from typing import Literal, Optional
from pydantic import BaseModel, Field


Uloga = Literal["Liječnik","Pacijent","Ostalo"]

Kategorija = Literal[
    "simptom",
    "nalazi",
    "dijagnoze",
    "lijekovi",
    "plan",
    "negacije",
    "nesigurnost",
    "temporalnost"
]
Status = Literal[
    "sadašnje",
    "negated",
    "nesigurno",
    "vjerojatno",
    "malo vjerojatno",
    "isključeno",
    "planirano",
    "uvjetno_plan",
    "povijest",
    "obitelj_anamneza",
    "preporučeno",
    "samo_spomenuto",
]
Sigurnost = Literal[
    "sigurno",
    "nesigurno",
    "vjerojatno",
    "moguće",
    "malo vjerojatno"
]

Temporalnost =Literal[
    "sadašnje",
    "prošlost",
    "kronično",
    "buduće",
    "nepoznato"
]

TipPlana = Literal["TEMP"]

class InputUtterance(BaseModel):
    utterance_id: str
    speaker: str = "unknown"
    role: Uloga = "unknown"
    start: Optional[float] = None
    end: Optional[float] = None
    text: str

class MedicalEntity(BaseModel):
    text: str = Field(description="Originalni izraz iz transkripta.")
    normalized_name: str = Field(description="Normalizirani medicinski pojam.")
    kategorija: Kategorija
    status: Status
    sigurnost: Sigurnost = "sigurno"
    temporalnost: Temporalnost = "nepoznato"

    speaker: Uloga = "nepoznato"
    #experiencer: str = "pacijent"
    
    #duration: Optional[str] = None
    #severity: Optional[str] = None
    #body_location: Optional[str] = None
    #value: Optional[str] = None
    #unit: Optional[str] = None

    evidence: str
    utterance_id: str


class TipPlana(BaseModel):
    type: TipPlana
    name: str
    status: Literal["planirano", "uvjet_plan", "preporučeno"]
    #condition: Optional[str] = None
    #timeframe: Optional[str] = None
    evidence: str
    utterance_id: str


class WarningItem(BaseModel):
    type: str
    message: str
    utterance_id: Optional[str] = None


class MedicalUnderstandingOutput(BaseModel):
    entities: list[MedicalEntity]
    plan: list[TipPlana]
    warnings: list[WarningItem] = []
