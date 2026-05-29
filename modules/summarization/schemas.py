from pydantic import BaseModel, Field
from typing import List


class StyledMedicalNote(BaseModel):
    summary: str = Field(description="Stilizirani medicinski zapis.")
    used_facts: List[str] = Field(default_factory=list)
    