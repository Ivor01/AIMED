
from structuredSummary import StructuredSummaryGenerator
from narratedSummary import NarratedSummaryGenerator
class MedicalSummarizer:
    def __init__(self):
        self.structured_summator = StructuredSummaryGenerator()
        self.narrator_summator = NarratedSummaryGenerator()
    def summarize(self, med_entities: dict[dict]) -> str:
        structured_summary = self.structured_summator.structurize(med_entities)
        narrated_summary = self.narrator_summator.narrate(med_entities)
        return structured_summary, narrated_summary