
from structuredSummary import StructuredSummaryGenerator
class MedicalSummarizer:
    def __init__(self):
        self.structured_summator = StructuredSummaryGenerator()
    def summarize(self, med_entities: dict[dict]) -> str:
        structured_summary = self.structured_summator.structurize(med_entities)
        print(structured_summary)