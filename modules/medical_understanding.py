class MedicalUnderstander:
    def __init__(self, extractor, validator=None):
        self.extractor = extractor
        self.validator = validator

    def understand(self, diarized_transcript: list[dict]) -> dict:
        normalized = self.normalize_transcript(diarized_transcript)
        utterances = self.segment_utterances(normalized)
        extracted = self.extract_medical_facts(utterances)
        resolved = self.resolve_context(extracted)
        validated = self.validate_output(resolved)
        return validated