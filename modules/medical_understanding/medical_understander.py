class MedicalUnderstander:
    def __init__(self, extractor, validator=None):
        self.extractor = extractor
        self.validator = validator
    def understand(self, diarized_transcript: list[dict]) -> dict: #-> znači očekujemo da funkcija vraća objekt dict 
        
        extracted = self.extract_medical_facts(diarized_transcript)
        
        return "TEMP"