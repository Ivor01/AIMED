from extractor import MedicalExtractor


class MedicalUnderstander:
    def __init__(self, validator=None):
        #self.input_validator = MedicalInputValidator()
        self.extractor = MedicalExtractor()
        #self.rule_engine = CroatianMedicalRuleEngine()
        #self.merger = MedicalEntityMerger()
        #self.safety_validator = MedicalSafetyValidator()

    def understand(self, diarized_transcript: list[dict]) -> dict: #-> znači očekujemo da funkcija vraća objekt dict 

        #utterances = self.input_validator.validate(input_segments)
        extracted = self.extractor.extract(diarized_transcript)
        #corrected = self.rule_engine.apply(extracted)

        #merged = self.merger.merge(corrected)

        #validated = self.safety_validator.validate(
        #    output=merged,
        #    utterances=utterances,
        #)
        
        return "TEMP"