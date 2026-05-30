from modules.electronicRecords.internalEHR import EHRBuilder
from modules.electronicRecords.ehrExporter import EHRJsonExporter
from modules.electronicRecords.fhirMapper import FHIRMapper
class ElectronicRecorder:
    def __init__(self):
        pass
    def record(self, input_metadata, med_entities, medical_summary):
        ###OBRADA ulaznih arg u korisne dict ili sl
        ####
        builder = EHRBuilder()
        internalEHR = builder.build(input_metadata, med_entities, medical_summary)

        exporter = EHRJsonExporter()
        
        mapper = FHIRMapper()
        fhir_bundle = mapper.to_minimal_bundle(internalEHR)

        exporter.save_to_file(
            ehr_record=internalEHR,
            output_path="outputs/ehr/internal_ehr_demo.json",
        )
        exporter.save_dict_to_file(
            data=fhir_bundle,
            output_path="outputs/ehr/fhir_bundle_demo.json",
        )

        
        