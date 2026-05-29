import json
from openai import OpenAI

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from schemas import StyledMedicalNote

class NarratedSummaryGenerator:
    def __init__(self, model = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model = model
    
    def narrate(self, med_entities):
        
        med_entities_json_str = json.dumps(
            med_entities,
            ensure_ascii=False,
            indent=2
        )
        prompt = USER_PROMPT_TEMPLATE.format(med_understanding_json=med_entities)
        response = self.client.responses.parse(
            model = self.model,
            input = [{
                "role":"system",
                "content":SYSTEM_PROMPT
            },
            {
                "role":"user",
                "content":prompt,
            },
            ], text_format=StyledMedicalNote
        )
        return (response.output_parsed).model_dump()
    #Višak?
    def segs_to_text(self,segs):
        lines = []
        for seg in segs:
            lines.append(
                f"utterance_id:{seg.utterance_id} | speaker:{seg.speaker} |"
                f"start-end:{seg.start}-{seg.end} text: {seg.text}"
            )

        return "\n".join(lines)