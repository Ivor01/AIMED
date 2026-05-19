from openai import OpenAI

from schemas import MedicalUnderstandingOutput, InputUtterance
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

class MedicalExtractor:
    def __init__(self, model = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model = model
    
    def extract(self, segmenti):
        transcript = self.segs_to_text(segmenti)
        prompt = USER_PROMPT_TEMPLATE.format(transcript=transcript)
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
            ], text_format = MedicalUnderstandingOutput
        )
        return response.output_parsed
    def segs_to_text(self,segs):
        lines = []
        for seg in segs:
            lines.append(
                f"utterance_id:{seg.utterance_id} | speaker:{seg.speaker} |"
                f"start-end:{seg.start}-{seg.end} text: {seg.text}"
            )

        return "\n".join(lines)