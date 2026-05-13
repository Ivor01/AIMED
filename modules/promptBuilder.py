# -----------------------------
# Prompt handling
# -----------------------------
import json
import os


def build_initial_prompt(odjel, tema):
    prompt_parts = []

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"prompts")

    def load_json(filename):
        if not filename:
            return []

        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Expecting: [{ "text": "..." }, ...]
        return [
            item["text"].strip()
            for item in data
            if isinstance(item, dict) and "text" in item
        ]

    
    if odjel.lower() == "opća":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/intern")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"misc.json"))
        elif tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"misc.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        elif tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/doctor.json"))
            prompt_parts.extend(load_json(f"consults/patient.json"))

    if odjel.lower() == "kardiologija":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/cardio")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))
    
    if odjel.lower() == "pulmologija":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/pulmo")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))

    if odjel.lower() == "gastrologija":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/gastro")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))

    if odjel.lower() == "neurologija":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/neuro")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))

    if odjel.lower() == "infektologija":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/infecto")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))

    if odjel.lower() == "hitna medicina":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/emergency")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))
        
    return " ".join(prompt_parts)

def combine_jsons(folder_path):
    combined_texts = []

    if not os.path.exists(folder_path):
        return combined_texts  # Folder does not exist, return empty list

    # Loop over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Extract "text" field
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        combined_texts.append(item["text"].strip())
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    return combined_texts
