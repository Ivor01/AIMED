class StructuredSummaryGenerator:
    def structurize(self, entities):
        med_entities = entities["entities"]
        symptoms = self.get_symptoms(med_entities)
        history= self.get_history(med_entities)
        #family_history, family_history_neg = self.get_history(med_entities)
        findings = self.get_findings(med_entities)
        medicine_curr, medicine_prev = self.get_medicine(med_entities)
        negateds = self.get_negs(med_entities)
        #plan = self.get_plan(plan_entities)
        iskljuceno = ""
        
        for neg_kat, neg_val in negateds.items():
            iskljuceno+= f"{str(neg_kat).capitalize()}: "
            for val in neg_val:
                iskljuceno+= f"{str(val)};"
            iskljuceno+= "\n"
            
        structured_summary = f"Prisutne tegobe:\n{symptoms}\nNalazi:\n{findings}\nPovijest bolesti:\n{history}\nTerapije - trenutne:\n- {", ".join(medicine_curr)}\n"
        structured_summary += f"Terapije - prijašnje:\n- {", ".join(medicine_prev)}\n\nIsključeno:\n- {iskljuceno}"
        return structured_summary
    def get_symptoms(self, med_entities):
        symptoms = ""
        symptoms_neg = ""
        for entity in med_entities:
            if entity["kategorija"] == "simptom":
                if entity["status"] == "sadašnje":
                    if entity["normalized_name"] != "NA":
                        symptoms+= f"- {entity["normalized_name"]}; "
                    else:
                        symptoms+= f"- {entity["text"]}; "
                    if entity["duration"] != "NA":
                        symptoms+= f"Trajanje: {entity["duration"]}; "
                    if entity["severity"] != "NA":
                        symptoms+= f"Intezitet: {entity["severity"]}; "
                    if entity["body_location"] != "NA":
                        symptoms+= f"Mjesto na tijelu: {entity["body_location"]}; "
                    symptoms+= "\n"       
        return symptoms

    def get_history(self, med_entities):
        symptoms = ""
        for entity in med_entities:
            if entity["kategorija"]=="simptom":
                if entity["status"]=="povijest":
                    if entity["normalized_name"] != "NA":
                        symptoms+= f"- {entity["normalized_name"]};\n"
                    else:
                        symptoms+= f"- {entity["text"]};\n"
        return symptoms

    def get_findings(self, med_entities):
        symptoms = ""
        for entity in med_entities:
            if entity["kategorija"]=="nalaz":
                if entity["status"] != "negated":
                     symptoms+= f"- {entity["normalized_name"]};\n"
        return symptoms


    def get_medicine(self, med_entities):
        drugs_curr = []
        drugs_prev = []
        for entity in med_entities:
            if entity["kategorija"]=="lijekovi":
                if entity["status"]=="sadašnje":
                    drugs_curr.append(entity["normalized_name"])
                elif entity["status"]=="povijest":
                    drugs_prev.append(entity["normalized_name"])
                    
        return drugs_curr, drugs_prev
    
    def get_negs(self, med_entities):
        negs = {}
        for entity in med_entities:
            if entity["status"] in ("negated","isključeno"):
                if entity["kategorija"] not in negs:
                    negs[entity["kategorija"]] = []
                negs[entity["kategorija"]].append(entity["normalized_name"])

        return negs



