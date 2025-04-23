import spacy
import re

nlp = spacy.load("en_core_web_sm")

def parse_instruction(text: str) -> list:
    commands = []

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.lower())

    # Match all actionable phrases
    matches = re.findall(r"(search for .*?)(?=remind me to|$)", text) + \
              re.findall(r"(remind me to .*?)(?=search for|$)", text)

    for part in matches:
        part = part.strip()

        if part.startswith("search for"):
            query = part.replace("search for", "").strip()
            commands.append({"action": "search", "query": query})

        elif part.startswith("remind me to"):
            time_match = re.search(r"(?:at|by)\s+(\d{1,2}(?:am|pm))", part)
            time = time_match.group(1) if time_match else None
            task = re.sub(r"remind me to|at\s+\d{1,2}(?:am|pm)", "", part).strip()
            commands.append({"action": "reminder", "task": task, "time": time})

    return commands
