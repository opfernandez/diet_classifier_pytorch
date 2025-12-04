import torch
from torch import nn
import re
import sys
import yaml

# rasa expample:
# nlu:
# - intent: encender_luz
#   examples: |
#     - enciende la luz del [salon](sala)
#     - por favor, enciende la luz de la [cocina](sala)
#     - quiero que enciendas la luz del [dormitorio](sala)
#     - activa la luz del [aseo](sala)

# We want to extract in format:
# {
#   "text": "enciende la luz del salon",
#   "entity_tags": ["O", "O", "O", "O", "B-sala"], # BIO format
#   "intent": "encender_luz"
# }
class DataLoader:
    def __init__(self, data_path: str, batch_size: int = 32):
        self.data_path = data_path
        self.batch_size = batch_size
        self.data = self.load_data()
        self.num_batches = len(self.data) // batch_size + (1 if len(self.data) % batch_size != 0 else 0)

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        samples = []
        for item in data.get("nlu", []):
            intent = item.get("intent")
            examples = item.get("examples", "")
            for line in examples.strip().split("\n"):
                line = line.strip().lstrip("-").strip()
                for part in re.split(r"(\[.*?\]\(.*?\))", line):
                    if re.match(r"\[.*?\]\(.*?\)", part):
                        # Entity part
                        entity_text = re.findall(r"\[(.*?)\]\((.*?)\)", part)[0][0]
                        entity_label = re.findall(r"\[(.*?)\]\((.*?)\)", part)[0][1]
                        samples.append({
                            "text": entity_text,
                            "entity_tags": [f"B-{entity_label}"] + ["I-{entity_label}"] * (len(entity_text.split()) - 1),
                            "intent": intent
                        })
                    else:
                        # Plain text part
                        if part.strip():
                            samples.append({
                                "text": part.strip(),
                                "entity_tags": ["O"] * len(part.strip().split()),
                                "intent": intent
                            })
        return samples