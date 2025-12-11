import torch
import re
import random
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
    def __init__(self, data_path: str, 
                 batch_size: int = 32,
                 intent_labels: list[str] = None,
                 entity_labels: list[str] = None,
                 cls_token: str = "[CLS]",
                 pad_token: str = "[PAD]",
                 pad_entity_tag: str = "PAD"):
        self.data_path = data_path
        self.batch_size = batch_size
        self.intent_labels = intent_labels
        self.entity_labels = entity_labels
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.pad_entity_tag = pad_entity_tag
        # Load data
        self.data = self.load_data()
        self.num_batches = len(self.data) // batch_size + (1 if len(self.data) % batch_size != 0 else 0)

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        samples = []
        for item in data.get("nlu", []):
            intent = item.get("intent")
            examples = item.get("examples", "")
            for sentence_idx, line in enumerate(examples.strip().split("\n")):
                line = line.strip().lstrip("-").strip()
                for part in re.split(r"(\[.*?\]\(.*?\))", line):
                    if re.match(r"\[.*?\]\(.*?\)", part):
                        # Entity part
                        entity_text = re.findall(r"\[(.*?)\]\((.*?)\)", part)[0][0]
                        entity_label = re.findall(r"\[(.*?)\]\((.*?)\)", part)[0][1]
                        # Check if we can append to the last sample
                        if len(samples) > 0 and samples[-1]["index"] == sentence_idx:
                            samples[-1]["text"] += " " + entity_text
                            samples[-1]["entity_tags"].extend([f"B-{entity_label}"] + [f"I-{entity_label}"] * (len(entity_text.split()) - 1))
                        else:
                            # Otherwise, create a new sample
                            samples.append({
                                "text": self.cls_token + " " + entity_text,
                                "entity_tags": ["O"] + [f"B-{entity_label}"] + [f"I-{entity_label}"] * (len(entity_text.split()) - 1),
                                "intent": intent,
                                "index": sentence_idx
                            })
                    else:
                        # Plain text part
                        if part.strip():
                            # Check if we can append to the last sample
                            if len(samples) > 0 and samples[-1]["index"] == sentence_idx:
                                samples[-1]["text"] += " " + part.strip()
                                samples[-1]["entity_tags"].extend(["O"] * len(part.strip().split()))
                            else:
                                # Otherwise, create a new sample
                                samples.append({
                                    "text": self.cls_token + " " + part.strip(),
                                    "entity_tags": ["O"] * (len(part.strip().split()) + 1),
                                    "intent": intent,
                                    "index": sentence_idx
                            })
        # Group samles into batches randomly
        random.shuffle(samples)
        batches = [samples[i:i + self.batch_size] for i in range(0, len(samples), self.batch_size)]
        return batches

    def format_batch(self, batch):
        text_inputs = []
        one_hot_intent_labels = torch.zeros(len(batch), len(self.intent_labels), dtype=torch.float32)
        max_seq_len = max(len(sample["entity_tags"]) for sample in batch)
        entity_tag_indices = torch.zeros(len(batch), max_seq_len, dtype=torch.long)
        for i, sample in enumerate(batch):
            # Create copies to avoid modifying the original sample in the dataset
            text = sample["text"]
            entity_tags = sample["entity_tags"].copy()
            # Add padding tokens to text sample
            seq_len = len(entity_tags)
            padding_needed = max_seq_len - seq_len
            if padding_needed > 0:
                text += " " + " ".join([self.pad_token] * padding_needed)
                entity_tags.extend([self.pad_entity_tag] * padding_needed)
            # List of text inputs
            text_inputs.append(text)
            # Tensor of entity tag labels (indices)
            for j, tag in enumerate(entity_tags):
                try:
                    entity_tag_indices[batch.index(sample), j] = self.entity_labels.index(tag)
                except ValueError:
                    raise ValueError(f"Tag {tag} not found in entity labels list.")
            # One-hot tensor of intent labels
            try:
                intent_idx = self.intent_labels.index(sample["intent"])
                one_hot_intent_labels[i, intent_idx] = 1.0
            except ValueError:
                raise ValueError(f"Intent {sample['intent']} not found in intent labels list.")
        return text_inputs, entity_tag_indices, one_hot_intent_labels