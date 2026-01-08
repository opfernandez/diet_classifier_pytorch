import torch
import sys
import os
import time
import json
import socket
import threading

from ..model.diet import DIETModel
from ..model.sparse_features_extractor import SparseFeatureExtractor

class DIETServer:
    def __init__(self, 
                 device: str = 'cuda',
                 model_path: str = None,
                 word_dict_path: str = None,
                 ngram_dict_path: str = None,
                 entity_labels_path: list = None,
                 intent_labels_path: list = None):
        
        if word_dict_path is None or ngram_dict_path is None:
            raise ValueError("Word and ngram dictionary paths must be provided.")
        if entity_labels_path is None or intent_labels_path is None:
            raise ValueError("Entity and intent labels paths must be provided.")
        if model_path is None:
            raise ValueError("Model path must be provided.")

        self.device = device
        # Create SparseFeatureExtractor instance
        sparse_extractor = SparseFeatureExtractor(
            word_dict_size=300,
            ngram_dict_size=1000,
            ngram_overflow_size=100,
            ngram_min=2,
            ngram_max=5,
            pad_token="[PAD]",
            cls_token="[CLS]",
            unk_token="[UNK]"
        )

        # Load dictionaries
        sparse_extractor.load_dicts(
            word_dict_path,
            ngram_dict_path
        )

        # Load entity and intent labels
        self.entity_labels = self._load_json(entity_labels_path)
        self.intent_labels = self._load_json(intent_labels_path)

        # Initialize DIET model
        self.model = DIETModel(
            device=device,
            sparse_extractor=sparse_extractor,
            num_entity_tags=len(self.entity_labels),
            num_intent_tags=len(self.intent_labels),
            pad_entity_tag_idx=self.entity_labels.index("PAD"),
            eos_entity_tag_idx=self.entity_labels.index("EOS"),
            bos_entity_tag_idx=self.entity_labels.index("BOS")
        )

        # Load model weights
        self._load_model(model_path)

    def _load_json(self, filepath: str) -> dict:
        """Load JSON file."""
        try:
            with open(filepath, "r", encoding="utf8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            raise
    
    def _load_model(self, model_path: str):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)  # Move model to the correct device
        self.model.eval()
        print(f"Model loaded from {model_path} and set to eval mode.")
    
    def predict(self, text_inputs: list[str]):
        """Perform inference on the input text list."""
        init_time = time.perf_counter()
        with torch.no_grad():
            tensor_entities, tensor_intent = self.model(text_inputs)
        end_time = time.perf_counter()
        inference_time = (end_time - init_time) * 1000  # Convert to ms
        
        # Format results
        results = []
        for b in range(len(text_inputs)):
            predicted_entities = tensor_entities[b].tolist()
            predicted_intent_idx = torch.argmax(tensor_intent[b]).item()
            
            result = {
                "text": text_inputs[b],
                "intent": self.intent_labels[predicted_intent_idx],
                "intent_confidence": float(tensor_intent[b][predicted_intent_idx]),
                "entities": [self.entity_labels[idx] for idx in predicted_entities],
                "inference_time_ms": inference_time
            }
            results.append(result)
        print("=="*40)
        print("Inference results:\n", results)
        print("=="*40)
        for result in results:
            result = self.format_entities(result)
        
        return results
    
    def format_entities(self, result: dict) -> dict:
        """Format the entities from the result dictionary to a readable output."""
        # Convert entities list into spans
        entities = []
        current_entity = None
        for idx, tag in enumerate(result["entities"][1:]):
            if tag.startswith("B-"):
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {
                    "type": tag[2:],
                    "start": idx,
                    "end": idx,
                    "words": result["text"].split()[idx]
                }
            elif tag.startswith("I-") and current_entity is not None:
                current_entity["end"] = idx
                current_entity["words"] += " " + result["text"].split()[idx]
            else:
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        if current_entity is not None:
            entities.append(current_entity)
        result["entities"] = entities
        return result
