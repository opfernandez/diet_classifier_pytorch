import torch
import sys
import os
import time
import json
import socket
import threading

from data_loader import DataLoader
sys.path.append("../model")
from diet import DIETModel
from sparse_features_extractor import SparseFeatureExtractor



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
                print(result["text"].split()[idx])
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
    
    def run(self, host: str = '0.0.0.0', port: int = 5555):
        """Run the inference server on a socket."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        
        print(f"DIET Inference Server listening on {host}:{port}")
        print("Waiting for connections...")
        
        try:
            while True:
                client_socket, client_address = server_socket.accept()
                print(f"Connection from {client_address}")
                
                # Handle client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            server_socket.close()
    
    def _handle_client(self, client_socket, client_address):
        """Handle a client connection."""
        try:
            while True:
                # Receive data from client
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                print(f"Received from {client_address}: {data}")
                
                try:
                    # Parse JSON request
                    request = json.loads(data)
                    text_input = request.get("text", "")
                    
                    if not text_input:
                        response = {
                            "error": "No text provided",
                            "status": "error"
                        }
                    else:
                        # Perform inference
                        results = self.predict([text_input])
                        response = {
                            "status": "success",
                            "result": results[0]
                        }
                    
                except json.JSONDecodeError:
                    response = {
                        "error": "Invalid JSON format",
                        "status": "error"
                    }
                except Exception as e:
                    response = {
                        "error": str(e),
                        "status": "error"
                    }
                
                # Send response back to client
                response_json = json.dumps(response, ensure_ascii=False) + "\n"
                client_socket.sendall(response_json.encode('utf-8'))
                
        except Exception as e:
            print(f"Error handling client {client_address}: {e}")
        finally:
            client_socket.close()
            print(f"Connection closed: {client_address}")
            


def main():
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Hyperparameters
    entities_file = os.path.join(script_dir, "../data/entity_labels.json")
    intents_file =  os.path.join(script_dir, "../data/intent_labels.json")
    model_path = os.path.join(script_dir, "../model/diet_model.pt")
    word_dict_path = os.path.join(script_dir, "../data/word_dict.json")
    ngram_dict_path = os.path.join(script_dir, "../data/ngram_dict.json")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Initialize DIETServer
    server = DIETServer(
        device=device,
        model_path=model_path,
        word_dict_path=word_dict_path,
        ngram_dict_path=ngram_dict_path,
        entity_labels_path=entities_file,
        intent_labels_path=intents_file
    )
    
    # Start the server
    server.run(host='0.0.0.0', port=5555)


if __name__ == "__main__":
    main()
