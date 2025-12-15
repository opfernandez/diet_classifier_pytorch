import os
import torch
from diet_classifier.inference.server import DIETServer

def main():
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Config files
    entities_file = os.path.join(script_dir, "../data/entity_labels.json")
    intents_file =  os.path.join(script_dir, "../data/intent_labels.json")
    model_path = os.path.join(script_dir, "../models/diet_model.pt")
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