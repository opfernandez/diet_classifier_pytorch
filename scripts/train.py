import torch
import os
import time
import json

from diet_classifier.training import DataLoader, Trainer
from diet_classifier.model import DIETModel, SparseFeatureExtractor

def load_json(filepath: str) -> dict:
        """Load JSON file."""
        try:
            with open(filepath, "r", encoding="utf8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            raise

def main():
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Hyperparameters
    batch_size = 32
    lr = 1e-3
    epochs = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Load data
    data_path = os.path.join(script_dir, "../data/data.yml")
    entity_labels = load_json(os.path.join(script_dir, "../data/entity_labels.json"))
    intent_labels = load_json(os.path.join(script_dir, "../data/intent_labels.json"))

    # Initialize DataLoader
    data_loader = DataLoader(data_path=data_path, 
                             batch_size=batch_size,
                             intent_labels=intent_labels,
                             entity_labels=entity_labels,
                             cls_token="[CLS]",
                             pad_token="[PAD]",
                             pad_entity_tag="PAD")
    
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

    # Generate word and ngram dictionaries from training data
    all_texts = [sample["text"] for batch in data_loader.data for sample in batch]
    sparse_extractor.build_word_dict(all_texts)
    sparse_extractor.build_ngram_dict(all_texts)
    # Save dictionaries
    sparse_extractor.save_dicts(
        os.path.join(script_dir, "../data/word_dict.json"),
        os.path.join(script_dir, "../data/ngram_dict.json")
    )

    # Initialize DIET model
    model = DIETModel(
        device=device,
        sparse_extractor=sparse_extractor,
        num_entity_tags=len(entity_labels),
        num_intent_tags=len(intent_labels),
        pad_entity_tag_idx=entity_labels.index("PAD"),
        eos_entity_tag_idx=entity_labels.index("EOS"),
        bos_entity_tag_idx=entity_labels.index("BOS")
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        device=device,
        lr=lr,
        epochs=epochs,
        data_loader=data_loader
    )

    # Start training
    trainer.train()
    # Plot loss history
    trainer.plot_loss_history()

    # Test saved model
    model_path = os.path.join(trainer.checkpoint_path, trainer.checkpoint_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    test_sentence1 = "encender la luz de la cocina"
    test_sentence2 = "apagar el enchufe del televisor [PAD]"
    input_batch = [test_sentence1, test_sentence2]
    with torch.no_grad():
        tensor_entities, tensor_intent = model(input_batch)

    # Print results
    print("--"*30)
    print("Available intents:", intent_labels)
    print("Available entity tags:", entity_labels)
    print("--"*30)
    for b, sentence in enumerate(input_batch):
        init_time = time.perf_counter()
        print(f"Sentence: '{sentence}'")
        print("Predicted entity tags:", [entity_labels[idx] for idx in tensor_entities[b].tolist()])
        print("Predicted intent tensor:", tensor_intent[b])
        print("Predicted intent:", intent_labels[torch.argmax(tensor_intent[b]).item()])
        end_time = time.perf_counter()
        print(f"Inference time: {(end_time - init_time)*1000:.2f} ms")
        print("--"*30)

if __name__ == "__main__":
    main()