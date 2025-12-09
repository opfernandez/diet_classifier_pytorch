import torch
import sys
import os
import matplotlib.pyplot as plt

from data_loader import DataLoader
from trainer import Trainer
sys.path.append("../model")
from diet import DIETModel

def main():
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Hyperparameters
    data_path = os.path.join(script_dir, "../data/data.yml")
    batch_size = 32
    lr = 1e-3
    epochs = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    entity_labels = ["O", "B-sala", "I-sala", "B-dispositivo", "I-dispositivo"]
    intent_labels = ["encender_luz", "apagar_luz", 
                     "apagar_enchufe", "activar_enchufe",
                     "subir_persiana", "bajar_persiana", "parar_persiana"]

    # Initialize DataLoader
    data_loader = DataLoader(data_path=data_path, 
                             batch_size=batch_size,
                             intent_labels=intent_labels,
                             entity_labels=entity_labels,
                             cls_token="[CLS]")

    # Initialize DIET model
    model = DIETModel(
        device=device,
        word_dict_size=300,
        ngram_dict_size=1000,
        ngram_min=2,
        ngram_max=5,
        num_entity_tags=len(entity_labels),
        num_intent_tags=len(intent_labels)
    )

    # Generate word and ngram dictionaries from training data
    all_texts = [sample["text"] for batch in data_loader.data for sample in batch]
    model.sparse_extractor.build_word_dict(all_texts)
    model.sparse_extractor.build_ngram_dict(all_texts)
    # Save dictionaries
    model.sparse_extractor.save_dicts(
        os.path.join(script_dir, "../data/word_dict.json"),
        os.path.join(script_dir, "../data/ngram_dict.json")
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

if __name__ == "__main__":
    main()