import torch
import sys
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data_loader import DataLoader
from trainer import Trainer
sys.path.append("../model")
from diet import DIETModel
from sparse_features_extractor import SparseFeatureExtractor

def main():
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Hyperparameters
    data_path = os.path.join(script_dir, "../data/validation.yml")
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    entity_labels = ["EOS", "BOS", "PAD", "O", "B-sala", "I-sala", 
                     "B-dispositivo", "I-dispositivo"]
    intent_labels = ["encender_luz", "apagar_luz", 
                     "apagar_enchufe", "activar_enchufe",
                     "subir_persiana", "bajar_persiana", "parar_persiana"]

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

    # Load dictionaries
    sparse_extractor.load_dicts(
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

    # Test saved model
    model_path = "../model/diet_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move model to the correct device
    model.eval()

    # Print Classification Info
    print("--"*30)
    print("Available intents:", intent_labels)
    print("Available entity tags:", entity_labels)
    print("--"*30)
    
    # For entity F1 score (excluding PAD, BOS, EOS, O)
    entity_true_positives = {label: 0 for label in entity_labels if label not in ["PAD", "BOS", "EOS", "O"]}
    entity_false_positives = {label: 0 for label in entity_labels if label not in ["PAD", "BOS", "EOS", "O"]}
    entity_false_negatives = {label: 0 for label in entity_labels if label not in ["PAD", "BOS", "EOS", "O"]}
    
    total_samples = 0
    correct_intents = 0

    # Do inference on validation data
    # Compute F1 score for entities and accuracy for intents
    # and confusion matrix for intents
    entity_preds = []
    intent_preds = []
    entity_true = []
    intent_true = []
    for batch in data_loader.data:
        # 1. Fromat the batch
        (text_inputs, entity_tag_indices, 
         one_hot_intent_labels) = data_loader.format_batch(batch)
        # Do inference
        init_time = time.perf_counter()
        with torch.no_grad():
            tensor_entities, tensor_intent = model(text_inputs)
        end_time = time.perf_counter()
        print(f"Inference time for batch: {(end_time - init_time)*1000:.2f} ms")
        
        # Process each sample in batch
        for b, sentence in enumerate(text_inputs):
            # Get predicted and true intents
            predicted_intent_idx = torch.argmax(tensor_intent[b]).item()
            true_intent_idx = torch.argmax(one_hot_intent_labels[b]).item()
            intent_preds.append(predicted_intent_idx)
            intent_true.append(true_intent_idx)
            
            total_samples += 1
            
            if predicted_intent_idx == true_intent_idx:
                correct_intents += 1
            
            # Get predicted and true entity tags
            predicted_entities = tensor_entities[b].tolist()
            true_entities = entity_tag_indices[b]
            entity_preds.append(predicted_entities)
            entity_true.append(true_entities.tolist())
            
            # Calculate entity metrics (excluding special tokens)
            for i, (pred_idx, true_idx) in enumerate(zip(predicted_entities, true_entities)):
                pred_label = entity_labels[pred_idx]
                true_label = entity_labels[true_idx]
                
                # Skip special tokens
                if true_label in ["PAD", "BOS", "EOS", "O"] and pred_label in ["PAD", "BOS", "EOS", "O"]:
                    continue
                
                # Count entity matches
                if true_label not in ["PAD", "BOS", "EOS", "O"]:
                    if pred_label == true_label:
                        entity_true_positives[true_label] += 1
                    else:
                        entity_false_negatives[true_label] += 1
                        if pred_label not in ["PAD", "BOS", "EOS", "O"]:
                            entity_false_positives[pred_label] += 1
                elif pred_label not in ["PAD", "BOS", "EOS", "O"]:
                    entity_false_positives[pred_label] += 1
            
            # Print results
            print(f"Sentence: '{sentence}'")
            print("Predicted entity tags:", [entity_labels[idx] for idx in predicted_entities])
            print("True entity tags:", [entity_labels[idx] for idx in true_entities])
            print("Predicted intent:", intent_labels[predicted_intent_idx])
            print("True intent:", intent_labels[true_intent_idx])
            print("--"*30)

    # Calculate and print metrics
    print("\n" + "=="*30)
    print("VALIDATION METRICS")
    print("=="*30)
    
    # -----------------------------------------------------------------------------------

    # Intent accuracy
    intent_accuracy = (correct_intents / total_samples) * 100 if total_samples > 0 else 0
    print(f"\nIntent Accuracy: {intent_accuracy:.2f}% ({correct_intents}/{total_samples})")

    # -----------------------------------------------------------------------------------
    
    # Entity F1 Score
    print("\n" + "--"*30)
    print("Entity Recognition Metrics (F1 Score):")
    print("--"*30)
    
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    
    for label in entity_true_positives.keys():
        tp = entity_true_positives[label]
        fp = entity_false_positives[label]
        fn = entity_false_negatives[label]
        
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{label:20s} - Precision: {precision:.3f}, Recall: {recall:.3f}," + 
              f"F1: {f1:.3f} (TP:{tp}, FP:{fp}, FN:{fn})")
    
    # -----------------------------------------------------------------------------------
    
    # Confusion Matrix for Intents
    print("\nConfusion Matrix for Intents:")
    confusion_matrix_intent = confusion_matrix(intent_true, 
                                               intent_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_intent, 
                                  display_labels=intent_labels)
    disp.plot(cmap='Blues')
    plt.show()

if __name__ == "__main__":
    main()