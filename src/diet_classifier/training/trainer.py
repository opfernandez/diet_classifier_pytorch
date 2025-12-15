import torch
import sys
import matplotlib.pyplot as plt

from data_loader import DataLoader
sys.path.append("../model")
from diet import DIETModel


class Trainer:
    def __init__(self, 
                 model: DIETModel = None, 
                 device: str = 'cuda',
                 lr: float = 1e-3,
                 epochs: int = 100,
                 data_loader: DataLoader = None,
                 checkpoint_path: str = None,
                 checkpoint_name: str = None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.loss_hstr = []
        self.data_loader = data_loader
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is None:
            self.checkpoint_path = "../model"
            print(f"Model checkpoints will be saved to {self.checkpoint_path}")
        if self.checkpoint_name is None:
            self.checkpoint_name = "diet_model.pt"
            print(f"Model checkpoint name: {self.checkpoint_name}")
    
    def train_step(self, batch):
        # Batches comes as a list of samples (dictionaries)
        # 1. Fromat the batch
        (text_inputs, entity_tag_indices, 
         one_hot_intent_labels) = self.data_loader.format_batch(batch)
        # 2. Move to device
        entity_tag_indices = entity_tag_indices.to(self.device)
        one_hot_intent_labels = one_hot_intent_labels.to(self.device)
        # 3. Reset gradients
        self.optimizer.zero_grad()
        # 4. Forward pass
        loss = self.model.train_forward(
            input_texts=text_inputs,
            entity_labels=entity_tag_indices,
            intent_labels=one_hot_intent_labels)
        # 5. Backward pass
        loss.backward()
        self.optimizer.step()
        # Fix values of transition matrix for PAD tag
        with torch.no_grad():
            pad_idx = self.model.entity_pad_idx
            self.model.crf.transitions[pad_idx, :] = -1e4  # No transition from PAD to any tag
            self.model.crf.transitions[:, pad_idx] = -1e4  # No transition to PAD from any tag
        return loss.item()
    
    def train(self):
        self.model.train()
        last_loss = None
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_idx, batch in enumerate(self.data_loader.data):
                loss = self.train_step(batch)
                total_loss += loss
            avg_loss = total_loss / self.data_loader.num_batches
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            if last_loss is None or avg_loss < last_loss:
                last_loss = avg_loss
                # Save model checkpoint
                torch.save(self.model.state_dict(), 
                            f"{self.checkpoint_path}/{self.checkpoint_name}")
                print("--"*30)
                print(f"Model checkpoint saved to {self.checkpoint_path}/{self.checkpoint_name}")
                print("--"*30)
            self.loss_hstr.append(avg_loss)

    def plot_loss_history(self):
        plt.plot(range(1, len(self.loss_hstr)+1), self.loss_hstr)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss History")
        plt.show()

    def __call__(self):
        self.train()