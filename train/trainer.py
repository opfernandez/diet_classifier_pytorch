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
                 data_loader: DataLoader = None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.loss_hstr = []
        self.data_loader = data_loader
    
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
        return loss.item()
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_idx, batch in enumerate(self.data_loader.data):
                loss = self.train_step(batch)
                total_loss += loss
            avg_loss = total_loss / self.data_loader.num_batches
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            self.loss_hstr.append(avg_loss)

    def plot_loss_history(self):
        plt.plot(range(1, len(self.loss_hstr)+1), self.loss_hstr)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss History")
        plt.show()

    def __call__(self):
        self.train()