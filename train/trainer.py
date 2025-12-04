import torch
from torch import nn
import re
import sys

sys.path.append("../model")
from sparse_features_extractor import SparseFeatureExtractor
from crf import CRF
from diet import DIETModel

class Trainer:
    def __init__(self, 
                 model: DIETModel, 
                 device: str = 'cuda',
                 lr: float = 1e-3,
                 intent_labels: list = None,
                 entity_labels: list = None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion_intent = nn.CrossEntropyLoss()
    
    def train_step(self, batch):
        # self.model.train()
        # self.optimizer.zero_grad()

        # inputs, sparse_features, entity_tags, intent_tags, mask = batch
        # inputs = inputs.to(self.device)
        # sparse_features = sparse_features.to(self.device)
        # entity_tags = entity_tags.to(self.device)
        # intent_tags = intent_tags.to(self.device)
        # mask = mask.to(self.device)

        # emissions_entity, intent_logits = self.model(inputs, sparse_features, mask)

        # loss_entity = self.model.crf.compute_loss(emissions_entity, entity_tags, mask)
        # loss_intent = self.criterion_intent(intent_logits, intent_tags)

        # total_loss = loss_entity + loss_intent
        # total_loss.backward()
        # self.optimizer.step()

        # return total_loss.item()