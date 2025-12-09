import torch
from torch import nn
from sparse_features_extractor import SparseFeatureExtractor
from crf import CRF

class DIETModel(nn.Module):
    def __init__(self, word_dict_size: int = 300,
                 ngram_dict_size: int = 1000, 
                 ngram_min: int = 2, 
                 ngram_max: int = 5, 
                 embed_dims: int = 512,
                 tf_layers: int = 2,
                 tf_dims: int = 256,
                 tf_n_heads: int = 8,
                 device: str = 'cuda',
                 pad_token: str = "[PAD]",
                 cls_token: str = "[CLS]",
                 unk_token: str = "[UNK]",
                 num_entity_tags: int = 10,
                 num_intent_tags: int = 5):
        
        # Initialize the DIET model
        super(DIETModel, self).__init__()

        # Intent criterion
        self.intent_criterion = nn.CrossEntropyLoss()

        # Sparse extractor
        self.sparse_extractor = SparseFeatureExtractor(
            word_dict_size=word_dict_size,
            ngram_dict_size=ngram_dict_size,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            pad_token=pad_token,
            cls_token=cls_token,
            unk_token=unk_token
        )

        # Instead of big sparse vectors + FFL we use embeddings
        self.pad_idx = self.sparse_extractor.token_to_word_index(pad_token)
        self.word_emb = nn.Embedding(word_dict_size, embed_dims, padding_idx=self.pad_idx)
        self.ngram_emb_bag =  nn.EmbeddingBag(ngram_dict_size, embed_dims, mode="sum")
        # TODO: add dropout and do pruning 
        self.fc2 = nn.Linear(embed_dims, tf_dims) 
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_dims,      # embedding dimension
            nhead=tf_n_heads,        # number of attention heads
            dim_feedforward=tf_dims*4,  # feedforward hidden dimension
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=tf_layers
        )
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(tf_dims)
        self.dropout = nn.Dropout(0.8)
        # Conditional Random Field (CRF) for sequence entity labeling
        self.crf_ff = nn.Linear(tf_dims, num_entity_tags)
        self.crf = CRF(num_tags=num_entity_tags, pad_idx=self.pad_idx)
        # Linear layer for intent classification
        self.intent_ff = nn.Linear(tf_dims, num_intent_tags)
        # Select device
        self.device = device
        # Special tokens token
        self.pad_token = pad_token
        self.unkg_token = unk_token
        self.cls_token = cls_token
    
    def compute_sparse_features(self, input_texts: list[str]):
        """
        Compute sparse features for a batch of input texts.
        Returns:
            word_indices: [B, S] LongTensor
            ngram_indices: List of ngram indices for EmbeddingBag
            offsets: List of offsets for EmbeddingBag
            padding_mask: [B, S] BoolTensor (True=pad)
        """
        # 1. Tokenization
        tokenized_batch = [self.sparse_extractor.tokenizer(text) for text in input_texts]
        batch_size = len(tokenized_batch)
        seq_lens = [len(seq) for seq in tokenized_batch]
        max_len = max(seq_lens)
        # Add padding
        padded_batch = [seq + [self.pad_token] * (max_len - len(seq)) for seq in tokenized_batch]
        # Prepare empty tensors and variables   
        word_indices = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        flat_ngrams = []
        offsets = []
        curr_offset = 0
        # 2. Sparse vectors
        for i, seq in enumerate(padded_batch):
            for j, tok in enumerate(seq):
                widx = self.sparse_extractor.token_to_word_index(tok)
                word_indices[i, j] = widx
                # Create padding mask and ngram indices
                ngrams = [0]
                if tok == self.pad_token:
                    padding_mask[i, j] = True
                    ngrams = [0]   # dummy for bag
                else:
                    ngrams = self.sparse_extractor.token_to_ngram_indices(tok)
                
                # EmbeddingBag offsets
                offsets.append(curr_offset)
                flat_ngrams.extend(ngrams)
                curr_offset += len(ngrams)
        # Convert to tensors
        flat_ngrams = torch.tensor(flat_ngrams, dtype=torch.long, device=self.device)
        offsets = torch.tensor(offsets, dtype=torch.long, device=self.device)
        
        return word_indices, flat_ngrams, offsets, padding_mask

    def forward(self, input_texts: list[str]):
        """
        Forward pass of the DIET model.
        """
        # 1. Compute sparse features
        (word_indices, flat_ngrams, 
         offsets, padding_mask) = self.compute_sparse_features(input_texts)
        batch_size, max_len = word_indices.shape
        # 3. Forward pass
        # Compute sparse feature from embeddings
        word_emb = self.word_emb(word_indices)                # [B, S, embed_dims]
        ngram_emb = self.ngram_emb_bag(flat_ngrams, offsets)  # [B*S, embed_dims]
        ngram_emb = ngram_emb.view(batch_size, max_len, -1)   # [B, S, embed_dims]
        token_repr = word_emb + ngram_emb                     # [B, S, embed_dims]
        # Feedforward + Transformer
        x = self.activation(self.fc2(self.dropout(token_repr))) # [B, S, tf_dims]
        x = self.norm(x)
        x = x.transpose(0,1) # Transformer expects [S, B, tf_dims]
        x = self.transformer(x, 
                             src_key_padding_mask=padding_mask,
                             is_causal=False) # [S, B, tf_dims]
        x = x.transpose(0,1) # back to [B, S, tf_dims]
        # Infer entities
        emissions = self.crf_ff(x)  # [B, S, num_tags]
        x_entity = self.crf(emissions, ~padding_mask)  # Viterbi decode [B, S]
        # Infer intents
        intent_logits = self.intent_ff(x[:,0,:])  # [B, num_intent_tags]
        # Compute softmax over intent logits
        intent_probs = torch.softmax(intent_logits, dim=-1) # [B, num_intent_tags]
        
        return x_entity, intent_probs
    
    def train_forward(self, input_texts: list[str],
                      entity_labels: torch.Tensor, 
                      intent_labels: torch.Tensor): 
        """
        Forward pass for training (returns emissions for CRF).
        """
        # 1. Compute sparse features
        (word_indices, flat_ngrams, 
         offsets, padding_mask) = self.compute_sparse_features(input_texts)
        batch_size, max_len = word_indices.shape
        # 3. Forward pass
        # Compute sparse feature from embeddings
        word_emb = self.word_emb(word_indices)                # [B, S, embed_dims]
        ngram_emb = self.ngram_emb_bag(flat_ngrams, offsets)  # [B*S, embed_dims]
        print(word_emb.shape)
        print(ngram_emb.shape)
        ngram_emb = ngram_emb.view(batch_size, max_len, -1)   # [B, S, embed_dims]
        print(ngram_emb.shape)
        token_repr = word_emb + ngram_emb                     # [B, S, embed_dims]
        # Feedforward + Transformer
        x = self.activation(self.fc2(self.dropout(token_repr))) # [B, S, tf_dims]
        x = self.norm(x)
        x = x.transpose(0,1) # Transformer expects [S, B, tf_dims]
        x = self.transformer(x, src_key_padding_mask=padding_mask) # [S, B, tf_dims]
        x = x.transpose(0,1) # back to [B, S, tf_dims]
        # Emissions for CRF
        emissions = self.crf_ff(x)  # [B, S, num_tags]
        entity_loss = self.crf.compute_loss(emissions, entity_labels, ~padding_mask)  # CRF loss
        # Intent logits
        intent_logits = self.intent_ff(x[:,0,:])  # [B, num_intent_tags]
        intent_loss = self.intent_criterion(intent_logits, intent_labels)  # Intent loss
        total_loss = entity_loss + intent_loss
        
        return total_loss