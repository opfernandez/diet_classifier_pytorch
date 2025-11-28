import torch
from torch import nn
from DIET.model.diet_arch_blocks import SparseFeatureExtractor

class DIETModel(nn.Module):
    def __init__(self, word_dict_size: int = 300,
                 ngram_dict_size: int = 1000, 
                 ngram_min: int = 2, 
                 ngram_max: int = 4, 
                 ff_dim: int = 512,
                 tf_layers: int = 2,
                 tf_dims: int = 256,
                 tf_n_heads: int = 8,
                 device: str = 'cuda'):
        super(DIETModel, self).__init__()

        # Sparse extractor
        self.sparse_extractor = SparseFeatureExtractor(
            word_dict_size=word_dict_size,
            ngram_dict_size=ngram_dict_size,
            ngram_min=ngram_min,
            ngram_max=ngram_max
        )

        # Fully connected layers for sparse -> embedding
        # TODO: add dropout and do pruning
        self.fc1 = nn.Linear(word_dict_size + ngram_dict_size, ff_dim) 
        self.fc2 = nn.Linear(ff_dim, tf_dims) 
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
        # Select device
        self.device = device

    def forward(self, input_texts: list[str]):
        """
        Forward pass of the DIET model.
        """
        # 1. Tokenization
        tokenized_batch = [self.sparse_extractor.tokenizer(text) for text in input_texts]
        # 2. Sparse vectors
        batch_size = len(tokenized_batch)
        seq_lens = [len(seq) for seq in tokenized_batch]
        max_len = max(seq_lens)
        
        sparse_vecs = torch.zeros(batch_size, max_len, 
                                  self.fc1.in_features, device=self.device)
        
        for i, tokens in enumerate(tokenized_batch):
            for j, tok in enumerate(tokens):
                word_idx = self.sparse_extractor.token_to_word_index(tok)
                ngram_idx = self.sparse_extractor.token_to_ngram_indices(tok)
                
                # Word one-hot
                sparse_vecs[i, j, word_idx] = 1.0
                # Ngram multi-hot
                for idx in ngram_idx:
                    sparse_vecs[i, j, self.sparse_extractor.word_dict_size + idx] = 1.0
        
        # 3. FC layers
        x = self.activation(self.fc1(sparse_vecs))
        x = self.fc2(x)  # [batch, seq_len, tf_dims]
        
        # 4. Transformer expects [seq_len, batch, embed]
        x = x.transpose(0, 1)  # [seq_len, batch, tf_dims]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [batch, seq_len, tf_dims]
        
        return x