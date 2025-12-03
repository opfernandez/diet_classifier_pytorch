import torch
from torch import nn
from DIET.model.sparse_features_extractor import SparseFeatureExtractor
from DIET.model.crf import CRF

class DIETModel(nn.Module):
    def __init__(self, word_dict_size: int = 300,
                 ngram_dict_size: int = 1000, 
                 ngram_min: int = 2, 
                 ngram_max: int = 4, 
                 ff_dim: int = 512,
                 tf_layers: int = 2,
                 tf_dims: int = 256,
                 tf_n_heads: int = 8,
                 device: str = 'cuda',
                 pad_token: str = "[PAD]",
                 cls_token: str = "[CLS]",
                 sep_token: str = "[SEP]",
                 unk_token: str = "[UNK]",
                 num_entity_tags: int = 10):
        super(DIETModel, self).__init__()

        # Sparse extractor
        self.sparse_extractor = SparseFeatureExtractor(
            word_dict_size=word_dict_size,
            ngram_dict_size=ngram_dict_size,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            unk_token=unk_token
        )

        # Instead of big sparse vectors + FFL we use embeddings
        self.pad_idx = self.sparse_extractor.token_to_word_index(pad_token)
        self.word_emb = nn.Embedding(word_dict_size, ff_dim, padding_idx=self.pad_idx)
        self.ngram_emb_bag =  nn.EmbeddingBag(ngram_dict_size, ff_dim, mode="sum")
        # TODO: add dropout and do pruning 
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
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(tf_dims)
        self.dropout = nn.Dropout(0.8)
        # Conditional Random Field (CRF) for sequence entity labeling
        self.crf_ff = nn.Linear(tf_dims, num_entity_tags)
        self.crf = CRF(num_tags=num_entity_tags, pad_idx=self.pad_idx)
        # Select device
        self.device = device
        # Special tokens token
        self.pad_token = pad_token
        self.unkg_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token

    def forward(self, input_texts: list[str]):
        """
        Forward pass of the DIET model.
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
        # TODO: CLS token must be the aggregation of all sequence tokens
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

        # Compute sparse feature from embeddings
        word_emb = self.word_emb(word_indices)                # [B, S, ff]
        ngram_emb = self.ngram_emb_bag(flat_ngrams, offsets)  # [B*S, ff]
        ngram_emb = ngram_emb.view(batch_size, max_len, -1)   # [B, S, ff]
        token_repr = word_emb + ngram_emb                     # [B, S, ff]
        # Feedforward + Transformer
        x = self.activation(self.fc2(self.dropout(token_repr))) # [B, S, tf_dims]
        x = self.norm(x)
        x = x.transpose(0,1) # Transformer expects [S, B, tf_dims]
        x = self.transformer(x, src_key_padding_mask=padding_mask) # [S, B, tf_dims]
        x = x.transpose(0,1) # back to [B, S, D]
        x_entity = self.crf_ff(x)  # [B, S, num_tags]
        x_entity = self.crf(x_entity, ~padding_mask)  # Viterbi decode
        
        return x, x_entity