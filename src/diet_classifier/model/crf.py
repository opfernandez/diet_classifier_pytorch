import torch
import torch.nn as nn

class CRF(nn.Module):
    def __init__(self, num_tags: int, 
                 pad_idx: int = None,
                 eos_idx: int = None,
                 bos_idx: int = None):
        super().__init__()

        self.num_tags = num_tags
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx

        if pad_idx is None or eos_idx is None or bos_idx is None:
            raise ValueError("pad_idx, eos_idx, and bos_idx must be provided for CRF.")

        # Transition matrix [from state i, to state i+1]
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Penalize transitions to/from BOS and EOS
        self.transitions.data[:, bos_idx] = -1e4
        self.transitions.data[eos_idx, :] = -1e4
        # Penalize transitions to/from padding tag
        self.transitions.data[:, pad_idx] = -1e4
        self.transitions.data[pad_idx, :] = -1e4
        # Except for transitions from PAD to PAD or PAD to EOS
        self.transitions.data[pad_idx, pad_idx] = 0.0
        self.transitions.data[pad_idx, eos_idx] = 0.0

    def compute_loss(self, emissions, tags, mask):
        """
        Negative log likelihood (for training)
        emissions: [B, S, num_tags]
        tags: [B, S]
        mask: [B, S] (1=keep, 0=ignore)
        """
        # Compute log not normalized probability of all paths and labeled path
        # _score_labeled_path is the sum of emissions and transitions for the given tags
        # _logadd_all_paths is the log-sum-exp over all possible tag sequences
        # During training we want to maximize the log prob of the correct path relative to all paths
        # So we minimize the negative log likelihood
        # from paper: https://arxiv.org/pdf/1603.01360.pdf
        log_Z = self._logadd_all_paths(emissions, mask)
        score = self._score_labeled_path(emissions, tags, mask)
        # The mean negative log likelihood over the batch is returned
        # to make the loss independent of batch size
        # Invert order of terms as we want to maximize the log-probability of
        # the correct tag sequence by minimizing the negative log-probability
        return torch.mean(log_Z - score)

    def forward(self, emissions, mask):
        """
        Viterbi
        emissions: [B, S, num_tags]
        mask: [B, S]
        """
        return self._viterbi_decode(emissions, mask)
    
    # -------------------- INTERNALS -------------------- #

    def _score_labeled_path(self, emissions, tags, mask):
        # emmissions: [B, S, C] -> Network outputs (logits) for each token
        # tags: [B, S] -> Ground truth tags
        # mask: [B, S] -> Mask for valid positions (1=valid, 0=pad)
        # transitions: [C, C] -> Transition scores between tags
        B, S, _ = emissions.shape
        score = torch.zeros(B, device=emissions.device)
        first_tags = tags[:, 0]
        # First emission
        emit_score = emissions[:, 0, :].gather(1, first_tags.unsqueeze(1)).squeeze(1)  # (B,)
        # Transition from BOS to first tag
        first_trans_score = self.transitions[self.bos_idx, first_tags]  # (B,)
        # Accumulate
        score += emit_score + first_trans_score

        for i in range(1, S):
            curr_tag = tags[:, i] # (B,)
            prev_tag = tags[:, i - 1] # (B,)
            # emissions[:, i, :]: (B, C) -> gather the emission score for the current tag (B,)
            emit_score = emissions[:, i, :].gather(1, curr_tag.unsqueeze(1)).squeeze(1) # (B,)
            # transitions[curr_tag, next_tag]: (B,) -> gather the transition score from curr_tag to next_tag
            trans_score = self.transitions[prev_tag, curr_tag] # (B,)
            # Apply mask and accumulate
            score += (emit_score + trans_score) * mask[:, i] # (B,)
        # Last emission
        last_valid_idx = mask.sum(dim=1) - 1 # (B,)
        last_tag = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze(1) # (B,)
        last_trans_score = self.transitions[last_tag, self.eos_idx]  # (B,)
        # Apply transition to EOS
        score += last_trans_score
        return score

    def _logadd_all_paths(self, emissions, mask):
        B, S, C = emissions.shape

        emit_score = emissions[:, 0, :]  # (B, C)
        first_trans_score = self.transitions[self.bos_idx, :].unsqueeze(0)  # (1, C) 

        alpha = emit_score + first_trans_score  # (B, C)

        for i in range(1, S):
            emit = emissions[:, i, :].unsqueeze(1)       # (B,1,C)
            trans = self.transitions.unsqueeze(0)        # (1,C,C)
            scores = alpha.unsqueeze(2) + emit + trans   # (B,C,C)
                                     # alpha.unsqueeze(2): (B,C,1)
            # scores is a matrix of size (B,C,C) where for each batch we have 
            # the scores of transitioning from each tag (dim 1) to each tag (dim 2)
            # at position i. 
            # We need to sum over the previous tags (dim 1) 
            # to get the total score for each current tag (dim 2)
            # So the result is (B,C) and for each batch we have the total score/logit
            # of being in each tag at position i no matter the previous tag comming from.
            # That is what we got at torch.logsumexp(scores, dim=1)
            # If mask[:, i] is 1, we take the new scores, else we keep the old alpha
            # mask[:, i] : (B,) -> unsqueeze(1) -> (B,1) to broadcast, that is to say,
            # over a batch the value of the mask is multiplied to each class score
            mask_i = mask[:, i].unsqueeze(1)  # (B, 1)
            alpha = torch.logsumexp(scores, dim=1) * mask_i \
                    + alpha * (1 - mask_i) # (B,C)
        # Transition to EOS
        last_trans_score = self.transitions[:, self.eos_idx].unsqueeze(0)  # (1, C)
        alpha = alpha + last_trans_score  # (B,C)
        # (B,) aggregate of all class scores over the same batch
        return torch.logsumexp(alpha, dim=1) 

    def _viterbi_decode(self, emissions, mask):
        B, S, C = emissions.shape

        backpointers = []
        emit_score = emissions[:, 0, :]  # (B, C)
        first_trans_score = self.transitions[self.bos_idx, :].unsqueeze(0)  # (1, C) 
        last_valid_idx = mask.sum(dim=1) # (B,)

        alpha = emit_score + first_trans_score  # (B, C)
        # (B = batch size, C = From tags, C = To tags)
        for i in range(1, S):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) # (B,C,C)
            best_scores, best_tags = scores.max(1) # (B,C)

            # Apply mask: keep alpha unchanged for padded positions
            new_alpha = best_scores + emissions[:, i, :]  # (B,C)
            mask_i = mask[:, i].unsqueeze(1)  # (B, 1)
            alpha = new_alpha * mask_i + alpha * (1 - mask_i)  # (B,C)
            
            backpointers.append(best_tags)  # list of (B,C) -> (S-1, B, C)
        
        # Transition to EOS
        last_trans_score = self.transitions[:, self.eos_idx].unsqueeze(0)  # (1, C)
        alpha = alpha + last_trans_score  # (B,C)

        # Backtrack
        best_last_tags = alpha.argmax(1) # (B,)
        best_paths = [] 

        for b in range(B):
            # Length of the sequence for batch b considering only valid tokens
            seq_len = int(last_valid_idx[b].item())
            best_tag = best_last_tags[b].item() # Get the bestlast tag
            
            # Initialize path with the best last tag
            # and backtrack through backpointers
            path = [best_tag]
            
            # Backpropagate only through valid positions (seq_len - 1 backpointers)
            for t in range(seq_len - 2, -1, -1):
                # gets the index from last steps that led to best current tag
                # so it gathers the path from the best last tag to the first
                best_tag = backpointers[t][b, best_tag].item()
                path.insert(0, best_tag) # List of length seq_len
            
            best_paths.append(path) # List of B paths each one of their own length
        
        # Convert best paths to tensor with padding
        max_path_len = max(len(p) for p in best_paths)
        padded_paths = torch.full((B, max_path_len), self.pad_idx, 
                                   dtype=torch.long, device=emissions.device)
        
        for b, path in enumerate(best_paths):
            padded_paths[b, :len(path)] = torch.tensor(path, dtype=torch.long)
        
        return padded_paths  # (B, max_seq_len)
