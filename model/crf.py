import torch
import torch.nn as nn

class CRF(nn.Module):
    def __init__(self, num_tags: int, pad_idx: int = None):
        super().__init__()

        self.num_tags = num_tags
        self.pad_idx = pad_idx

        # Transition matrix [from state i, to state i+1]
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Penalize transitions to/from padding tag
        if pad_idx is not None:
            self.transitions.data[:, pad_idx] = -10000
            self.transitions.data[pad_idx, :] = -10000

    def compute_loss(self, emissions, tags, mask):
        """
        Negative log likelihood (for training)
        emissions: [B, S, num_tags]
        tags: [B, S]
        mask: [B, S] (1=keep, 0=ignore)
        """
        # Compute log not normalized probability of all paths and labeled path
        # _score_labeled_path is the sum of emissions and transitions for the given tags
        # _sum_logprob_all_paths is the log-sum-exp over all possible tag sequences
        # During training we want to maximize the log prob of the correct path relative to all paths
        # So we minimize the negative log likelihood
        # from paper: https://arxiv.org/pdf/1603.01360.pdf
        log_Z = self._sum_logprob_all_paths(emissions, mask)
        log_p = self._score_labeled_path(emissions, tags, mask)
        print("log_p:", log_p)
        print("log_Z:", log_Z)
        # The mean negative log likelihood over the batch is returned
        # to make the loss independent of batch size
        return torch.mean(log_p - log_Z)

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

        for i in range(S - 1):
            curr_tag = tags[:, i] # (B,)
            next_tag = tags[:, i + 1] # (B,)
            # emissions[:, i, :]: (B, C) -> gather the emission score for the current tag (B,)
            emit_score = emissions[:, i, :].gather(1, curr_tag.unsqueeze(1)).squeeze(1) # (B,)
            # transitions[curr_tag, next_tag]: (B,) -> gather the transition score from curr_tag to next_tag
            trans_score = self.transitions[curr_tag, next_tag] # (B,)
            # Apply mask and accumulate
            score += (emit_score + trans_score) * mask[:, i] # (B,)
        # Last emission
        last_tag = tags[:, -1]
        last_emit = emissions[:, -1, :].gather(1, last_tag.unsqueeze(1)).squeeze(1)
        score += last_emit * mask[:, -1]

        return score

    def _sum_logprob_all_paths(self, emissions, mask):
        B, S, C = emissions.shape

        alpha = emissions[:, 0, :]  # (B, C)

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
        # (B,) aggregate of all class scores over the same batch
        return torch.logsumexp(alpha, dim=1) 

    def _viterbi_decode(self, emissions, mask):
        B, S, C = emissions.shape

        backpointers = []
        alpha = emissions[:, 0, :]  # (B, C)
        # (B = batch size, C = From tags, C = To tags)
        for i in range(1, S):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) # (B,C,C)
            best_scores, best_tags = scores.max(1) # (B,C)

            # Apply mask: keep alpha unchanged for padded positions
            new_alpha = best_scores + emissions[:, i, :]  # (B,C)
            mask_i = mask[:, i].unsqueeze(1)  # (B, 1)
            alpha = new_alpha * mask_i + alpha * (1 - mask_i)  # (B,C)
            
            backpointers.append(best_tags)  # list of (B,C) -> (S-1, B, C)

        # Backtrack
        best_last_tags = alpha.argmax(1) # (B,)
        best_paths = [best_last_tags] 

        for backptrs in reversed(backpointers):
            # gets the index from last steps that led to best current tag
            # so it gathers the path from the best last tag to the first
            best_last_tags = backptrs.gather(1, best_last_tags.unsqueeze(1)).squeeze(1) # (B,)
            best_paths.insert(0, best_last_tags) # list of (B,) of length S

        return torch.stack(best_paths, dim=1) # (B, S)
