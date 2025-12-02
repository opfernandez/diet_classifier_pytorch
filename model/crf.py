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

    def forward(self, emissions, tags, mask):
        """
        Negative log likelihood negativa (for training)
        emissions: [B, S, num_tags]
        tags: [B, S]
        mask: [B, S] (1=keep, 0=ignore)
        """

        log_Z = self._hmm_log_forward(emissions, mask)
        log_p = self._compute_path_score(emissions, tags, mask)

        return torch.mean(log_Z - log_p)

    def decode(self, emissions, mask):
        """
        Viterbi
        emissions: [B, S, num_tags]
        mask: [B, S]
        """
        return self._viterbi_decode(emissions, mask)

    # -------------------- INTERNALS -------------------- #

    def _compute_path_score(self, emissions, tags, mask):
        B, S, _ = emissions.shape

        score = torch.zeros(B, device=emissions.device)

        for i in range(S - 1):
            curr_tag = tags[:, i]
            next_tag = tags[:, i + 1]

            emit_score = emissions[:, i, :].gather(1, curr_tag.unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[curr_tag, next_tag]

            score += (emit_score + trans_score) * mask[:, i]

        # Last emission
        last_tag = tags[:, -1]
        last_emit = emissions[:, -1, :].gather(1, last_tag.unsqueeze(1)).squeeze(1)
        score += last_emit * mask[:, -1]

        return score

    def _hmm_log_forward(self, emissions, mask):
        B, S, C = emissions.shape

        alpha = emissions[:, 0, :]  # (B, C)

        for i in range(1, S):
            emit = emissions[:, i, :].unsqueeze(1)       # (B,1,C)
            trans = self.transitions.unsqueeze(0)        # (1,C,C)
            scores = alpha.unsqueeze(2) + emit + trans   # (B,C,C)

            alpha = torch.logsumexp(scores, dim=1) * mask[:, i].unsqueeze(1) \
                    + alpha * (1 - mask[:, i]).unsqueeze(1) # (B,C)

        return torch.logsumexp(alpha, dim=1)

    def _viterbi_decode(self, emissions, mask):
        B, S, C = emissions.shape

        backpointers = []

        alpha = emissions[:, 0]

        for i in range(1, S):

            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_tags = scores.max(1)

            alpha = best_scores + emissions[:, i]
            backpointers.append(best_tags)

        # Backtrack
        best_last_tags = alpha.argmax(1)

        best_paths = [best_last_tags]

        for backptrs in reversed(backpointers):
            best_last_tags = backptrs.gather(1, best_last_tags.unsqueeze(1)).squeeze(1)
            best_paths.insert(0, best_last_tags)

        return torch.stack(best_paths, dim=1)
