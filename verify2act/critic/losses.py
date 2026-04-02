import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """NT-Xent (InfoNCE) loss over explicit triplets.

    Inputs are expected to be L2-normalized embeddings from model.project().
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
        use_inbatch_negatives: bool = True,
    ) -> torch.Tensor:
        bsz = anchor_emb.size(0)
        tau = self.temperature

        pos_sim = (anchor_emb * positive_emb).sum(dim=-1) / tau
        neg_sim = (anchor_emb * negative_emb).sum(dim=-1) / tau

        if use_inbatch_negatives and bsz > 1:
            sim_matrix = torch.mm(anchor_emb, positive_emb.T) / tau
            mask = ~torch.eye(bsz, dtype=torch.bool, device=anchor_emb.device)
            inbatch = sim_matrix[mask].view(bsz, bsz - 1)
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1), inbatch], dim=1)
        else:
            logits = torch.stack([pos_sim, neg_sim], dim=1)

        targets = torch.zeros(bsz, dtype=torch.long, device=anchor_emb.device)
        return F.cross_entropy(logits, targets)
