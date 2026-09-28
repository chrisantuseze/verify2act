"""Spatial, text-conditioned goal-success head: P(goal text is satisfied | DINOv2 patch tokens).

The dual-head critic mean-pools the patch tokens and matches them to a pooled CLIP text vector. Both poolings lose the
spatial relations the eval tasks ask about: a probe on mean-pooled twin features cannot tell "A left of B" (0.50),
and pooled CLIP puts "red left of blue" and "blue left of red" at cosine 0.99. This head keeps the 16x16 patch grid and
the CLIP per-token text features and lets a small transformer relate them.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ClipTokenEncoder:
    """Frozen CLIP ViT-B/32 text transformer, returning per-token features [N, L, 512] and a padding mask.
    Results are cached by string (the goal vocabulary is small)."""

    def __init__(self, device: torch.device, max_len: int = 48):
        import clip as openai_clip
        self.device, self.max_len = device, max_len
        self.model, _ = openai_clip.load("ViT-B/32", device=device)
        self.model.eval()
        self.tokenize = openai_clip.tokenize
        self.cache: Dict[str, tuple] = {}

    @torch.no_grad()
    def _encode(self, texts: List[str]):
        tok = self.tokenize(texts, truncate=True).to(self.device)
        m = self.model
        x = m.token_embedding(tok).type(m.dtype) + m.positional_embedding.type(m.dtype)
        x = m.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = m.ln_final(x).float()[:, :self.max_len]
        eot = tok.argmax(dim=-1)
        pad = torch.arange(self.max_len, device=self.device)[None] > eot[:, None]
        return x, pad

    def __call__(self, texts: List[str]):
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(new), 256):
            x, pad = self._encode(new[i:i + 256])
            for j, t in enumerate(new[i:i + 256]):
                self.cache[t] = (x[j], pad[j])
        return (torch.stack([self.cache[t][0] for t in texts]), torch.stack([self.cache[t][1] for t in texts]))


class SpatialGoalHead(nn.Module):
    def __init__(self, dino_channels: int = 1024, text_dim: int = 512, d: int = 256, layers: int = 4,
                 heads: int = 8, num_patches: int = 256, text_len: int = 48, dropout: float = 0.1):
        super().__init__()
        self.config = dict(dino_channels=dino_channels, text_dim=text_dim, d=d, layers=layers, heads=heads,
                           num_patches=num_patches, text_len=text_len, dropout=dropout)
        self.img_in = nn.Sequential(nn.LayerNorm(dino_channels), nn.Linear(dino_channels, d))
        self.txt_in = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.img_pos = nn.Parameter(torch.zeros(1, num_patches, d))
        self.txt_pos = nn.Parameter(torch.zeros(1, text_len, d))
        self.type_emb = nn.Parameter(torch.zeros(3, d))          # cls / text / image
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.img_pos, std=0.02); nn.init.normal_(self.txt_pos, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02); nn.init.normal_(self.cls, std=0.02)
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True), layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, patches: torch.Tensor, txt: torch.Tensor, txt_pad: torch.Tensor) -> torch.Tensor:
        """patches [B, P, C], txt [B, L, 512], txt_pad [B, L] (True = padding) -> logits [B]."""
        B = patches.size(0)
        h_img = self.img_in(patches.float()) + self.img_pos[:, :patches.size(1)] + self.type_emb[2]
        h_txt = self.txt_in(txt.float()) + self.txt_pos[:, :txt.size(1)] + self.type_emb[1]
        h = torch.cat([self.cls.expand(B, -1, -1) + self.type_emb[0], h_txt, h_img], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=h.device), txt_pad,
                         torch.zeros(B, patches.size(1), dtype=torch.bool, device=h.device)], 1)
        return self.out(self.tf(h, src_key_padding_mask=pad)[:, 0]).squeeze(-1)


class GoalScorer:
    """Inference wrapper: score patch tokens against goal texts. Returns P(satisfied) in [0, 1]."""

    def __init__(self, ckpt_path: str, device: torch.device):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)      # our own file; metrics hold numpy scalars
        self.head = SpatialGoalHead(**ck["config"]).to(device).eval()
        self.head.load_state_dict(ck["state_dict"])
        self.text = ClipTokenEncoder(device, ck["config"]["text_len"])
        self.device = device

    @torch.no_grad()
    def __call__(self, patches: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """patches [B, P, C] (B == len(texts), or B == 1 broadcast) -> probabilities [len(texts)]."""
        txt, pad = self.text(texts)
        if patches.size(0) == 1 and len(texts) > 1:
            patches = patches.expand(len(texts), -1, -1)
        return torch.sigmoid(self.head(patches.to(self.device), txt, pad))
