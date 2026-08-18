"""
PhonoGraph: part-decoupled, dominance-normalized graph net with a phonological
prototype head. pose-only isolated sign recognition, trained from scratch on one
WLASL subset. not a DSTA-Net derivative.

why each piece, given ~1.5k training clips:

- motion-energy dominance normalization routes the higher-motion hand into the
  "dominant" slot, so a shared hand encoder sees both hands (2x effective hand
  data) and left- vs right-handed signers look the same.
- three feature-isolated part encoders (body, dominant hand, non-dominant hand),
  each CTR-GCN topology-refined conv + multi-scale TCN. a failed hand detection
  cannot poison the body pathway. Siformer's idea, in a GCN instead of a
  data-hungry transformer.
- one multi-head attention over the 3 part tokens for two-hand interaction and
  hand-relative-to-body location.
- ProtoGCN-style prototype head: learnable bank of sub-sign primitives, fused
  feature refined by prototype attention, orthogonality penalty to keep the bank
  diverse. compositional phonemes done the way that actually works.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import part_nodes, part_adjacency
from .modules import PartEncoder


class PhonoGraph(nn.Module):
    def __init__(self, num_class, in_channels=3, base=64, num_proto=64, drop=0.2):
        super().__init__()
        nodes = part_nodes()
        adj = part_adjacency()
        self.register_buffer('body_idx', torch.tensor(nodes['body'], dtype=torch.long))
        self.register_buffer('handA_idx', torch.tensor(nodes['handA'], dtype=torch.long))
        self.register_buffer('handB_idx', torch.tensor(nodes['handB'], dtype=torch.long))

        self.body_enc = PartEncoder(in_channels, adj['body'], base, drop=drop)
        # one shared hand encoder for both hands, they share topology
        self.hand_enc = PartEncoder(in_channels, adj['handA'], base, drop=drop)
        C = self.body_enc.out_channels

        # role tokens: 0=body, 1=dominant hand, 2=non-dominant hand
        self.role_emb = nn.Parameter(torch.zeros(3, C))
        nn.init.trunc_normal_(self.role_emb, std=0.02)

        self.cross = nn.MultiheadAttention(C, num_heads=4, dropout=drop, batch_first=True)
        self.cross_norm = nn.LayerNorm(C)

        # phonological prototype memory
        self.num_proto = num_proto
        self.proto = nn.Parameter(torch.randn(num_proto, C))
        nn.init.trunc_normal_(self.proto, std=0.02)
        self.proj = nn.Linear(C, C)
        self.scale = C ** 0.5

        self.head_drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * C, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        nn.init.constant_(self.fc.bias, 0)

    def _hand_energy(self, hand):
        # hand: (N, C, T, Vh); use xy channels and hand-only joints (skip wrist at local 0)
        xy = hand[:, :2, :, 1:]
        e = (xy[:, :, 1:, :] - xy[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
        return e  # (N,)

    def forward(self, x, return_aux=False):
        # x: (N, C, T, V, M); we use M=1
        if x.dim() == 5:
            x = x[..., 0]
        body = x[:, :, :, self.body_idx]      # (N,C,T,7)
        hA = x[:, :, :, self.handA_idx]       # (N,C,T,11)
        hB = x[:, :, :, self.handB_idx]       # (N,C,T,11)

        eA, eB = self._hand_energy(hA), self._hand_energy(hB)
        domA = (eA >= eB).view(-1, 1, 1, 1)
        dom = torch.where(domA, hA, hB)
        nondom = torch.where(domA, hB, hA)

        be, _ = self.body_enc(body)
        de, _ = self.hand_enc(dom)
        ne, _ = self.hand_enc(nondom)

        tokens = torch.stack([be + self.role_emb[0],
                              de + self.role_emb[1],
                              ne + self.role_emb[2]], dim=1)   # (N,3,C)
        attn, _ = self.cross(tokens, tokens, tokens)
        tokens = self.cross_norm(tokens + attn)
        f = tokens.mean(1)                                      # (N,C)

        # prototype refinement
        q = self.proj(f)
        a = torch.softmax(q @ self.proto.t() / self.scale, dim=1)   # (N,K)
        r = a @ self.proto                                          # (N,C)

        feat = torch.cat([f, r], dim=1)
        logits = self.fc(self.head_drop(feat))

        if return_aux:
            Pn = F.normalize(self.proto, dim=1)
            G = Pn @ Pn.t()
            ortho = (G - torch.eye(self.num_proto, device=G.device)).pow(2).mean()
            return logits, f, ortho
        return logits
