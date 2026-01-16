"""
Anatomical Graph Attention Network (AGAN)
=========================================
Graph Attention Network that respects hand skeleton topology.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_hand_adjacency() -> torch.Tensor:
    """
    Create anatomical adjacency matrix for hand skeleton.

    MediaPipe hand landmarks:
        0: Wrist
        1-4: Thumb (CMC, MCP, IP, TIP)
        5-8: Index finger (MCP, PIP, DIP, TIP)
        9-12: Middle finger
        13-16: Ring finger
        17-20: Pinky finger
    """
    A = torch.zeros(21, 21)

    # Finger chains
    fingers = [
        [0, 1, 2, 3, 4],       # Thumb
        [0, 5, 6, 7, 8],       # Index
        [0, 9, 10, 11, 12],    # Middle
        [0, 13, 14, 15, 16],   # Ring
        [0, 17, 18, 19, 20],   # Pinky
    ]

    for finger in fingers:
        for i in range(len(finger) - 1):
            A[finger[i], finger[i + 1]] = 1
            A[finger[i + 1], finger[i]] = 1  # Symmetric

    # Cross-finger connections (MCP joints)
    mcp_joints = [5, 9, 13, 17]
    for i in range(len(mcp_joints) - 1):
        A[mcp_joints[i], mcp_joints[i + 1]] = 1
        A[mcp_joints[i + 1], mcp_joints[i]] = 1

    # Fingertip connections (optional - helps with spread detection)
    fingertips = [4, 8, 12, 16, 20]
    for i in range(len(fingertips) - 1):
        A[fingertips[i], fingertips[i + 1]] = 0.5  # Weaker connection
        A[fingertips[i + 1], fingertips[i]] = 0.5

    # Self-loops
    A = A + torch.eye(21)

    return A


class GraphAttentionLayer(nn.Module):
    """Single graph attention layer with multi-head attention."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat

        # Linear transformations for each head
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)

        # Attention parameters
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features))

        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, F_in) - node features
            adj: (N, N) - adjacency matrix
        Returns:
            (B, N, F_out * num_heads) if concat else (B, N, F_out)
        """
        B, N, _ = x.shape

        # Linear transformation
        h = self.W(x)  # (B, N, num_heads * out_features)
        h = h.view(B, N, self.num_heads, self.out_features)  # (B, N, H, F)

        # Compute attention scores using additive attention
        # e_ij = LeakyReLU(a_src @ h_i + a_dst @ h_j)
        attn_src = (h * self.a_src).sum(dim=-1)  # (B, N, H)
        attn_dst = (h * self.a_dst).sum(dim=-1)  # (B, N, H)

        # Broadcast to get pairwise scores
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)  # (B, N, N, H)
        attn = self.leaky_relu(attn)

        # Mask with adjacency (only attend to neighbors)
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        attn = attn.masked_fill(mask, float('-inf'))

        # Softmax over neighbors
        attn = F.softmax(attn, dim=2)  # (B, N, N, H)
        attn = self.dropout(attn)

        # Aggregate neighbor features
        h = h.permute(0, 2, 1, 3)  # (B, H, N, F)
        attn = attn.permute(0, 3, 1, 2)  # (B, H, N, N)
        out = torch.matmul(attn, h)  # (B, H, N, F)
        out = out.permute(0, 2, 1, 3)  # (B, N, H, F)

        if self.concat:
            return out.reshape(B, N, -1)  # (B, N, H*F)
        else:
            return out.mean(dim=2)  # (B, N, F)


class AnatomicalGraphAttention(nn.Module):
    """
    Graph Attention Network respecting hand skeleton topology.

    Key innovations:
    1. Anatomical prior in adjacency matrix
    2. Learnable edge weights for adaptive connections
    3. Multi-head attention for different relationship types
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_heads: int = 4,
        num_nodes: int = 21,
        dropout: float = 0.1
    ):
        super().__init__()

        # Fixed anatomical adjacency
        self.register_buffer('A_anat', create_hand_adjacency())

        # Learnable adjacency residual (discovers non-obvious connections)
        self.A_learn = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Graph attention layers
        self.gat1 = GraphAttentionLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=True
        )

        self.gat2 = GraphAttentionLayer(
            in_features=hidden_dim * num_heads,
            out_features=out_dim,
            num_heads=1,
            dropout=dropout,
            concat=False
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm2 = nn.LayerNorm(out_dim)

        # Node pooling to single vector
        self.node_pool = nn.Sequential(
            nn.Linear(num_nodes * out_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) - batch, time, nodes, coords
        Returns:
            (B, T, D) - spatial embeddings per frame
        """
        B, T, N, C = x.shape

        # Compute adaptive adjacency
        A = self.A_anat + torch.sigmoid(self.A_learn) * 0.5
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)  # Normalize

        # Reshape for batch processing
        x = x.view(B * T, N, C)
        x = self.input_proj(x)  # (B*T, N, hidden)

        # Graph attention layers with residual connections
        h = self.gat1(x, A)  # (B*T, N, hidden*heads)
        h = self.norm1(F.elu(h))

        h = self.gat2(h, A)  # (B*T, N, out_dim)
        h = self.norm2(F.elu(h))

        # Pool nodes to single vector per frame
        h = h.view(B * T, -1)  # (B*T, N*out_dim)
        h = self.node_pool(h)  # (B*T, out_dim)
        h = h.view(B, T, -1)   # (B, T, out_dim)

        return h
