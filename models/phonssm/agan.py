"""
Anatomical Graph Attention Network (AGAN)
=========================================
Graph Attention Network that respects hand/body skeleton topology.
Supports multiple input modes: single_hand, both_hands, pose_hands, full.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


def create_adjacency(mode: Literal["single_hand", "both_hands", "pose_hands", "full"] = "single_hand") -> torch.Tensor:
    """Create adjacency matrix based on input mode."""
    if mode == "single_hand":
        return create_hand_adjacency()
    elif mode == "both_hands":
        return create_both_hands_adjacency()
    elif mode == "pose_hands":
        return create_pose_hands_adjacency()
    elif mode == "full":
        return create_full_adjacency()
    else:
        raise ValueError(f"Unknown mode: {mode}")


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


def create_both_hands_adjacency() -> torch.Tensor:
    """
    Create adjacency for both hands (42 landmarks).
    Left hand: 0-20, Right hand: 21-41
    """
    A = torch.zeros(42, 42)

    # Left hand (indices 0-20)
    left_hand = create_hand_adjacency()
    A[:21, :21] = left_hand

    # Right hand (indices 21-41)
    A[21:42, 21:42] = left_hand  # Same topology

    # Cross-hand connections (wrist to wrist, weak)
    A[0, 21] = 0.3
    A[21, 0] = 0.3

    return A


def create_pose_hands_adjacency() -> torch.Tensor:
    """
    Create adjacency for pose + both hands (75 landmarks).

    MediaPipe Pose landmarks (33 total):
        0: nose, 1-4: left/right eye, 5-6: left/right ear
        7-8: mouth, 9-10: left/right shoulder
        11-12: left/right elbow, 13-14: left/right wrist
        15-22: hand landmarks (simplified), 23-28: hip/knee/ankle
        29-32: foot landmarks

    Layout:
        0-32: Pose (33 landmarks)
        33-53: Left hand (21 landmarks)
        54-74: Right hand (21 landmarks)
    """
    A = torch.zeros(75, 75)

    # === Pose skeleton (0-32) ===
    # Face
    face_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Nose to eyes to ears
        (0, 5), (0, 6),  # Nose to mouth
    ]

    # Upper body
    body_connections = [
        (9, 10),  # Shoulders
        (9, 11), (11, 13),  # Left arm: shoulder -> elbow -> wrist
        (10, 12), (12, 14),  # Right arm: shoulder -> elbow -> wrist
        (9, 23), (10, 24),  # Shoulders to hips
        (23, 24),  # Hips
    ]

    # Lower body (optional, less important for signs)
    lower_body = [
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
    ]

    for i, j in face_connections + body_connections + lower_body:
        if i < 33 and j < 33:
            A[i, j] = 1
            A[j, i] = 1

    # === Left hand (33-53) ===
    left_hand = create_hand_adjacency()
    A[33:54, 33:54] = left_hand

    # === Right hand (54-74) ===
    A[54:75, 54:75] = left_hand

    # === Connect hands to pose wrists ===
    # Pose left wrist (13) to left hand wrist (33)
    A[13, 33] = 1
    A[33, 13] = 1

    # Pose right wrist (14) to right hand wrist (54)
    A[14, 54] = 1
    A[54, 14] = 1

    # Cross-hand connection (weak)
    A[33, 54] = 0.3
    A[54, 33] = 0.3

    # Self-loops
    A = A + torch.eye(75)

    return A


def create_full_adjacency() -> torch.Tensor:
    """
    Create adjacency for pose + hands + face key points (130 landmarks).

    Layout:
        0-32: Pose (33 landmarks)
        33-53: Left hand (21 landmarks)
        54-74: Right hand (21 landmarks)
        75-129: Face key points (55 landmarks - subset of MediaPipe face)
    """
    A = torch.zeros(130, 130)

    # Start with pose_hands adjacency
    pose_hands = create_pose_hands_adjacency()
    A[:75, :75] = pose_hands

    # Face mesh key points (simplified connections)
    # Key facial landmarks for expression recognition
    # Eyebrows, eyes, nose, mouth outline
    face_start = 75

    # Connect face points in a mesh-like pattern (simplified)
    # Upper face (eyebrows + eyes): 0-19
    for i in range(19):
        A[face_start + i, face_start + i + 1] = 0.5
        A[face_start + i + 1, face_start + i] = 0.5

    # Nose: 20-29
    for i in range(20, 29):
        A[face_start + i, face_start + i + 1] = 0.5
        A[face_start + i + 1, face_start + i] = 0.5

    # Mouth: 30-54
    for i in range(30, 54):
        A[face_start + i, face_start + i + 1] = 0.5
        A[face_start + i + 1, face_start + i] = 0.5
    # Close mouth loop
    A[face_start + 30, face_start + 54] = 0.5
    A[face_start + 54, face_start + 30] = 0.5

    # Connect face to pose nose
    A[0, face_start + 25] = 0.5  # Pose nose to face nose center
    A[face_start + 25, 0] = 0.5

    # Self-loops
    A = A + torch.eye(130)

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
    Graph Attention Network respecting hand/body skeleton topology.

    Key innovations:
    1. Anatomical prior in adjacency matrix
    2. Learnable edge weights for adaptive connections
    3. Multi-head attention for different relationship types

    Supports multiple input modes:
    - single_hand: 21 landmarks (original, for webcam)
    - both_hands: 42 landmarks
    - pose_hands: 75 landmarks (pose + both hands)
    - full: 130 landmarks (pose + hands + face)
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_heads: int = 4,
        num_nodes: int = 21,
        dropout: float = 0.1,
        input_mode: str = "single_hand"
    ):
        super().__init__()
        self.input_mode = input_mode
        self.num_nodes = num_nodes

        # Fixed anatomical adjacency based on input mode
        self.register_buffer('A_anat', create_adjacency(input_mode))

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
