"""AGAN: graph attention over the hand/body skeleton topology.

input modes: single_hand, both_hands, pose_hands, full, sign27.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


def create_adjacency(mode: str = "single_hand") -> torch.Tensor:
    """Create adjacency matrix based on input mode."""
    if mode == "single_hand":
        return create_hand_adjacency()
    elif mode == "both_hands":
        return create_both_hands_adjacency()
    elif mode == "pose_hands":
        return create_pose_hands_adjacency()
    elif mode == "full":
        return create_full_adjacency()
    elif mode == "sign27":
        return create_sign27_adjacency()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_sign27_adjacency() -> torch.Tensor:
    """SAM-SLR / DSTA-SLR 27-joint skeleton (7 upper-body + 10 left-hand +
    10 right-hand), the HRNet whole-body layout behind pose-only WLASL SOTA.
    Node order: 0 nose, 1/2 L/R shoulder, 3/4 L/R elbow, 5/6 L/R wrist,
    7-16 left hand, 17-26 right hand. Edge list from DSTA-SLR graph/sign_27.py."""
    edges = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6),
             (7, 8), (7, 9), (7, 11), (7, 13), (7, 15), (9, 10), (11, 12),
             (13, 14), (15, 16),
             (17, 18), (17, 19), (17, 21), (17, 23), (17, 25), (19, 20),
             (21, 22), (23, 24), (25, 26),
             (5, 7), (6, 17)]  # wrist -> hand root (left, right)
    A = torch.zeros(27, 27)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


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

    # finger chains
    fingers = [
        [0, 1, 2, 3, 4],       # thumb
        [0, 5, 6, 7, 8],       # index
        [0, 9, 10, 11, 12],    # middle
        [0, 13, 14, 15, 16],   # ring
        [0, 17, 18, 19, 20],   # pinky
    ]

    for finger in fingers:
        for i in range(len(finger) - 1):
            A[finger[i], finger[i + 1]] = 1
            A[finger[i + 1], finger[i]] = 1  # symmetric

    # cross-finger connections (MCP joints)
    mcp_joints = [5, 9, 13, 17]
    for i in range(len(mcp_joints) - 1):
        A[mcp_joints[i], mcp_joints[i + 1]] = 1
        A[mcp_joints[i + 1], mcp_joints[i]] = 1

    # fingertip connections (optional - helps with spread detection)
    fingertips = [4, 8, 12, 16, 20]
    for i in range(len(fingertips) - 1):
        A[fingertips[i], fingertips[i + 1]] = 0.5  # weaker connection
        A[fingertips[i + 1], fingertips[i]] = 0.5

    # self-loops
    A = A + torch.eye(21)

    return A


def create_both_hands_adjacency() -> torch.Tensor:
    """
    Create adjacency for both hands (42 landmarks).
    Left hand: 0-20, Right hand: 21-41
    """
    A = torch.zeros(42, 42)

    # left hand (indices 0-20)
    left_hand = create_hand_adjacency()
    A[:21, :21] = left_hand

    # right hand (indices 21-41)
    A[21:42, 21:42] = left_hand  # same topology

    # cross-hand connections (wrist to wrist, weak)
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

    # pose skeleton (0-32)
    face_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # nose to eyes to ears
        (0, 5), (0, 6),  # nose to mouth
    ]

    # upper body
    body_connections = [
        (9, 10),  # shoulders
        (9, 11), (11, 13),  # left arm: shoulder -> elbow -> wrist
        (10, 12), (12, 14),  # right arm: shoulder -> elbow -> wrist
        (9, 23), (10, 24),  # shoulders to hips
        (23, 24),  # hips
    ]

    # lower body (optional, less important for signs)
    lower_body = [
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]

    for i, j in face_connections + body_connections + lower_body:
        if i < 33 and j < 33:
            A[i, j] = 1
            A[j, i] = 1

    # left hand (33-53)
    left_hand = create_hand_adjacency()
    A[33:54, 33:54] = left_hand

    # right hand (54-74)
    A[54:75, 54:75] = left_hand

    # connect hands to pose wrists
    # pose left wrist (13) to left hand wrist (33)
    A[13, 33] = 1
    A[33, 13] = 1

    # pose right wrist (14) to right hand wrist (54)
    A[14, 54] = 1
    A[54, 14] = 1

    # cross-hand connection (weak)
    A[33, 54] = 0.3
    A[54, 33] = 0.3

    # self-loops
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

    # start with pose_hands adjacency
    pose_hands = create_pose_hands_adjacency()
    A[:75, :75] = pose_hands

    # face mesh key points (simplified connections)
    # key facial landmarks for expression recognition
    # eyebrows, eyes, nose, mouth outline
    face_start = 75

    # connect face points in a mesh-like pattern (simplified)
    # upper face (eyebrows + eyes): 0-19
    for i in range(19):
        A[face_start + i, face_start + i + 1] = 0.5
        A[face_start + i + 1, face_start + i] = 0.5

    # nose: 20-29
    for i in range(20, 29):
        A[face_start + i, face_start + i + 1] = 0.5
        A[face_start + i + 1, face_start + i] = 0.5

    # mouth: 30-54
    for i in range(30, 54):
        A[face_start + i, face_start + i + 1] = 0.5
        A[face_start + i + 1, face_start + i] = 0.5
    # close mouth loop
    A[face_start + 30, face_start + 54] = 0.5
    A[face_start + 54, face_start + 30] = 0.5

    # connect face to pose nose
    A[0, face_start + 25] = 0.5  # pose nose to face nose center
    A[face_start + 25, 0] = 0.5

    # self-loops
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

        # linear transformations for each head
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)

        # attention parameters
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

        # linear transformation
        h = self.W(x)  # (B, N, num_heads * out_features)
        h = h.view(B, N, self.num_heads, self.out_features)  # (B, N, H, F)

        # compute attention scores using additive attention
        # e_ij = LeakyReLU(a_src @ h_i + a_dst @ h_j)
        attn_src = (h * self.a_src).sum(dim=-1)  # (B, N, H)
        attn_dst = (h * self.a_dst).sum(dim=-1)  # (B, N, H)

        # broadcast to get pairwise scores
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)  # (B, N, N, H)
        attn = self.leaky_relu(attn)

        # mask with adjacency (only attend to neighbors)
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        attn = attn.masked_fill(mask, float('-inf'))

        # Softmax over neighbors
        attn = F.softmax(attn, dim=2)  # (B, N, N, H)
        attn = self.dropout(attn)

        # aggregate neighbor features
        h = h.permute(0, 2, 1, 3)  # (B, H, N, F)
        attn = attn.permute(0, 3, 1, 2)  # (B, H, N, N)
        out = torch.matmul(attn, h)  # (B, H, N, F)
        out = out.permute(0, 2, 1, 3)  # (B, N, H, F)

        if self.concat:
            return out.reshape(B, N, -1)  # (B, N, H*F)
        else:
            return out.mean(dim=2)  # (B, N, F)


class AnatomicalGraphAttention(nn.Module):
    """graph attention over hand/body skeleton topology.

    anatomical prior in the adjacency, learnable edge weights on top, multi-head
    attention for different relationship types.

    input modes:
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

        # fixed anatomical adjacency based on input mode
        self.register_buffer('A_anat', create_adjacency(input_mode))

        # learnable adjacency residual (discovers non-obvious connections)
        self.A_learn = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        # input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # graph attention layers
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

        # layer norms
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm2 = nn.LayerNorm(out_dim)

        # attention pooling over nodes (replaces the old flatten-Linear pool,
        # which was Linear(num_nodes*out_dim, ...) — 2.46M params / 57% of the
        # whole model — and destroyed permutation structure). A learned query
        # attends over the N node embeddings; O(out_dim) params, keeps spatial
        # invariance, and frees capacity for a wider encoder.
        self.pool_query = nn.Parameter(torch.randn(out_dim) * 0.02)
        self.pool_out = nn.Sequential(nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) - batch, time, nodes, coords
        Returns:
            (B, T, D) - spatial embeddings per frame
        """
        B, T, N, C = x.shape

        # the learnable residual is masked to real skeletal edges. without the
        # mask sigmoid()*0.5 is positive everywhere, A never hits 0, and
        # GATLayer's (adj==0) test never fires — that was dense attention, not
        # the anatomical masking the paper claims. published checkpoints predate
        # this fix. self-loop so every node attends to itself.
        edge = (self.A_anat > 0).float()
        eye = torch.eye(self.num_nodes, device=self.A_anat.device)
        A = self.A_anat + torch.sigmoid(self.A_learn) * 0.5 * edge + eye
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)  # normalize

        # reshape for batch processing
        x = x.view(B * T, N, C)
        x = self.input_proj(x)  # (B*T, N, hidden)

        # graph attention layers with residual connections
        h = self.gat1(x, A)  # (B*T, N, hidden*heads)
        h = self.norm1(F.elu(h))

        h = self.gat2(h, A)  # (B*T, N, out_dim)
        h = self.norm2(F.elu(h))

        # attention-pool nodes to one vector per frame
        scores = (h @ self.pool_query) / (h.shape[-1] ** 0.5)  # (B*T, N)
        w = F.softmax(scores, dim=1).unsqueeze(-1)              # (B*T, N, 1)
        h = (h * w).sum(dim=1)                                  # (B*T, out_dim)
        h = self.pool_out(h)
        h = h.view(B, T, -1)                                   # (B, T, out_dim)
        return h
