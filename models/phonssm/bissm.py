"""
Bidirectional Selective State Space (BiSSM)
============================================
Mamba-inspired temporal modeling with O(n) complexity.
Uses selective state space for efficient long-range dependencies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - core of Mamba architecture.

    Key innovations:
    1. Input-dependent state transitions (selective)
    2. Hardware-aware parallel scan
    3. O(n) complexity vs O(n²) for attention
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Input projection (expand)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Conv for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner  # Depthwise
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Discretization parameters
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # State space matrices
        # A is structured as diagonal for efficiency
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - input sequence
        Returns:
            (B, T, D) - output sequence
        """
        B, T, D = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # Each (B, T, d_inner)

        # Conv for local context
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # Causal padding
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameters from input (selective)
        ssm_params = self.x_proj(x_conv)  # (B, T, d_state*2 + 1)
        B_param = ssm_params[:, :, :self.d_state]  # (B, T, d_state)
        C_param = ssm_params[:, :, self.d_state:2*self.d_state]  # (B, T, d_state)
        dt = F.softplus(ssm_params[:, :, -1:])  # (B, T, 1) - timestep

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_state,) - negative for stability
        dt_proj = self.dt_proj(dt)  # (B, T, d_inner)

        # Parallel scan (simplified sequential for clarity)
        # In production, use associative scan for parallelism
        y = self._ssm_scan(x_conv, A, B_param, C_param, dt_proj)

        # Skip connection with D
        y = y + x_conv * self.D

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return y

    def _ssm_scan(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Selective scan operation.

        Args:
            x: (B, T, d_inner) - input
            A: (d_state,) - state transition (diagonal)
            B: (B, T, d_state) - input projection
            C: (B, T, d_state) - output projection
            dt: (B, T, d_inner) - timestep
        """
        B_size, T, d_inner = x.shape
        d_state = A.shape[0]

        # Initialize state
        h = torch.zeros(B_size, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            # Discretized A: exp(A * dt)
            # A: (d_state,), dt[:, t, :]: (B, d_inner)
            # Want: (B, d_inner, d_state)
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            dA = torch.exp(A.view(1, 1, -1) * dt_t)  # (B, d_inner, d_state)

            # Discretized B: dt * B
            # dt[:, t, :]: (B, d_inner), B[:, t, :]: (B, d_state)
            # Want: (B, d_inner, d_state)
            dB = dt_t * B[:, t, :].unsqueeze(1)  # (B, d_inner, d_state)

            # State update: h = dA * h + dB * x
            x_t = x[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            h = dA * h + dB * x_t

            # Output: y = C @ h
            # h: (B, d_inner, d_state), C[:, t, :]: (B, d_state)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, T, d_inner)


class BiSSMLayer(nn.Module):
    """
    Bidirectional SSM layer - processes sequence in both directions.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Forward and backward SSM
        self.ssm_fwd = SelectiveSSM(d_model, d_state, d_conv, expand, dropout)
        self.ssm_bwd = SelectiveSSM(d_model, d_state, d_conv, expand, dropout)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Pre-norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        residual = x
        x = self.norm(x)

        # Forward direction
        y_fwd = self.ssm_fwd(x)

        # Backward direction (flip, process, flip back)
        x_bwd = torch.flip(x, dims=[1])
        y_bwd = self.ssm_bwd(x_bwd)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # Fuse directions
        y = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.fusion(y)

        return y + residual


class BiSSM(nn.Module):
    """
    Bidirectional Selective State Space Model.

    Stack of BiSSM layers for temporal modeling with:
    - O(n) complexity (vs O(n²) for transformers)
    - Bidirectional context
    - Input-dependent state transitions
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            BiSSMLayer(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - phonologically-aware spatial features
        Returns:
            (B, T, D) - temporal features
        """
        for layer in self.layers:
            x = layer(x)

        return self.final_norm(x)
