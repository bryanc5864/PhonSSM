"""
faithful port of DSTA-SLR's model/fstgan.py — the model that actually produced
their published WLASL numbers. despite the repo name this is not the "decoupled
spatial-temporal attention" transformer: it is a tanh-gated 8-subset learnable
spatial mixing unit (unit_san) + Edge_feature_conv joint-to-edge aggregation +
3x multi-branch temporal conv (MSTCN) + a squeeze-excite spatial/temporal/channel
gate, in 4 residual blocks with channels 64->128->256->512.

dead code in the original (unit_tan, global_tan, Edge_conv, MHSA/RPE_MHSA,
MultiScale_TemporalConv, unit_tcn_dilated, UnfoldTemporalWindows) is dropped —
checked unused by Block/Model. the hardcoded device="cuda" in unit_san is gone so
parameters follow .to(device). deprecated nn.init.constant/normal swapped for the
underscore forms; same behaviour.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from .graph import Graph


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding,
                     groups=dim_in, stride=stride, bias=bias),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class DropBlock_Ske(nn.Module):
    def __init__(self, num_point, block_size=7):
        super().__init__()
        self.keep_prob = 0.0
        self.block_size = block_size
        self.num_point = num_point

    def forward(self, input, keep_prob, A):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()
        input_abs = torch.mean(torch.mean(torch.abs(input), dim=2), dim=1).detach()
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()
        gamma = (1.0 - self.keep_prob) / (1 + 1.92)
        M_seed = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).to(
            device=input.device, dtype=input.dtype)
        M = torch.matmul(M_seed, A)
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        mask = (1 - M).view(n, 1, 1, self.num_point)
        return input * mask * mask.numel() / mask.sum()


class DropBlockT_1d(nn.Module):
    def __init__(self, block_size=7):
        super().__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()
        input_abs = torch.mean(torch.mean(torch.abs(input), dim=3), dim=1).detach()
        input_abs = (input_abs / torch.sum(input_abs) * input_abs.numel()).view(n, 1, t)
        gamma = (1.0 - self.keep_prob) / self.block_size
        input1 = input.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1, c * v, 1)
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        mask = (1 - Msum).to(device=input.device, dtype=input.dtype)
        return (input1 * mask * mask.numel() / mask.sum()).view(n, c, v, t).permute(0, 1, 3, 2)


class Edge_feature_conv(nn.Module):
    def __init__(self, in_channels, num_joint=27, edge_num=6):
        super().__init__()
        self.num_joint = num_joint
        self.edge_num = edge_num
        self.edge_features = nn.Parameter(torch.randn(in_channels, edge_num), requires_grad=True)
        nn.init.trunc_normal_(self.edge_features, std=0.02)
        self.norm1 = nn.LayerNorm(in_channels)
        self.transform = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.alphas = nn.Parameter(torch.ones(1, 1, edge_num, 1), requires_grad=True)

    def forward(self, x):
        n, c, t, v = x.shape
        x = rearrange(x, "b c t v -> (b t) v c")
        x = self.norm1(x)
        x = rearrange(x, "(b t) v c -> b c t v", b=n)
        x = self.transform(x)
        attention = torch.tanh(torch.einsum("nctv, ce->ntev", x, self.edge_features)) / c * self.alphas
        x = torch.einsum("ce,ntev->nctv", (self.edge_features, attention))
        return x


class unit_san(nn.Module):
    """The actual "spatial mixing" unit used by Block/Model: tanh-gated 8-subset
    learnable attention over joints + an edge-token auxiliary path. Not
    scaled-dot-product self-attention despite living in a class named for it."""

    def __init__(self, in_channels, out_channels, A, groups, num_point, num_subset=3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        num_subset = 8  # matches upstream: hardcoded, overrides ctor arg
        self.num_subset = num_subset
        self.inter_channels = out_channels // num_subset
        self.attention0s = nn.Parameter(
            torch.ones(1, num_subset, self.inter_channels, num_point, num_point) / num_point,
            requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.in_nets = nn.Conv2d(out_channels, 2 * num_subset * self.inter_channels, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
        self.bn0 = nn.BatchNorm2d(self.inter_channels * num_subset)
        self.tan = nn.Tanh()
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        mlp_hidden_dim = int(out_channels * 4)
        self.mlp = Mlp(in_features=out_channels, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.0)

        self.Linear_weight = nn.Parameter(
            torch.zeros(in_channels, self.inter_channels * num_subset), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(0.5 / (self.inter_channels * num_subset)))

        self.Linear_bias = nn.Parameter(
            torch.zeros(1, self.inter_channels * num_subset, 1, 1), requires_grad=True)
        nn.init.constant_(self.Linear_bias, 1e-6)

        self.Edge_conv = Edge_feature_conv(out_channels)
        self.edge_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x0):
        x = torch.einsum("nctw,cd->ndtw", (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, c, t, v = x.size()
        x_resi = x
        edge_features = self.Edge_conv(x) * self.edge_weight
        x = rearrange(x, "b c t v -> (b t) v c")
        x = self.norm1(x)
        x = rearrange(x, "(b t) v c -> b c t v", b=n)
        q, k = torch.chunk(
            self.in_nets(x).view(n, 2 * self.num_subset, self.inter_channels, t, v), 2, dim=1)
        attention = self.tan(
            torch.einsum("nkctu,nkctv->nkuv", [q, k]) / (self.inter_channels * t)
        ) * self.alphas
        x = x.view(n, self.num_subset, -1, t, v)
        x = torch.einsum("nkctv,nkvw->nkctw", (x, attention)).view(n, -1, t, v) + x_resi

        x_resi = x
        x = rearrange(x, "b c t v -> (b t) v c")
        x = self.norm2(x)
        x = x_resi + rearrange(self.mlp(x), "(b t) v c -> b c t v", b=n) + edge_features

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum("nkctv,nkcvw->nkctw", (x, self.attention0s.repeat(n, 1, 1, 1, 1))).view(n, -1, t, v)

        x = self.bn(x)
        x = x + self.down(x0)
        x = self.relu(x)
        return x


class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.pool = DepthWiseConv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                    padding=0, stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.pool(x))


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, num_point=25, block_size=41):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)
        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))
        return self.dropT(self.dropS(x, keep_prob, A), keep_prob)


class MSTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 7), stride=1, num_point=27, block_size=41):
        super().__init__()
        self.num_branches = len(kernel_sizes)
        assert out_channels % self.num_branches == 0
        branch_channels = out_channels // self.num_branches
        self.branches = nn.ModuleList([
            unit_tcn(in_channels, branch_channels, kernel_size=k, stride=stride,
                    num_point=num_point, block_size=block_size)
            for k in kernel_sizes
        ])

    def forward(self, x, keep_prob, A):
        return torch.cat([b(x, keep_prob, A) for b in self.branches], dim=1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size,
                stride=1, residual=True, is_first=False, window_size=120, **kwargs):
        super().__init__()
        tmp_c = out_channels if is_first else in_channels
        self.san = unit_san(in_channels, tmp_c, A, groups, num_point)

        self.tcn = MSTCN(tmp_c, out_channels, stride=1, num_point=num_point)
        self.tcn2 = MSTCN(out_channels, out_channels, stride=1, num_point=num_point)
        self.tcn3 = MSTCN(out_channels, out_channels, stride=stride, num_point=num_point)

        self.relu = nn.ReLU()

        self.attention = True
        self.sigmoid = nn.Sigmoid()
        self.conv_ta = nn.Conv1d(tmp_c, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)
        ker_jpt = num_point - 1 if not num_point % 2 else num_point
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(tmp_c, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)
        rr = 2
        self.fc1c = nn.Linear(tmp_c, tmp_c // rr)
        self.fc2c = nn.Linear(tmp_c // rr, tmp_c)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

        self.register_buffer(
            "A", torch.tensor(np.sum(np.reshape(A.astype(np.float32), [3, num_point, num_point]), axis=0),
                              dtype=torch.float32))

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn_skip(in_channels, out_channels, kernel_size=1, stride=stride)

        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob):
        y = self.san(x)
        # spatial attention
        se = y.mean(-2)
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-2) + y
        # temporal attention
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y
        # channel attention
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        y = self.tcn(y, keep_prob, self.A)
        y = self.tcn2(y, keep_prob, self.A)
        y = self.tcn3(y, keep_prob, self.A)
        x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(y + x_skip)


class Model(nn.Module):
    """Faithful port of model.fstgan.Model with graph.sign_27.Graph baked in."""

    def __init__(self, num_class=2000, num_point=27, num_person=1, groups=16,
                block_size=41, in_channels=3, inner_dim=64, drop_layers=2, depth=4,
                window_size=120):
        super().__init__()
        self.graph = Graph(labeling_mode="spatial")
        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.drop_layers = depth - drop_layers
        inner_dim_expansion = [2 ** i for i in range(depth)]

        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(Block(in_channels, inner_dim, A, groups, num_point, block_size,
                                    residual=False, window_size=window_size, is_first=True))
            else:
                layers.append(Block(
                    inner_dim * inner_dim_expansion[i - 1], inner_dim * inner_dim_expansion[i],
                    A, groups, num_point, block_size,
                    stride=inner_dim_expansion[i] // inner_dim_expansion[i - 1],
                    residual=True, window_size=window_size // inner_dim_expansion[i]))
        self.layers = nn.ModuleList(layers)

        self.fc = nn.Linear(inner_dim * inner_dim_expansion[-1], num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, keep_prob=0.9):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for u, blk in enumerate(self.layers):
            x = blk(x, 1.0 if u < self.drop_layers else keep_prob)
        c_new = x.size(1)
        x = x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        return self.fc(x)
