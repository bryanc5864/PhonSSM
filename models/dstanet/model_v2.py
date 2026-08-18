"""
DSTA-Net v2: two changes on top of the faithful port in model.py, aimed at
WLASL100-only training with no external pretraining.

MSTCN gets 4 temporal branches (kernels 3/5/7/9) instead of 2 (5/7), for a wider
multi-scale receptive field — the usual ST-GCN++/CTR-GCN lever.

the head no longer takes one flat mean over (T*V). it mean-pools over joints,
then attention-pools over time. clips are uniformly resampled to 120 frames, so a
plain mean weights idle/padding frames the same as informative ones.

everything else (unit_san, Edge_feature_conv, SE gates, DropBlock, block
structure, channel progression) is identical to model.py.
"""
import math
import torch
import torch.nn as nn
import numpy as np

from .graph import Graph
from .model import (unit_san, unit_tcn_skip, unit_tcn, DropBlock_Ske, DropBlockT_1d,
                     conv_init, bn_init)


class MSTCN4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7, 9), stride=1,
                num_point=27, block_size=41):
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


class BlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size,
                stride=1, residual=True, is_first=False, window_size=120, **kwargs):
        super().__init__()
        tmp_c = out_channels if is_first else in_channels
        self.san = unit_san(in_channels, tmp_c, A, groups, num_point)

        self.tcn = MSTCN4(tmp_c, out_channels, stride=1, num_point=num_point)
        self.tcn2 = MSTCN4(out_channels, out_channels, stride=1, num_point=num_point)
        self.tcn3 = MSTCN4(out_channels, out_channels, stride=stride, num_point=num_point)

        self.relu = nn.ReLU()

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
        se = y.mean(-2)
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-2) + y
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        y = self.tcn(y, keep_prob, self.A)
        y = self.tcn2(y, keep_prob, self.A)
        y = self.tcn3(y, keep_prob, self.A)
        x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(y + x_skip)


class TemporalAttnPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(channels, channels // 4)
        self.act = nn.GELU()
        self.score = nn.Linear(channels // 4, 1)

    def forward(self, x):  # x: (N, C, T)
        h = self.act(self.fc(x.transpose(1, 2)))     # (N, T, C/4)
        w = torch.softmax(self.score(h), dim=1)       # (N, T, 1)
        return (x.transpose(1, 2) * w).sum(1)          # (N, C)


class ModelV2(nn.Module):
    """DSTA-Net with multi-scale MSTCN + spatial-mean/temporal-attention head."""

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
                layers.append(BlockV2(in_channels, inner_dim, A, groups, num_point, block_size,
                                      residual=False, window_size=window_size, is_first=True))
            else:
                layers.append(BlockV2(
                    inner_dim * inner_dim_expansion[i - 1], inner_dim * inner_dim_expansion[i],
                    A, groups, num_point, block_size,
                    stride=inner_dim_expansion[i] // inner_dim_expansion[i - 1],
                    residual=True, window_size=window_size // inner_dim_expansion[i]))
        self.layers = nn.ModuleList(layers)

        final_c = inner_dim * inner_dim_expansion[-1]
        self.temporal_pool = TemporalAttnPool(final_c)
        self.fc = nn.Linear(final_c, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, keep_prob=0.9):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for u, blk in enumerate(self.layers):
            x = blk(x, 1.0 if u < self.drop_layers else keep_prob)
        n_, c_new, t_new, v_new = x.size()
        x = x.mean(-1)                                    # spatial mean over joints -> (N*M, C, T)
        x = self.temporal_pool(x)                          # attention pool over time -> (N*M, C)
        x = x.view(N, M, c_new).mean(1)
        return self.fc(x)
