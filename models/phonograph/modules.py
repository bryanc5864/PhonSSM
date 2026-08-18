"""
Backbone building blocks for PhonoGraph. Deliberately GCN-based (not transformer /
Mamba) because the literature is unanimous that attention/SSM backbones are
data-hungry and lose to topology-prior GCNs when trained from scratch on ~1.5k
clips.

- CTRGC: channel-wise topology-refinement graph conv (CTR-GCN, Chen et al. ICCV'21).
  A static skeleton adjacency is refined by a per-channel, sample-specific delta
  computed from pairwise joint-feature differences. Strong local prior + adaptivity
  without the data appetite of full attention.
- MultiScaleTCN: multi-branch dilated temporal conv (MS-G3D / CTR-GCN style) --
  cheap multi-scale temporal receptive field.
- PartBlock: residual [unit_gcn -> MultiScaleTCN].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        rel_channels = max(8, in_channels // rel_reduction)
        self.conv1 = nn.Conv2d(in_channels, rel_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, rel_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv4 = nn.Conv2d(rel_channels, out_channels, 1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            conv_init(m)

    def forward(self, x, A, alpha):
        # x: (N, C, T, V)
        x1 = self.conv1(x).mean(-2)   # (N, rel, V)
        x2 = self.conv2(x).mean(-2)   # (N, rel, V)
        x3 = self.conv3(x)            # (N, out, T, V)
        # pairwise joint difference -> per-channel topology delta
        delta = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))   # (N, rel, V, V)
        delta = self.conv4(delta)                                 # (N, out, V, V)
        A_eff = delta * alpha + A.unsqueeze(0).unsqueeze(0)        # (N, out, V, V)
        return torch.einsum('ncuv,nctv->nctu', A_eff, x3)         # (N, out, T, V)


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_subset=3):
        super().__init__()
        self.num_subset = num_subset
        self.convs = nn.ModuleList([CTRGC(in_channels, out_channels) for _ in range(num_subset)])
        # learnable base adjacency, initialized from the normalized part adjacency
        A_init = torch.from_numpy(A).float().unsqueeze(0).repeat(num_subset, 1, 1)
        self.PA = nn.Parameter(A_init.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                      nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(self.bn.weight, 1e-6)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = None
        for i in range(self.num_subset):
            z = self.convs[i](x, self.PA[i], self.alpha)
            out = z if out is None else out + z
        out = self.bn(out)
        out = out + self.down(x)
        return self.relu(out)


class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 dilations=(1, 2), residual=True):
        super().__init__()
        n_branches = len(dilations) + 2
        assert out_channels % n_branches == 0
        bc = out_channels // n_branches
        self.branches = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size + (kernel_size - 1) * (d - 1) - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(),
                nn.Conv2d(bc, bc, (kernel_size, 1), (stride, 1), (pad, 0), dilation=(d, 1)),
                nn.BatchNorm2d(bc)))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(),
            nn.MaxPool2d((3, 1), (stride, 1), (1, 0)), nn.BatchNorm2d(bc)))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, bc, 1, stride=(stride, 1)), nn.BatchNorm2d(bc)))
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels))
        self.act = nn.ReLU()
        self.apply(conv_init)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return self.act(out + self.residual(x))


class PartBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.gcn = unit_gcn(in_channels, out_channels, A)
        self.tcn = MultiScaleTCN(out_channels, out_channels, stride=stride, residual=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels))
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.tcn(self.gcn(x))
        return self.act(y + self.residual(x))


class PartEncoder(nn.Module):
    """Per-part GCN encoder: 3->base->2base->2base with one temporal downsample.
    Returns a temporally-pooled part embedding (N, C) and the pre-pool map."""
    def __init__(self, in_channels, A, base=64, drop=0.0):
        super().__init__()
        num_point = A.shape[0]
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        self.num_point = num_point
        self.in_channels = in_channels
        self.l1 = PartBlock(in_channels, base, A, residual=False)
        self.l2 = PartBlock(base, base, A)
        self.l3 = PartBlock(base, 2 * base, A, stride=2)
        self.l4 = PartBlock(2 * base, 2 * base, A)
        self.drop = nn.Dropout(drop)
        self.out_channels = 2 * base

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
        x = self.drop(x)
        emb = x.mean(-1).mean(-1)   # (N, out_c) global pool over T and V
        return emb, x
