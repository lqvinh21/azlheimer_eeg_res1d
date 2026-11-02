import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c, groups=8):
    return nn.GroupNorm(min(groups, c), c)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1, drop=0.0, k=7):
        super().__init__()
        self.n1 = gn(cin)
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.n2 = gn(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.drop = nn.Dropout(drop)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Conv1d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + (x if self.short is None else self.short(x))


class ResNet1Dv2(nn.Module):
    """Pre-act 1D-ResNet (GroupNorm) + nhánh tuổi (tắt được bằng use_age=False)."""

    def __init__(self, in_ch=19, base=32, n_classes=2, drop=0.25, use_age=True):
        super().__init__()
        self.use_age = use_age
        self.stem = nn.Conv1d(in_ch, base, 15, 2, 7, bias=False)
        chans = [base, base * 2, base * 4, base * 8]
        blocks, cin = [], base
        for i, c in enumerate(chans):
            blocks += [ResBlock1D(cin, c, 1 if i == 0 else 2, drop * 0.3),
                       ResBlock1D(c, c, 1, drop * 0.3)]
            cin = c
        self.blocks = nn.Sequential(*blocks)
        self.out_norm = gn(cin)
        self.age_mlp = nn.Sequential(nn.Linear(1, 16), nn.SiLU()) if use_age else None
        feat = cin * 2 + (16 if use_age else 0)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(feat, 64), nn.SiLU(),
                                  nn.Dropout(drop), nn.Linear(64, n_classes))

    def forward(self, x, age=None):
        h = F.silu(self.out_norm(self.blocks(self.stem(x))))
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        if self.use_age:
            h = torch.cat([h, self.age_mlp(age.unsqueeze(1))], 1)
        return self.head(h)


def make_model(use_age, cfg, device):
    return ResNet1Dv2(base=cfg.base_ch, drop=cfg.dropout, use_age=use_age).to(device)


@torch.no_grad()
def ema_update(ema, net, decay):
    for pe, p in zip(ema.parameters(), net.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for be, b in zip(ema.buffers(), net.buffers()):
        be.copy_(b)


def check_models(cfg, device):
    """Chạy thử forward cho cả hai biến thể và in số tham số."""
    for ua in (True, False):
        m = make_model(ua, cfg, device).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 19, cfg.win, device=device), torch.zeros(2, device=device))
        print(f'use_age={ua} | output {tuple(out.shape)} | params {sum(p.numel() for p in m.parameters()):,}')
        del m
