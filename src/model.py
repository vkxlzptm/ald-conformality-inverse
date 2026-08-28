"""1D CNN encoder with a mixture-density head.

The head outputs a full mixture of diagonal Gaussians over the five parameters
rather than a point estimate, because this inverse problem is genuinely partly
degenerate: several parameter combinations reproduce the same profile to within
measurement noise.  A point estimate hides that; a posterior shows it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2.0 * math.pi)


class ProfileMDN(nn.Module):
    def __init__(self, in_ch=7, n_cond=2, n_out=5, n_mix=8, width=96):
        super().__init__()
        w = width

        def blk(i, o, stride):
            return nn.Sequential(
                nn.Conv1d(i, o, 5, stride=stride, padding=2),
                nn.BatchNorm1d(o), nn.GELU())

        self.trunk = nn.Sequential(
            blk(in_ch, w, 1), blk(w, w, 2),
            blk(w, 2 * w, 2), blk(2 * w, 2 * w, 2))
        self.head = nn.Sequential(
            nn.Linear(4 * w + n_cond, 4 * w), nn.GELU(),
            nn.Linear(4 * w, 4 * w), nn.GELU())
        self.pi = nn.Linear(4 * w, n_mix)
        self.mu = nn.Linear(4 * w, n_mix * n_out)
        self.ls = nn.Linear(4 * w, n_mix * n_out)
        self.n_mix, self.n_out = n_mix, n_out

    def forward(self, x, c):
        h = self.trunk(x)
        h = torch.cat([h.mean(-1), h.amax(-1), c], dim=1)
        h = self.head(h)
        k, d = self.n_mix, self.n_out
        return (self.pi(h),
                self.mu(h).view(-1, k, d),
                self.ls(h).view(-1, k, d).clamp(-7.0, 3.0))


def mdn_nll(pi_logits, mu, log_sigma, y):
    """Negative log likelihood of y under the predicted mixture."""
    logpi = F.log_softmax(pi_logits, dim=1)
    z = (y.unsqueeze(1) - mu) / log_sigma.exp()
    comp = -0.5 * (z ** 2 + LOG2PI) - log_sigma          # (B, K, D)
    return -torch.logsumexp(logpi + comp.sum(-1), dim=1).mean()


@torch.no_grad()
def mdn_moments(pi_logits, mu, log_sigma):
    """Mixture mean and standard deviation, per output dimension."""
    p = F.softmax(pi_logits, dim=1).unsqueeze(-1)
    mean = (p * mu).sum(1)
    var = (p * (log_sigma.exp() ** 2 + mu ** 2)).sum(1) - mean ** 2
    return mean, var.clamp_min(1e-12).sqrt()


@torch.no_grad()
def mdn_sample(pi_logits, mu, log_sigma, n):
    """Draw n posterior samples per item: (B, n, D)."""
    p = F.softmax(pi_logits, dim=1)
    k = torch.multinomial(p, n, replacement=True)         # (B, n)
    idx = k.unsqueeze(-1).expand(-1, -1, mu.shape[-1])
    m = torch.gather(mu, 1, idx)
    s = torch.gather(log_sigma, 1, idx).exp()
    return m + s * torch.randn_like(m)
