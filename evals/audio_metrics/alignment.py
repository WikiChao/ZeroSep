import torch, torchaudio
from .bss_eval import sdr_mean

def choose_assignment_sdr(gt1p, gt2p, pr1p, pr2p):
    g1, _ = torchaudio.load(gt1p); g2, _ = torchaudio.load(gt2p)
    p1, _ = torchaudio.load(pr1p); p2, _ = torchaudio.load(pr2p)
    g1, g2 = g1.mean(0, keepdim=True), g2.mean(0, keepdim=True)
    p1, p2 = p1.mean(0, keepdim=True), p2.mean(0, keepdim=True)
    if sdr_mean(g1, g2, p1, p2) >= sdr_mean(g1, g2, p2, p1):
        return (g1, g2, p1, p2), False
    return (g1, g2, p2, p1), True

def choose_assignment_l2(gt1p, gt2p, pr1p, pr2p):
    def _mean_l2(g1, g2, p1, p2):
        L = min(g1.shape[-1], g2.shape[-1], p1.shape[-1], p2.shape[-1])
        return ((g1[..., :L] - p1[..., :L]) ** 2).mean() + ((g2[..., :L] - p2[..., :L]) ** 2).mean()

    g1, _ = torchaudio.load(gt1p); g2, _ = torchaudio.load(gt2p)
    p1, _ = torchaudio.load(pr1p); p2, _ = torchaudio.load(pr2p)
    g1, g2 = g1.mean(0, keepdim=True), g2.mean(0, keepdim=True)
    p1, p2 = p1.mean(0, keepdim=True), p2.mean(0, keepdim=True)

    if _mean_l2(g1, g2, p1, p2) <= _mean_l2(g1, g2, p2, p1):
        return (g1, g2, p1, p2), False
    return (g1, g2, p2, p1), True
