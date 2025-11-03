import torch, torchaudio, numpy as np, mir_eval

def _stack_pair(ref1, ref2, est1, est2):
    L = min(ref1.size(-1), ref2.size(-1), est1.size(-1), est2.size(-1))
    r = torch.vstack([ref1[..., :L], ref2[..., :L]]).cpu().numpy()
    e = torch.vstack([est1[..., :L], est2[..., :L]]).cpu().numpy()
    return r, e

def sdr_mean(ref1, ref2, est1, est2):
    r, e = _stack_pair(ref1, ref2, est1, est2)
    sdr, _, _, _ = mir_eval.separation.bss_eval_sources(r, e, compute_permutation=False)
    return float(np.mean(sdr))

def sdr_sir_sar(ref1, ref2, est1, est2):
    r, e = _stack_pair(ref1, ref2, est1, est2)
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(r, e, compute_permutation=False)
    return float(np.mean(sdr)), float(np.mean(sir)), float(np.mean(sar))
