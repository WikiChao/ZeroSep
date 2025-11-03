import torch, torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EncodecEmbedMetric:
    def __init__(self):
        self.model = EncodecModel.encodec_model_48khz().to(DEVICE).eval()
        self.sr = self.model.sample_rate  # 48000
        self.ch = self.model.channels     # 2

    @torch.inference_mode()
    def embed_l1_l2(self, ref, est, sr_ref, sr_est):
        ref_c = convert_audio(ref.cpu(), sr_ref, self.sr, self.ch)
        est_c = convert_audio(est.cpu(), sr_est, self.sr, self.ch)
        zb = self.model.encoder(ref_c.unsqueeze(0).to(DEVICE))
        za = self.model.encoder(est_c.unsqueeze(0).to(DEVICE))
        if isinstance(zb, (list, tuple)): zb = zb[0]
        if isinstance(za, (list, tuple)): za = za[0]
        Tm = min(zb.shape[-1], za.shape[-1])
        zb, za = zb[..., :Tm], za[..., :Tm]
        l1 = torch.nn.functional.l1_loss(za, zb).item()
        l2 = torch.nn.functional.mse_loss(za, zb).item()
        return l1, l2
