import torch
import torch.nn as nn
from collections import namedtuple
import torchaudio
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLAP_SR = 48_000

def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([1, 2], keepdim=keepdim)

def upsample(in_tens, out_HW=(64, 64)):
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

class LPAPS(nn.Module):
    """LPIPS-like perceptual distance over CLAP audio-tower feature maps.

    Pass in a CLAP handle (CLAPScore or laion_clap.CLAP_Module wrapped by .model).
    """
    def __init__(self, clap_handle, spatial=False):
        super().__init__()
        self.net = clap_handle if hasattr(clap_handle, "model") else clap_handle
        if hasattr(self.net, "model"):
            self.net = self.net.model  # laion_clap module
        self.spatial = spatial
        self.audio_branch = self.net.model.audio_branch
        self.L = len(self.audio_branch.layers)
        self.eval()

    @torch.inference_mode()
    def _forward_features(self, x, longer_idx=None):
        x = self.audio_branch.patch_embed(x, longer_idx=longer_idx)
        if getattr(self.audio_branch, "ape", False):
            x = x + self.audio_branch.absolute_pos_embed
        x = self.audio_branch.pos_drop(x)

        outs = []
        for layer in self.audio_branch.layers:
            x, _ = layer(x)
            outs.append(x)
        clap_outputs = namedtuple("ClapOutputs", ['swin1', 'swin2', 'swin3', 'swin4'])
        return clap_outputs(outs[0], outs[1], outs[2], outs[3])

    @torch.inference_mode()
    def _preprocess(self, wav):
        # expects wav: [1, T] in [-1,1], at 48 kHz
        x = self.audio_branch.spectrogram_extractor(wav)
        x = self.audio_branch.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.audio_branch.bn0(x)
        x = x.transpose(1, 3)
        x = self.audio_branch.reshape_wav2img(x)
        return x

    @torch.inference_mode()
    def forward(self, wav_a, wav_b):
        # Accept file paths or tensors; ensure 48k mono tensors
        if isinstance(wav_a, (str, bytes, os.PathLike)):
            wav_a, sr_a = torchaudio.load(wav_a)
            if sr_a != CLAP_SR: wav_a = torchaudio.functional.resample(wav_a, sr_a, CLAP_SR)
            wav_a = wav_a.mean(0, keepdim=True)
        if isinstance(wav_b, (str, bytes, os.PathLike)):
            wav_b, sr_b = torchaudio.load(wav_b)
            if sr_b != CLAP_SR: wav_b = torchaudio.functional.resample(wav_b, sr_b, CLAP_SR)
            wav_b = wav_b.mean(0, keepdim=True)

        a = self._forward_features(self._preprocess(wav_a.to(DEVICE)))
        b = self._forward_features(self._preprocess(wav_b.to(DEVICE)))

        diffs = []
        for aa, bb in zip(a, b):
            fa = normalize_tensor(aa)
            fb = normalize_tensor(bb)
            d  = (fa - fb) ** 2
            if self.spatial:
                diffs.append(upsample(d.sum(dim=1, keepdim=True), out_HW=wav_a.shape[1:]))
            else:
                diffs.append(spatial_average(d.sum(dim=1, keepdim=True), keepdim=True))

        val = torch.zeros_like(diffs[0])
        for r in diffs: val = val + r
        return val.squeeze().item()
