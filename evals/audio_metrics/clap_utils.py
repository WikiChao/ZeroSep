import torch, torchaudio
from functools import lru_cache
from frechet_audio_distance import CLAPScore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLAP_SR = 48_000

def init_clap_score(submodel_name: str = "630k-audioset", enable_fusion: bool = False, verbose: bool = False):
    clap = CLAPScore(
        submodel_name=submodel_name,
        verbose=verbose,
        enable_fusion=enable_fusion,
    )
    return clap

@lru_cache(maxsize=None)
def _text_emb_cached(prompt: str, clap):
    handle = clap if hasattr(clap, "get_text_embedding") else clap.model
    return handle.get_text_embedding([prompt], use_tensor=True).to(DEVICE)

def clap_sim_text(wav_path, prompt: str, clap) -> float:
    wav, sr = torchaudio.load(str(wav_path))
    if sr != CLAP_SR:
        wav = torchaudio.functional.resample(wav, sr, CLAP_SR)
    wav = wav.mean(0, keepdim=True)
    model = clap if hasattr(clap, "get_audio_embedding_from_data") else clap.model
    with torch.no_grad():
        a_emb = model.get_audio_embedding_from_data(wav.to(DEVICE), use_tensor=True)
        return torch.nn.functional.cosine_similarity(a_emb, _text_emb_cached(prompt, clap))[0].item()

def clap_sim_audio(wav1_path, wav2_path, clap) -> float:
    wav1, sr1 = torchaudio.load(str(wav1_path))
    wav2, sr2 = torchaudio.load(str(wav2_path))
    if sr1 != CLAP_SR:
        wav1 = torchaudio.functional.resample(wav1, sr1, CLAP_SR)
    if sr2 != CLAP_SR:
        wav2 = torchaudio.functional.resample(wav2, sr2, CLAP_SR)
    wav1 = wav1.mean(0, keepdim=True)
    wav2 = wav2.mean(0, keepdim=True)
    model = clap if hasattr(clap, "get_audio_embedding_from_data") else clap.model
    with torch.no_grad():
        a1 = model.get_audio_embedding_from_data(wav1.to(DEVICE), use_tensor=True)
        a2 = model.get_audio_embedding_from_data(wav2.to(DEVICE), use_tensor=True)
        return torch.nn.functional.cosine_similarity(a1, a2)[0].item()
