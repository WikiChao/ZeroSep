from pathlib import Path

def find_two_wavs(pred_dir: Path, nmf_suffix: bool):
    if nmf_suffix:
        cands = [pred_dir / "pred_1.wav", pred_dir / "pred_2.wav"]
    else:
        cands = [pred_dir / "pred1.wav", pred_dir / "pred2.wav"]
    if all(p.exists() for p in cands):
        return cands[0], cands[1]

    wavs = sorted(pred_dir.glob("*.wav"))
    if len(wavs) >= 2:
        return wavs[0], wavs[1]
    raise FileNotFoundError(f"Could not find two .wav files in {pred_dir}")

def find_mix_file(pred_dir: Path):
    for pat in ["mix.wav", "mixture.wav", "orig1.wav", "mix_pred.wav"]:
        p = pred_dir / pat
        if p.exists():
            return p
    return None

def parse_prompts_from_folder(name: str):
    if '+' not in name:
        return "source1", "source2"
    a, b = name.split('+', 1)
    a_tok = a.split('-')[-1]
    b_tok = b.split('-')[-1]
    return a_tok, b_tok
