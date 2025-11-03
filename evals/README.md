# Audio Separation Evaluation (FAD · LPAPS · CLAP · SDR/SIR/SAR)

This toolkit evaluates 2-source separation results with a rich set of metrics:

- **FAD** (Fréchet Audio Distance) using CLAP embeddings
- **LPAPS** (LPIPS-style distance built on CLAP features) — lower is better
- **CLAP (text–audio)** similarity: predicted audio vs. text prompts from folder names
- **CLAP (audio–audio)** similarity: predicted audio vs. the corresponding GT source
- **BSS Eval**: SDR / SIR / SAR (via `mir_eval`)
- **Optional**: EnCodec encoder embedding L1/L2 distances (perceptual-ish)

## Install

```bash
pip install -r requirements.txt
```

> If you're in Colab, you may need: `pip install --upgrade pip` first.

## Expected Data Layout

Ground truth root (`--gt`) contains one subfolder per mixture:

```
GT_ROOT/
  sample_000_piano-xxx+violin-yyy/
    gt1.wav
    gt2.wav
    mix.wav
  sample_001_.../
    gt1.wav
    gt2.wav
    mix.wav
  ...
```

Predictions root (`--pred`) mirrors subfolders; each has two predicted files (default names: `pred1.wav`, `pred2.wav`). If your method saved them as `pred_1.wav` / `pred_2.wav`, pass `--nmf-suffix`.

```
PRED_ROOT/
  sample_000_piano-xxx+violin-yyy/
    pred1.wav
    pred2.wav
  sample_001_.../
    pred1.wav
    pred2.wav
  ...
```

Folder names should include two instrument keywords separated by `+`, such as
`piano-...+violin-...`. The script extracts prompts for CLAP from the **last token** of each side by splitting on `-` (customize if you need).

## Quickstart

```bash
python scripts/evaluate_separation.py   --gt /path/to/GT_ROOT   --pred /path/to/PRED_ROOT   --direct   --nmf-suffix   --save-csv results.csv   --include-encodec
```
- `--direct`: if pred1 -->gt1(pred2 -->gt2), pass --direct; Otherwise, you need to configure 'assignment' below
- `--assignment`: how to align (`gt1`↔`pred?`, `gt2`↔`pred?`); choose from `sdr`, `lpaps`, or `l2`.
- `--nmf-suffix`: use this if your predictions are named `pred_1.wav` / `pred_2.wav`.
- `--include-encodec`: also compute EnCodec encoder embedding losses (L1/L2).
- `--save-csv`: save per-source metrics to a CSV file.

The script prints aggregate metrics and also writes:
- `results_summary.json` (aggregate numbers)
- (optional) a CSV with per-track/per-source entries.

## Notes & Tips

- We use CLAP (630k-audioset, fusion disabled). Audio is resampled to 48 kHz where needed.
- FAD uses CLAP embeddings; temporary aligned WAVs are saved internally at 48 kHz.
- If `mir_eval` complains, double-check you installed `mir_eval` (not `mir_evals`).

## License

This repository assembles glue code around open-source components (laion-clap, frechet_audio_distance, mir_eval, encodec). Check each dependency for its license.
