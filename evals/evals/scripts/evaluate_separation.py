#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, json, tempfile, shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import torch, torchaudio

from audio_metrics.torch_patch import torch  # patch torch.load
from audio_metrics.fad_utils import make_fad_clap
from audio_metrics.clap_utils import init_clap_score, clap_sim_text, clap_sim_audio
from audio_metrics.lpaps import LPAPS, CLAP_SR
from audio_metrics.alignment import choose_assignment_sdr, choose_assignment_l2
from audio_metrics.bss_eval import sdr_sir_sar
from audio_metrics.io_utils import find_two_wavs, find_mix_file, parse_prompts_from_folder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ap = argparse.ArgumentParser(description="Evaluate separation metrics (FAD, LPAPS, CLAP, SDR/SIR/SAR).")
    ap.add_argument("--gt", required=True, type=Path, help="GT root (contains subfolders with gt1.wav, gt2.wav, mix.wav)")
    ap.add_argument("--pred", required=True, type=Path, help="Prediction root (mirrors subfolders, contains two predicted wavs)")
    ap.add_argument("--assignment", choices=["sdr", "lpaps", "l2"], default="sdr", help="How to assign pred↔gt (default: sdr)")
    ap.add_argument("--direct", action="store_true", help="Do not permute; assume pred1→gt1, pred2→gt2")
    ap.add_argument("--nmf-suffix", action="store_true", help="Use pred_1.wav / pred_2.wav naming for predictions")
    ap.add_argument("--include-encodec", action="store_true", help="Also compute EnCodec encoder embedding L1/L2")
    ap.add_argument("--save-csv", type=Path, default=None, help="Save per-source metrics to CSV path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # init metric handles
    if args.verbose: print("[Init] CLAP / FAD / LPAPS ...")
    clap = init_clap_score(submodel_name="630k-audioset", enable_fusion=False, verbose=False)
    fad  = make_fad_clap(sample_rate=CLAP_SR, submodel="630k-audioset", enable_fusion=False, verbose=False)
    lpaps_model = LPAPS(clap_handle=clap).to(DEVICE).eval()

    if args.include_encodec:
        from audio_metrics.encodec_metrics import EncodecEmbedMetric
        enc_metric = EncodecEmbedMetric()
    else:
        enc_metric = None

    # tmp dirs for FAD & CLAP prompts
    tmp_root = Path(tempfile.mkdtemp(prefix="eval_sep_"))
    tmp_gt   = tmp_root / "gt"
    tmp_pr   = tmp_root / "pred"
    (tmp_gt).mkdir(parents=True, exist_ok=True)
    (tmp_pr).mkdir(parents=True, exist_ok=True)
    prompts_csv = tmp_pr / "prompts.csv"

    # CSV header
    per_rows = []
    if args.save_csv:
        csv_f = open(args.save_csv, "w", newline="")
        writer = csv.DictWriter(csv_f, fieldnames=[
            "sample","side","LPAPS","CLAP_text","CLAP_text_neg","CLAP_audio","CLAP_audio_neg",
            "SDR","SIR","SAR","Emb_L1","Emb_L2"
        ])
        writer.writeheader()
    else:
        writer = None
        csv_f = None

    with open(prompts_csv, "w", newline="") as f_csv:
        p_writer = csv.DictWriter(f_csv, fieldnames=["file", "caption"])
        p_writer.writeheader()

        samples = sorted([p for p in args.gt.iterdir() if p.is_dir()])
        for sample_dir in tqdm(samples, desc="Evaluating"):
            name = sample_dir.name
            gt1p, gt2p = sample_dir / "gt1.wav", sample_dir / "gt2.wav"
            mix_gt = sample_dir / "mix.wav"
            if not (gt1p.exists() and gt2p.exists() and mix_gt.exists()):
                if args.verbose: print(f"[Skip] {name}: missing gt1/gt2/mix")
                continue

            pred_dir = args.pred / name
            try:
                pr1p, pr2p = find_two_wavs(pred_dir, args.nmf_suffix)
            except Exception as e:
                if args.verbose: print(f"[Skip] {name}: {e}")
                continue
            mix_pred = find_mix_file(pred_dir)  # may be None

            # sr info
            sr_g1 = torchaudio.info(gt1p).sample_rate
            sr_g2 = torchaudio.info(gt2p).sample_rate
            sr_p1 = torchaudio.info(pr1p).sample_rate
            sr_p2 = torchaudio.info(pr2p).sample_rate

            # load mono tensors
            g1, _ = torchaudio.load(gt1p); g2, _ = torchaudio.load(gt2p)
            p1, _ = torchaudio.load(pr1p); p2, _ = torchaudio.load(pr2p)
            g1, g2 = g1.mean(0, keepdim=True), g2.mean(0, keepdim=True)
            p1, p2 = p1.mean(0, keepdim=True), p2.mean(0, keepdim=True)

            # assignment
            if args.direct:
                G1, G2, P1, P2 = g1, g2, p1, p2
                swapped = False
            else:
                if args.assignment == "sdr":
                    (G1, G2, P1, P2), swapped = choose_assignment_sdr(gt1p, gt2p, pr1p, pr2p)
                elif args.assignment == "l2":
                    (G1, G2, P1, P2), swapped = choose_assignment_l2(gt1p, gt2p, pr1p, pr2p)
                else:  # lpaps
                    d_direct = LPAPS(clap_handle=clap)(gt1p, pr1p) + LPAPS(clap_handle=clap)(gt2p, pr2p)
                    d_swap   = LPAPS(clap_handle=clap)(gt1p, pr2p) + LPAPS(clap_handle=clap)(gt2p, pr1p)
                    if d_direct <= d_swap:
                        G1, G2, P1, P2 = g1, g2, p1, p2
                        swapped = False
                    else:
                        G1, G2, P1, P2 = g1, g2, p2, p1
                        swapped = True

            # derive srs per side
            sr_ref_a, sr_est_a = sr_g1, (sr_p1 if not swapped else sr_p2)
            sr_ref_b, sr_est_b = sr_g2, (sr_p2 if not swapped else sr_p1)

            # SDR/SIR/SAR (computed on raw tensors)
            sdr, sir, sar = sdr_sir_sar(G1, G2, P1, P2)

            # save aligned clips at 48k for FAD
            for tag, ref_t, est_t, sr_r, sr_e in (
                ("a", G1, P1, sr_ref_a, sr_est_a),
                ("b", G2, P2, sr_ref_b, sr_est_b),
            ):
                fn = f"{name}_{tag}.wav"
                rt = torchaudio.functional.resample(ref_t, sr_r, CLAP_SR) if sr_r != CLAP_SR else ref_t
                et = torchaudio.functional.resample(est_t, sr_e, CLAP_SR) if sr_e != CLAP_SR else est_t
                torchaudio.save(tmp_gt/fn, rt, CLAP_SR)
                torchaudio.save(tmp_pr/fn, et, CLAP_SR)
                # write prompt mapping
                inst1, inst2 = parse_prompts_from_folder(name)
                caption = inst1 if tag=="a" else inst2
                p_writer.writerow({"file": fn, "caption": caption})

            # per-side metrics
            for tag, ref_t, est_t, sr_r, sr_e in (
                ("a", G1, P1, sr_ref_a, sr_est_a),
                ("b", G2, P2, sr_ref_b, sr_est_b),
            ):
                fn = f"{name}_{tag}.wav"
                # LPAPS on tensors
                lp = lpaps_model(ref_t, est_t)

                # CLAP text/audio
                inst1, inst2 = parse_prompts_from_folder(name)
                cap = inst1 if tag=="a" else inst2
                cap_neg = inst2 if tag=="a" else inst1
                cap_pred = clap_sim_text(tmp_pr/fn, cap, clap)
                cap_pred_neg = clap_sim_text(tmp_pr/fn, cap_neg, clap)
                ca_pred = clap_sim_audio(tmp_pr/fn, tmp_gt/fn, clap)
                other_fn = f"{name}_{'b' if tag=='a' else 'a'}.wav"
                ca_pred_neg = clap_sim_audio(tmp_pr/fn, tmp_gt/other_fn, clap)

                # encodec metrics
                emb_l1 = emb_l2 = float("nan")
                if enc_metric is not None:
                    emb_l1, emb_l2 = enc_metric.embed_l1_l2(ref_t, est_t, sr_r, sr_e)

                row = {
                    "sample": name, "side": tag,
                    "LPAPS": lp,
                    "CLAP_text": cap_pred,
                    "CLAP_text_neg": cap_pred_neg,
                    "CLAP_audio": ca_pred,
                    "CLAP_audio_neg": ca_pred_neg,
                    "SDR": sdr, "SIR": sir, "SAR": sar,
                    "Emb_L1": emb_l1, "Emb_L2": emb_l2,
                }
                per_rows.append(row)
                if writer: writer.writerow(row)

    # Aggregate
    def mean_of(key):
        vals = [r[key] for r in per_rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    fad_score = fad.score(str(tmp_gt), str(tmp_pr), dtype="float32")

    summary = {
        "num_sources": len(per_rows),
        "FAD": round(fad_score, 6),
        "LPAPS_mean": round(mean_of("LPAPS"), 6),
        "CLAP_text_mean": round(mean_of("CLAP_text"), 6),
        "CLAP_text_neg_mean": round(mean_of("CLAP_text_neg"), 6),
        "CLAP_audio_mean": round(mean_of("CLAP_audio"), 6),
        "CLAP_audio_neg_mean": round(mean_of("CLAP_audio_neg"), 6),
        "SDR_mean_dB": round(mean_of("SDR"), 3),
        "SIR_mean_dB": round(mean_of("SIR"), 3),
        "SAR_mean_dB": round(mean_of("SAR"), 3),
        "Emb_L1_mean": round(mean_of("Emb_L1"), 6),
        "Emb_L2_mean": round(mean_of("Emb_L2"), 6),
    }

    print("\n============= SUMMARY =============")
    for k, v in summary.items():
        print(f"{k:20s}: {v}")
    out_json = Path("results_summary.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary → {out_json.resolve()}")

    if writer:
        csv_f.close()
        print(f"Saved per-source CSV → {args.save_csv.resolve()}")

    shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
