import os
import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
import random
from typing import Optional, List, Tuple, Dict
from models import PipelineWrapper
import torchaudio



def get_spec(wav: torch.Tensor, fn_STFT: torch.nn.Module) -> torch.Tensor:
    return fn_STFT.mel_spectrogram(torch.clip(wav[:, 0], -1, 1).cpu())[0]


def load_audio(audio_path: str, fn_STFT, left: int = 0, right: int = 0, device: Optional[torch.device] = None,
               return_wav: bool = False, stft: bool = False, model_sr: Optional[int] = None) -> torch.Tensor:
    if stft:  # AudioLDM/tango loading to spectrogram
        if type(audio_path) is str:
            import audioldm
            import audioldm.audio

            duration = audioldm.utils.get_duration(audio_path)

            mel, _, wav = audioldm.audio.wav_to_fbank(audio_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT)
            mel = mel.unsqueeze(0)
        else:
            mel = audio_path

        c, h, w = mel.shape
        left = min(left, w-1)
        right = min(right, w - left - 1)
        mel = mel[:, :, left:w-right]
        mel = mel.unsqueeze(0).to(device)

        if return_wav:
            return mel, 16000, duration, wav

        return mel, model_sr, duration
    else:
        waveform, sr = torchaudio.load(audio_path)
        if sr != model_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=model_sr)
        # waveform = waveform.numpy()[0, ...]

        def normalize_wav(waveform):
            waveform = waveform - torch.mean(waveform)
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            return waveform * 0.5

        waveform = normalize_wav(waveform)
        waveform = torch.FloatTensor(waveform)
        duration = waveform.shape[-1] / model_sr
        return waveform, model_sr, duration


def set_reproducability(seed: int, extreme: bool = True) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Extreme options
        if extreme:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        # Even more extreme options
        torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")


def get_height_of_spectrogram(length: int, ldm_stable: PipelineWrapper) -> int:
    vocoder_upsample_factor = np.prod(ldm_stable.model.vocoder.config.upsample_rates) / \
        ldm_stable.model.vocoder.config.sampling_rate

    if length is None:
        length = ldm_stable.model.unet.config.sample_size * ldm_stable.model.vae_scale_factor * \
            vocoder_upsample_factor

    height = int(length / vocoder_upsample_factor)

    # original_waveform_length = int(length * ldm_stable.model.vocoder.config.sampling_rate)
    if height % ldm_stable.model.vae_scale_factor != 0:
        height = int(np.ceil(height / ldm_stable.model.vae_scale_factor)) * ldm_stable.model.vae_scale_factor
        print(
            f"Audio length in seconds {length} is increased to {height * vocoder_upsample_factor} "
            f"so that it can be handled by the model. It will be cut to {length} after the "
            f"denoising process."
        )

    return height


def plot_corrs(args, corrs: List[np.array], in_corrs: List[List[torch.Tensor]],
               in_norms: List[List[torch.Tensor]], save_path: str,
               image_name_png: str, logging_dict: Dict[str, any], n_ev: int = 1) -> None:
    # Plot timesteps correlations
    # save_full_path_corrstxt = os.path.join(save_path, image_name_png + "_corrs.txt")
    save_full_path_corrspng = os.path.join(save_path, image_name_png + "_corrs.png")
    corrs_xs = np.arange(args.drift_start-1, args.drift_start-1 - len(corrs), -1)
    # with open(save_full_path_corrstxt, 'w') as f:
    #     f.write('\n'.join([str(x) for x in corrs]))
    for ev_num in range(n_ev):
        ev_corrs = [x[ev_num].detach().cpu().item() for x in corrs]
        plt.plot(corrs_xs, ev_corrs, label='ev ' + str(ev_num + 1))

        corrs_data = [[x, y] for (x, y) in zip(corrs_xs, ev_corrs)]
        corrs_table = wandb.Table(data=corrs_data, columns=["timestep", "correlation"])
        logging_dict[f"pc_correlations_{ev_num + 1}"] = wandb.plot.line(
            corrs_table, "timestep", "correlation", title=f"PCs Correlations #{ev_num + 1}")
    plt.legend()
    plt.savefig(save_full_path_corrspng)
    plt.close()

    incors_timesteps = np.arange(args.drift_start, args.drift_start - len(in_corrs), -1)
    # in_corrs = [[x.detach().cpu().item() for x in incorr] for incorr in in_corrs]
    # in_norms = [[x.detach().cpu().item() for x in in_norm] for in_norm in in_norms]
    if len(in_corrs) > 101:
        in_corrs1 = in_corrs[:len(in_corrs)//2]
        in_corrs2 = in_corrs[len(in_corrs)//2:]
        save_full_path_incorrs1png = os.path.join(save_path, image_name_png + "_incorrs1.png")
        save_full_path_incorrs2png = os.path.join(save_path, image_name_png + "_incorrs2.png")

        plt.figure(figsize=(10, 2*len(in_corrs1)))
        for i, incorr in enumerate(in_corrs1):
            plt.subplot(len(in_corrs1), 1, i+1)
            for ev_num in range(n_ev):
                ev_in_corrs = [x[ev_num].detach().cpu().item() for x in incorr]
                plt.plot(ev_in_corrs, label='ev ' + str(ev_num + 1))
            plt.title(f"timestep {incors_timesteps[i]}")
            plt.legend()
        plt.savefig(save_full_path_incorrs1png)
        plt.close()
        plt.figure(figsize=(10, 2*len(in_corrs2)))
        for i, incorr in enumerate(in_corrs2):
            plt.subplot(len(in_corrs2), 1, i+1)
            for ev_num in range(n_ev):
                ev_in_corrs = [x[ev_num].detach().cpu().item() for x in incorr]
                plt.plot(ev_in_corrs, label='ev ' + str(ev_num + 1))
            plt.title(f"timestep {incors_timesteps[i+len(in_corrs1)]}")
            plt.legend()
        plt.savefig(save_full_path_incorrs2png)
        plt.close()
    else:
        save_full_path_incorrspng = os.path.join(save_path, image_name_png + "_incorrs.png")
        plt.figure(figsize=(10, 2*len(in_corrs)))
        for i, incorr in enumerate(in_corrs):
            plt.subplot(len(in_corrs), 1, i+1)
            for ev_num in range(n_ev):
                ev_in_corrs = [x[ev_num].detach().cpu().item() for x in incorr]
                plt.plot(ev_in_corrs, label='ev ' + str(ev_num + 1))
            plt.title(f"timestep {incors_timesteps[i]}")
            plt.legend()
        plt.savefig(save_full_path_incorrspng)
        plt.close()

    for ev_num in range(n_ev):
        ev_in_corrs = [[x[ev_num].detach().cpu().item() for x in incorr] for incorr in in_corrs]

        logging_dict[f'convergence_{ev_num + 1}'] = wandb.plot.line_series(
            xs=np.arange(args.iters - 1), ys=ev_in_corrs,
            keys=np.arange(args.drift_start, args.drift_start - len(in_corrs), -1),
            title=f"Subspace iterations correlations PC#{ev_num + 1}", xname="iter")
