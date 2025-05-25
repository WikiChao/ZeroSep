import argparse
import calendar
import matplotlib.pyplot as plt
import os
import time
import torch
import torchaudio
import warnings
import wandb
from torch import inference_mode

from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from ddm_inversion.ddim_inversion import ddim_inversion, text2image_ldm_stable
from models import load_model
from utils import set_reproducability, load_audio, get_spec
from mir_eval.separation import bss_eval_sources

import os
os.environ["HF_HOME"] = "/home/cxu-serve/p62/chuang65/checkpoints"

HF_TOKEN = None # Needed for stable audio open. You can leave None when not using it


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run text-based audio editing.')
    parser.add_argument("--device_num", type=int, default=0, help="GPU device number")
    parser.add_argument('-s', "--seed", type=int, default=None, help="GPU device number")
    parser.add_argument("--model_id", type=str, choices=["cvssp/audioldm-s-full-v2",
                                                         "cvssp/audioldm-l-full",
                                                         "cvssp/audioldm2",
                                                         "cvssp/audioldm2-large",
                                                         "cvssp/audioldm2-music",
                                                         'declare-lab/tango-full-ft-audio-music-caps',
                                                         'declare-lab/tango-full-ft-audiocaps',
                                                         "stabilityai/stable-audio-open-1.0"
                                                         ],
                        default="cvssp/audioldm2-music", help='Audio diffusion model to use')

    parser.add_argument("--init_aud", type=str, required=True, help='Audio to invert and extract PCs from')
    parser.add_argument("--cfg_src", type=float, nargs='+', default=[3],
                        help='Classifier-free guidance strength for forward process')
    parser.add_argument("--cfg_tar", type=float, nargs='+', default=[12],
                        help='Classifier-free guidance strength for reverse process')
    parser.add_argument("--num_diffusion_steps", type=int, default=50,
                        help="Number of diffusion steps. TANGO and AudioLDM2 are recommended to be used with 200 steps"
                             ", while AudioLDM is recommeneded to be used with 100 steps")
    parser.add_argument("--target_prompt", type=str, nargs='+', default=[""], required=True,
                        help="Prompt to accompany the reverse process. Should describe the wanted edited audio.")
    parser.add_argument("--source_prompt", type=str, nargs='+', default=[""],
                        help="Prompt to accompany the forward process. Should describe the original audio.")
    parser.add_argument("--target_neg_prompt", type=str, nargs='+', default=[""],
                        help="Negative prompt to accompany the inversion and generation process")
    parser.add_argument("--tstart", type=int, nargs='+', default=[100],
                        help="Diffusion timestep to start the reverse process from. Controls editing strength.")
    parser.add_argument("--results_path", type=str, default="results", help="path to dump results")

    parser.add_argument("--cutoff_points", type=float, nargs='*', default=None)
    parser.add_argument("--mode", default="ours", choices=['ours', 'ddim'],
                        help="Run our editing or DDIM inversion based editing.")
    parser.add_argument("--fix_alpha", type=float, default=0.1)

    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_disable', action='store_true', default=True)

    args = parser.parse_args()
    args.eta = 1.
    args.numerical_fix = True
    args.test_rand_gen = False

    if args.model_id == "stabilityai/stable-audio-open-1.0" and HF_TOKEN is None:
        raise ValueError("HF_TOKEN is required for stable audio model")

    set_reproducability(args.seed, extreme=False)
    device = f"cuda:{args.device_num}"
    torch.cuda.set_device(args.device_num)

    model_id = args.model_id
    cfg_scale_src = args.cfg_src
    cfg_scale_tar = args.cfg_tar

    # same output
    current_GMT = time.gmtime()
    time_stamp_name = calendar.timegm(current_GMT)
    if args.mode == 'ours':
        image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
            f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
            f'skip_{int(args.num_diffusion_steps) - int(args.tstart[0])}_{time_stamp_name}'
    else:
        if args.tstart != args.num_diffusion_steps:
            image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
                f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
                f'skip_{int(args.num_diffusion_steps) - int(args.tstart[0])}_{time_stamp_name}'
        else:
            image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
                f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
                f'{args.num_diffusion_steps}timesteps_{time_stamp_name}'

    wandb.login(key='')
    wandb_run = wandb.init(project="AudInv", entity='', config={},
                           name=args.wandb_name if args.wandb_name is not None else image_name_png,
                           group=args.wandb_group,
                           mode='disabled' if args.wandb_disable else 'online',
                           settings=wandb.Settings(_disable_stats=True))
    wandb.config.update(args)

    eta = args.eta  # = 1
    if len(args.tstart) != len(args.target_prompt):
        if len(args.tstart) == 1:
            args.tstart *= len(args.target_prompt)
        else:
            raise ValueError("T-start amount and target prompt amount don't match.")
    args.tstart = torch.tensor(args.tstart, dtype=torch.int)
    skip = args.num_diffusion_steps - args.tstart

    ldm_stable = load_model(model_id, device, args.num_diffusion_steps, token=HF_TOKEN)
    x0, sr, duration = load_audio(args.init_aud, ldm_stable.get_fn_STFT(), device=device,
                                  stft=('stable-audio' not in model_id), model_sr=ldm_stable.get_sr())
    torch.cuda.empty_cache()
    with inference_mode():
        w0 = ldm_stable.vae_encode(x0)

        # find Zs and wts - forward process
        if args.mode == "ddim":
            if len(cfg_scale_src) > 1:
                raise ValueError("DDIM only supports one cfg_scale_src value")
            wT = ddim_inversion(ldm_stable, w0, args.source_prompt, cfg_scale_src[0],
                                num_inference_steps=args.num_diffusion_steps, skip=skip[0])
        else:
            wt, zs, wts, extra_info = inversion_forward_process(
                ldm_stable, w0, etas=eta,
                prompts=args.source_prompt, cfg_scales=cfg_scale_src,
                prog_bar=True,
                num_inference_steps=args.num_diffusion_steps,
                cutoff_points=args.cutoff_points,
                numerical_fix=args.numerical_fix,
                duration=duration)

        # iterate over decoder prompts
        save_path = os.path.join(f'./{args.results_path}/',
                                 model_id.split('/')[1],
                                 os.path.basename(args.init_aud).split('.')[0],
                                 'src_' + "__".join([x.replace(" ", "_") for x in args.source_prompt]),
                                 'dec_' + "__".join([x.replace(" ", "_") for x in args.target_prompt]) +
                                 "__neg__" + "__".join([x.replace(" ", "_") for x in args.target_neg_prompt]))
        os.makedirs(save_path, exist_ok=True)

        if args.mode == "ours":
            # reverse process (via Zs and wT)
            w0, _ = inversion_reverse_process(ldm_stable,
                                              xT=wts if not args.test_rand_gen else torch.randn_like(wts),
                                              tstart=args.tstart,
                                              fix_alpha=args.fix_alpha,
                                              etas=eta, prompts=args.target_prompt,
                                              neg_prompts=args.target_neg_prompt,
                                              cfg_scales=cfg_scale_tar, prog_bar=True,
                                              zs=zs[:int(args.num_diffusion_steps - min(skip))]
                                              if not args.test_rand_gen else torch.randn_like(
                                                  zs[:int(args.num_diffusion_steps - min(skip))]),
                                              #   zs=zs[skip:],
                                              cutoff_points=args.cutoff_points,
                                              duration=duration,
                                              extra_info=extra_info)
        else:  # ddim
            if skip != 0:
                warnings.warn("Plain DDIM Inversion should be run with t_start == num_diffusion_steps. "
                              "You are now running partial DDIM inversion.", RuntimeWarning)
            if len(cfg_scale_tar) > 1:
                raise ValueError("DDIM only supports one cfg_scale_tar value")
            if len(args.source_prompt) > 1:
                raise ValueError("DDIM only supports one args.source_prompt value")
            if len(args.target_prompt) > 1:
                raise ValueError("DDIM only supports one args.target_prompt value")
            w0 = text2image_ldm_stable(ldm_stable, args.target_prompt,
                                       args.num_diffusion_steps, cfg_scale_tar[0],
                                       wT,
                                       skip=skip)

    # vae decode image
    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w0)
        if 'stable-audio' not in model_id:
            if x0_dec.dim() < 4:
                x0_dec = x0_dec[None, :, :, :]
            min_freq_shape = min(x0.shape[2], x0_dec.shape[2])
            freq_mask = x0[:,:,:min_freq_shape] / (x0_dec[:,:,:min_freq_shape] + 1e-5)
            # print(freq_mask.max(), freq_mask.min())
            # print(freq_mask)
            with torch.no_grad():
                audio = ldm_stable.decode_to_mel(x0_dec)
                orig_audio = ldm_stable.decode_to_mel(x0)
        else:
            audio = x0_dec.detach().clone().cpu().squeeze(0)
            orig_audio = x0.detach().clone().cpu()
            x0_dec = get_spec(x0_dec, ldm_stable.get_fn_STFT())
            x0 = get_spec(x0.unsqueeze(0), ldm_stable.get_fn_STFT())

            if x0_dec.dim() < 4:
                x0_dec = x0_dec[None, :, :, :]
                x0 = x0[None, :, :, :]

    # same output
    current_GMT = time.gmtime()
    time_stamp_name = calendar.timegm(current_GMT)
    if args.mode == 'ours':
        image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
            f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
            f'skip_{"-".join([str(x) for x in skip.numpy()])}_{time_stamp_name}'
    else:
        if skip != 0:
            image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
                f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
                f'skip_{"-".join([str(x) for x in skip.numpy()])}_{time_stamp_name}'
        else:
            image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
                f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
                f'{args.num_diffusion_steps}timesteps_{time_stamp_name}'

    save_full_path_spec = os.path.join(save_path, image_name_png + ".png")
    save_full_path_wave = os.path.join(save_path, image_name_png + ".wav")
    save_full_path_origwave = os.path.join(save_path, "orig.wav")
    # Use a 3x3 average pooling (you can adjust kernel_size for more/less smoothing)
    # freq_mask = torch.nn.functional.avg_pool2d(freq_mask, kernel_size=3, stride=1, padding=1)
    # x0_dec = freq_mask
    # x0_dec = (x0_dec > 1).float()
    if x0_dec.shape[2] > x0_dec.shape[3]:
        x0_dec = x0_dec[0, 0].T.cpu().detach().numpy()
        x0 = x0[0, 0].T.cpu().detach().numpy()
    else:
        x0_dec = x0_dec[0, 0].cpu().detach().numpy()
        x0 = x0[0, 0].cpu().detach().numpy()
    plt.imsave(save_full_path_spec, x0_dec)
    torchaudio.save(save_full_path_wave, audio, sample_rate=sr)
    torchaudio.save(save_full_path_origwave, orig_audio, sample_rate=sr)
    
    #import numpy as np
    # Load the reference audio (tensor shape: [channels, samples])
    #ref_audio, sr = torchaudio.load(args.init_aud)
    #audio_channel = ref_audio[0]

    # Set parameters for the STFT/ISTFT.
    # n_fft = 1024            # FFT window size
    # hop_length = n_fft // 4  # hop length
    # # --- Extract magnitude and phase using STFT ---
    # # Using torch.stft with return_complex=True to obtain complex output.
    # stft_out = torch.stft(audio_channel, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    # # Extract magnitude and phase.
    # magnitude = stft_out.abs()
    # phase = stft_out.angle()
    # # To interpolate in both frequency and time dimensions, first add batch and channel dimensions.
    # # Expected shape for bilinear interpolation is [N, C, H, W]. Here, H corresponds to frequency, W to time.
    # mask_2d = freq_mask.transpose(2,3)  # Now shape: [1, 1, orig_freq, orig_time]
    # # Define target shape as the shape of our magnitude ([freq_bins, time_frames]).
    # target_shape = (magnitude.shape[0], magnitude.shape[1])
    # # Use bilinear interpolation to resize the mask.
    # mask_resized = torch.nn.functional.interpolate(
    #     mask_2d, size=target_shape, mode='bilinear', align_corners=False
    # )
    # # Remove the extra dimensions: now mask_resized has shape [freq_bins, time_frames].
    # mask_resized = mask_resized.squeeze(0).squeeze(0)
    # # Ensure the mask is on the same device as magnitude.
    # mask_resized = mask_resized.to(magnitude.device)
    
    # mask_resized = (mask_resized > 1.).float()
    # # mask_resized = mask_resized.clamp(0,5)
    # # ----- Apply the Mask on the Magnitude and Reconstruct the Audio -----
    # masked_magnitude = magnitude * mask_resized
    # # --- Convert back to audio ---
    # # Reconstruct the complex STFT from the masked magnitude and original phase.
    # reconstructed_stft = masked_magnitude * torch.exp(1j * phase)
    # reconstructed_audio = torch.istft(reconstructed_stft, n_fft=n_fft, hop_length=hop_length)


    # # ----- Define Parameters -----
    # n_fft = 1024            # FFT window size
    # hop_length = n_fft // 4  # hop length
    # n_mels = 64             # Number of mel bins

    # # ----- Compute Mel Spectrogram -----
    # mel_transform = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=sr,
    #     n_fft=n_fft,
    #     hop_length=hop_length,
    #     n_mels=n_mels
    # )
    # # Compute the mel spectrogram; shape: [n_mels, time_frames]
    # mel_spec = mel_transform(audio_channel)

    # # ----- Create a Dummy Mel Mask and Threshold It -----
    # # For demonstration, we create a dummy mask with the same frequency dimension (n_mels)
    # # but with a time dimension that is half the length of mel_spec.
    # orig_mel = mel_spec.shape[0]           # Expected to be n_mels (64)
    # orig_time = mel_spec.shape[1] // 2       # Half the number of time frames

    # # Clone the mask to allow in-place updates.
    # mel_mask = freq_mask.clone()
    # # # Apply threshold: values > 0.5 set to 1, else set to 0.
    # # mel_mask[mel_mask > 1] = 1
    # # mel_mask[mel_mask <= 0.5] = 0
    # mel_mask = (mel_mask>1).float()

    # # ----- Interpolate the Mel Mask to the Full Mel Spec Dimensions -----
    # # The bilinear interpolation function expects input in [N, C, H, W] format.
    # mask_2d = mel_mask.transpose(2,3) # New shape: [1, 1, orig_mel, orig_time]
    # target_shape = (mel_spec.shape[0], mel_spec.shape[1])  # Desired shape: [n_mels, time_frames]

    # mask_resized = torch.nn.functional.interpolate(
    #     mask_2d, size=target_shape, mode='bilinear', align_corners=False
    # )
    # # Remove extra dimensions so that mask_resized has shape: [n_mels, time_frames]
    # mask_resized = mask_resized.squeeze(0).squeeze(0).to(mel_spec.device)

    # # ----- Apply the Mask on the Mel Spectrogram -----
    # masked_mel_spec = mel_spec * mask_resized

    # # ----- Invert the Mel Spectrogram to Waveform -----
    # # First, convert the mel spectrogram back to a linear-scale spectrogram.
    # # Number of frequency bins in the linear spectrogram corresponding to n_fft is: n_stft = n_fft//2 + 1
    # n_stft = n_fft // 2 + 1
    # inv_mel_transform = torchaudio.transforms.InverseMelScale(
    #     n_stft=n_stft,
    #     n_mels=n_mels,
    #     sample_rate=sr
    # )
    # # Inverse transform from mel to linear spectrogram.
    # linear_spec = inv_mel_transform(masked_mel_spec)

    # # Reconstruction of the waveform from the linear spectrogram is not trivial.
    # # Here, we use the Griffin–Lim algorithm to approximate the inverse STFT.
    # griffin_lim = torchaudio.transforms.GriffinLim(
    #     n_fft=n_fft,
    #     hop_length=hop_length
    # )
    # reconstructed_audio = griffin_lim(linear_spec)



    # def normalize_audio(audio):
    #     """
    #     Normalize a tensor to the [-1, 1] range.
    #     Assumes audio is not already in that range.
    #     """
    #     max_val = audio.abs().max()
    #     if max_val > 0:
    #         return audio / max_val
    #     return audio

    # # normailze audio and orig_audio to [-1, 1]
    # audio_norm    = normalize_audio(audio)
    # orig_audio_norm = normalize_audio(orig_audio)
    # audio_norm    = audio
    # orig_audio_norm = orig_audio
    # ref_audio_norm  = normalize_audio(ref_audio)
    # # make them the same length and calculate the mask
    # min_length = min(audio_norm.shape[-1], orig_audio_norm.shape[-1])
    # audio_norm      = audio_norm[..., :min_length]
    # orig_audio_norm = orig_audio_norm[..., :min_length]
    # # apply mask on ref_audio
    # mask = audio_norm.abs() / (orig_audio_norm.abs() + 1e-5) 
    # mask = mask.clamp(-1, 1)
    # # reshape the mask as same length as ref_audio
    # # --- Step 4: Reshape/interpolate the mask to match ref_audio ---
    # if ref_audio_norm.shape[-1] != mask.shape[-1]:
    #     # Add batch and channel dimensions for interpolation: [N, C, L]
    #     mask = mask.unsqueeze(0)
    #     # Interpolate the mask using linear mode along time dimension
    #     mask = torch.nn.functional.interpolate(mask, size=ref_audio_norm.shape[-1], mode='linear', align_corners=False)
    #     # Remove extra dimensions
    #     mask = mask.squeeze(0)
    # reconstructed_audio = ref_audio * mask
    # reconstructed_audio = normalize_audio(reconstructed_audio)

    # torchaudio.save("reconstructed_audio.wav", reconstructed_audio.unsqueeze(0), sample_rate=sr)
    # L = min(reconstructed_audio.shape[0], ref_audio.shape[1])
    # sdr, sir, sar, _ = bss_eval_sources(
    # np.asarray(ref_audio[0,:L]),
    # np.asarray(reconstructed_audio[0, :L]),
    # False)
    # print(sdr, sir, sar)

    if not args.wandb_disable:
        logging_dict = {'orig': wandb.Audio(orig_audio.squeeze(), caption='orig', sample_rate=sr),
                        'orig_spec': wandb.Image(x0, caption='orig'),
                        'gen': wandb.Audio(audio[0].squeeze(), caption=image_name_png, sample_rate=sr),
                        'gen_spec': wandb.Image(x0_dec, caption=image_name_png)}
        wandb.log(logging_dict)

    wandb_run.finish()
