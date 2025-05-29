import argparse
import calendar
import matplotlib.pyplot as plt
import os
import time
import torch
import torchaudio
import warnings
import sys
import numpy as np
from torch import inference_mode
from pathlib import Path

from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from ddm_inversion.ddim_inversion import ddim_inversion, text2image_ldm_stable
from models import load_model
from utils import set_reproducability, load_audio, get_spec


# --- Debug helper: write messages to console ---
def write_debug(message):
    print(message)
    sys.stdout.flush()


def main():
    """Main function for command-line audio source separation"""
    # Define command-line arguments with more user-friendly descriptions and defaults
    parser = argparse.ArgumentParser(
        description='ZeroSep: Zero-Shot Audio Source Separation using Text Prompts.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help='Audio file to separate sources from')
    parser.add_argument("--output_dir", "-o", type=str, default="results", 
                        help='Directory to save results')
    parser.add_argument("--output_name", type=str, default=None,
                        help='Custom name for output file (defaults to automatic naming)')
                        
    # Prompt arguments
    parser.add_argument("--target", "-t", type=str, required=True,
                        help="Target prompt describing the sound you want to extract (e.g., 'drums', 'male speech')")
    parser.add_argument("--source", "-s", type=str, default="",
                        help="Source prompt describing the original mixture (optional)")
    
    # Model arguments
    parser.add_argument("--model", "-m", type=str, 
                        choices=["cvssp/audioldm-s-full-v2", "cvssp/audioldm-l-full",
                                 "cvssp/audioldm2", "cvssp/audioldm2-large",
                                 "cvssp/audioldm2-music", "declare-lab/tango-full-ft-audio-music-caps",
                                 "declare-lab/tango-full-ft-audiocaps"],
                        default="cvssp/audioldm-s-full-v2", 
                        help='Audio diffusion model to use')
    parser.add_argument("--mode", choices=['ddpm', 'ddim'], default="ddpm",
                        help="Separation mode: our DDPM approach (default) or DDIM inversion")
    
    # Diffusion process parameters
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--tstart", type=int, default=None,
                        help="Start timestep for reverse process (defaults to same as steps)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--target_guidance", type=float, default=1.0,
                        help='Classifier-free guidance strength for target prompt (reverse process)')
    parser.add_argument("--source_guidance", type=float, default=1.0,
                        help='Classifier-free guidance strength for source prompt (forward process)')
    
    # Device selection
    parser.add_argument("--device", type=int, default=0, 
                        help="GPU device number to use")
    
    # Advanced parameters (hidden from help by default)
    parser.add_argument("--fix_alpha", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--eta", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--wandb_disable", action="store_true", default=True, help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Set t_start to match steps if not specified
    if args.tstart is None:
        args.tstart = args.steps
    
    # Ensure tstart doesn't exceed steps
    args.tstart = min(args.tstart, args.steps)
    
    # Convert to format expected by processing functions
    tstart_tensor = torch.tensor([args.tstart], dtype=torch.int)
    skip = args.steps - args.tstart
    
    # Set up device
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    write_debug(f"Using device: {device}")
    
    # Set reproducability
    write_debug(f"Setting seed: {args.seed}")
    set_reproducability(args.seed, extreme=False)
    
    # Load model
    write_debug(f"Loading model: {args.model}")
    ldm_stable = load_model(args.model, device, args.steps)
    write_debug("Model loaded successfully.")
    
    # Load audio
    write_debug(f"Loading audio from: {args.input}")
    x0, sr, duration = load_audio(
        args.input, 
        ldm_stable.get_fn_STFT(), 
        device=device,
        stft=True, 
        model_sr=ldm_stable.get_sr()
    )
    write_debug(f"Audio loaded. Shape: {x0.shape}, Sample rate: {sr}, Duration: {duration}s")
    
    # Prepare prompts in the format expected by functions
    target_prompt = [args.target.strip()]
    source_prompt = [args.source.strip()] if args.source.strip() != "" else [""]
    
    # Process the audio
    with inference_mode():
        # Encode input audio to latent space
        write_debug("Encoding audio into latent space...")
        w0 = ldm_stable.vae_encode(x0)
        
        # Process based on mode
        if args.mode == "ddim":
            write_debug("Using DDIM inversion mode...")
            
            # Check for unsupported configurations in DDIM mode
            if skip != 0:
                warnings.warn("Plain DDIM Inversion should be run with t_start == steps. "
                            "You are now running partial DDIM inversion.", RuntimeWarning)
            
            # Forward DDIM inversion process
            write_debug("Starting DDIM inversion forward process...")
            wT = ddim_inversion(
                ldm_stable, w0, 
                source_prompt[0], args.source_guidance,
                num_inference_steps=args.steps, 
                skip=skip
            )
            write_debug("DDIM inversion forward process complete.")
            
            # Reverse DDIM generation process with target prompt
            write_debug("Starting DDIM text-to-image generation with target prompt...")
            w0_edited = text2image_ldm_stable(
                ldm_stable, target_prompt[0],
                args.steps, args.target_guidance,
                wT, skip=skip
            )
            write_debug("DDIM generation process complete.")
            
        else:  # DDPM mode
            write_debug("Using custom DDPM inversion mode...")
            
            # Forward process
            write_debug("Starting forward inversion process...")
            wt, zs, wts, extra_info = inversion_forward_process(
                ldm_stable, w0, 
                etas=args.eta,
                prompts=source_prompt, 
                cfg_scales=[args.source_guidance],
                prog_bar=True,
                num_inference_steps=args.steps,
                cutoff_points=None,
                numerical_fix=True,
                duration=duration
            )
            write_debug("Forward inversion process complete.")
            
            # Reverse process
            write_debug("Starting separation process with target prompt...")
            w0_edited, _ = inversion_reverse_process(
                ldm_stable,
                xT=wts,
                tstart=tstart_tensor,
                fix_alpha=args.fix_alpha,
                etas=args.eta,
                prompts=target_prompt,
                neg_prompts=[""],
                cfg_scales=[args.target_guidance],
                prog_bar=True,
                zs=zs[:int(args.steps - skip)],
                cutoff_points=None,
                duration=duration,
                extra_info=extra_info
            )
            write_debug("Separation process complete.")
        
        # Decode the edited latent representation
        write_debug("Decoding edited latent representation...")
        x0_dec = ldm_stable.vae_decode(w0_edited)
        if x0_dec.dim() < 4:
            x0_dec = x0_dec[None, :, :, :]
        with torch.no_grad():
            edited_audio_tensor = ldm_stable.decode_to_mel(x0_dec)
        write_debug("Decoding complete.")
    
    # Create output directory if it doesn't exist
    output_path = make_output_path(
        args.output_dir, 
        args.model, 
        args.input, 
        args.target, 
        args.source
    )
    os.makedirs(output_path, exist_ok=True)
    
    # Generate output filename
    if args.output_name:
        filename_base = args.output_name
    else:
        # Create a descriptive filename
        current_time = calendar.timegm(time.gmtime())
        filename_base = (
            f"{Path(args.input).stem}_{args.target.replace(' ', '_')}"
            f"_{args.mode}_g{args.target_guidance}_{current_time}"
        )
    
    # Save the output files
    spec_path = os.path.join(output_path, f"{filename_base}_spec.png")
    wav_path = os.path.join(output_path, f"{filename_base}.wav")
    orig_wav_path = os.path.join(output_path, "original.wav")
    params_path = os.path.join(output_path, f"{filename_base}_params.txt")
    
    # Save spectrogram visualization
    if x0_dec.shape[2] > x0_dec.shape[3]:
        x0_dec_vis = x0_dec[0, 0].T.cpu().detach().numpy()
        x0_vis = x0[0, 0].T.cpu().detach().numpy()
    else:
        x0_dec_vis = x0_dec[0, 0].cpu().detach().numpy()
        x0_vis = x0[0, 0].cpu().detach().numpy()
    plt.imsave(spec_path, x0_dec_vis)
    
    # Save audio files
    torchaudio.save(wav_path, edited_audio_tensor, sample_rate=sr)
    with torch.no_grad():
        original_audio = ldm_stable.decode_to_mel(x0)
    torchaudio.save(orig_wav_path, original_audio, sample_rate=sr)
    
    # Save parameters used for reproducibility
    save_parameters(params_path, vars(args))
    
    write_debug(f"\n✅ Separation complete!")
    write_debug(f"Output saved to: {output_path}")
    write_debug(f"Audio file: {wav_path}")
    write_debug(f"Original audio file: {orig_wav_path}")
    write_debug(f"Spectrogram visualization: {spec_path}")
    write_debug(f"Parameter log: {params_path}")


def make_output_path(base_dir, model, input_path, target, source):
    """Create a structured output path based on inputs"""
    # Extract model name without organization prefix
    model_name = model.split('/')[-1]
    
    # Get input file basename
    input_name = Path(input_path).stem
    
    # Create directory structure
    output_dir = os.path.join(
        base_dir,
        model_name,
        input_name,
        f"target_{target.replace(' ', '_')}"
    )
    
    # Add source to path if provided
    if source:
        output_dir = os.path.join(output_dir, f"source_{source.replace(' ', '_')}")
        
    return output_dir


def save_parameters(filepath, params):
    """Save parameters used for the separation to a text file"""
    with open(filepath, 'w') as f:
        f.write("ZeroSep Audio Separation Parameters\n")
        f.write("==================================\n\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nRun date: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
