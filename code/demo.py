import sys
import torch
import torchaudio
import gradio as gr
import numpy as np
import tempfile
import moviepy.editor as mp

# --- Debug helper: write messages to debug.log and print to console ---
def write_debug(message):
    with open("debug.log", "a") as f:
        f.write(message + "\n")
    print(message)
    sys.stdout.flush()

write_debug("Script starting...")

# --- Import custom modules ---
try:
    from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
    from models import load_model
    from utils import set_reproducability, load_audio, get_spec
    write_debug("Custom modules imported successfully.")
except Exception as e:
    write_debug("Error importing custom modules: " + str(e))
    raise e

# --- Default constant values (matching your working command-line version) ---
DEFAULT_MODEL_ID = "cvssp/audioldm-s-full-v2"
DEFAULT_NUM_DIFFUSION_STEPS = 50   # e.g., as in your original working config
DEFAULT_SOURCE_GUIDANCE = 1.0         # Guidance for forward (inversion) process
DEFAULT_TARGET_GUIDANCE = 1.0       # Guidance for reverse (editing) process
DEFAULT_ETA = 1.0
DEFAULT_FIX_ALPHA = 0.1
DEFAULT_SEED = 42



def combine_audio_video(video_path, audio_path, output_path):
    """Combine processed audio with original video """
    try:
        # Load the video and audio clips
        video_clip = mp.VideoFileClip(video_path)
        audio_clip = mp.AudioFileClip(audio_path)
        
        # Ensure audio duration matches video duration
        if audio_clip.duration > video_clip.duration:
            # If audio is longer, trim it to match video duration
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        elif audio_clip.duration < video_clip.duration:
            # If video is longer, trim it to match audio duration
            video_clip = video_clip.subclip(0, audio_clip.duration)
        
        # Set the audio of the video clip
        final_clip = video_clip.set_audio(audio_clip)
        
        # Write the final video with proper codec settings
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=video_clip.fps,
            preset='medium',
            threads=4,
            ffmpeg_params=['-pix_fmt', 'yuv420p']
        )
        
        # Clean up
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
    except Exception as e:
        write_debug(f"Error in combine_audio_video: {str(e)}")
        raise e

def process_media(
    video_file, audio_file, target_prompt, source_prompt,
    model_id, tstart, seed, num_diffusion_steps, target_guidance, source_guidance
):
    try:
        write_debug("process_media called.")
        write_debug("Target prompt: " + target_prompt)
        write_debug("Source prompt: " + source_prompt)
        
        # Determine which file to use: if a video is provided, extract its audio;
        # otherwise, use the provided audio file.
        input_file = None
        original_video = None
        if video_file and video_file != "":
            write_debug("Video file provided: " + str(video_file))
            try:
                clip = mp.VideoFileClip(video_file)
                original_video = video_file
                temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_audio.close()
                # Extract audio using a sample rate of 16000.
                clip.audio.write_audiofile(temp_audio.name, fps=16000, logger=None)
                write_debug("Extracted audio from video: " + temp_audio.name)
                input_file = temp_audio.name
            except Exception as e:
                write_debug("Error extracting audio from video: " + str(e))
                raise e
        elif audio_file and audio_file != "":
            write_debug("Using provided audio file: " + str(audio_file))
            input_file = audio_file
        else:
            raise Exception("No video or audio file provided.")
        
        # Set reproducibility and device.
        write_debug("Setting reproducibility with seed " + str(seed))
        set_reproducability(seed)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        write_debug("Using device: " + device)
        
        # Load the diffusion model.
        write_debug("Loading model: " + model_id)
        ldm_stable = load_model(model_id, device, num_diffusion_steps)
        write_debug("Model loaded successfully.")
        
        # Load the audio from the chosen file.
        write_debug("Loading audio using file: " + input_file)
        x0, sr, duration = load_audio(
            input_file, 
            ldm_stable.get_fn_STFT(), 
            device=device,
            stft=True,
            model_sr=ldm_stable.get_sr()
        )
        write_debug(f"Audio loaded. Shape: {x0.shape}, Sample rate: {sr}, Duration: {duration}")
        
        with torch.inference_mode():
            write_debug("Encoding audio into latent space.")
            w0 = ldm_stable.vae_encode(x0)
            
            write_debug("Starting forward inversion process.")
            forward_prompt = [source_prompt] if source_prompt.strip() != "" else [""]
            wt, zs, wts, extra_info = inversion_forward_process(
                ldm_stable, w0, etas=DEFAULT_ETA,
                prompts=forward_prompt, cfg_scales=[source_guidance],
                prog_bar=False,
                num_inference_steps=num_diffusion_steps,
                cutoff_points=None,
                numerical_fix=True,
                duration=duration
            )
            write_debug("Forward inversion process complete.")
            
            tstart_tensor = torch.tensor([tstart], dtype=torch.int)
            skip = num_diffusion_steps - tstart  # with tstart=50 and steps=50, skip is 0.
            write_debug("tstart: " + str(tstart_tensor.item()) + ", skip steps: " + str(skip))
            
            write_debug("Starting separation process with target prompt.")
            w0_edited, _ = inversion_reverse_process(
                ldm_stable,
                xT=wts,
                tstart=tstart_tensor,
                fix_alpha=DEFAULT_FIX_ALPHA,
                etas=DEFAULT_ETA,
                prompts=[target_prompt],
                neg_prompts=[""],
                cfg_scales=[target_guidance],
                prog_bar=False,
                zs=zs[:int(num_diffusion_steps - skip)],
                cutoff_points=None,
                duration=duration,
                extra_info=extra_info
            )
            write_debug("Separation process complete.")
            
            write_debug("Decoding edited latent representation.")
            x0_dec = ldm_stable.vae_decode(w0_edited)
            if x0_dec.dim() < 4:
                x0_dec = x0_dec[None, :, :, :]
            with torch.no_grad():
                edited_audio_tensor = ldm_stable.decode_to_mel(x0_dec)
            write_debug("Decoding complete.")
        
        # Convert the edited audio tensor to a numpy array.
        audio_np = edited_audio_tensor.cpu().numpy()
        if audio_np.ndim == 2 and audio_np.shape[0] == 1:
            audio_np = audio_np.squeeze(0)
        audio_np = audio_np.astype(np.float32)
        audio_np = np.clip(audio_np, -1.0, 1.0)
        write_debug("process_media completed successfully. Audio shape: " + str(audio_np.shape))
        
        # Write the processed audio to a temporary WAV file.
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        if audio_np.ndim == 1:
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            audio_tensor = torch.from_numpy(audio_np)
        torchaudio.save(tmp_wav.name, audio_tensor, sample_rate=sr)
        write_debug("Temporary WAV file saved: " + tmp_wav.name)
        
        # If we had a video input, combine the processed audio with the video
        if original_video:
            tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_video.close()
            combine_audio_video(original_video, tmp_wav.name, tmp_video.name)
            return tmp_wav.name, tmp_video.name
        else:
            return tmp_wav.name, None
            
    except Exception as e:
        write_debug("Error in process_media: " + str(e))
        raise e

# --- Build the Gradio Interface ---
iface = gr.Interface(
    fn=process_media,
    inputs=[
        gr.Video(label="Input Video (optional)", height=240),
        gr.Audio(type="filepath", label="Input Audio (optional)"),
        gr.Textbox(lines=2, placeholder="Enter target prompt (e.g., 'man speech')", label="Target Prompt"),
        gr.Textbox(lines=2, placeholder="Enter source prompt (optional, can be empty)", label="Source Prompt"),
        gr.Dropdown(choices=[
            "cvssp/audioldm-s-full-v2", "cvssp/audioldm-l-full", "cvssp/audioldm2",
            "cvssp/audioldm2-large", "cvssp/audioldm2-music",
            "declare-lab/tango-full-ft-audio-music-caps", "declare-lab/tango-full-ft-audiocaps"], value=DEFAULT_MODEL_ID, label="Model ID"),
        gr.Number(value=DEFAULT_SEED, label="Seed"),
        gr.Slider(minimum=10, maximum=200, step=1, value=DEFAULT_NUM_DIFFUSION_STEPS, label="Number of Diffusion Steps"),
        gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=DEFAULT_TARGET_GUIDANCE, label="Target Guidance Scale"),
        gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=DEFAULT_SOURCE_GUIDANCE, label="Source Guidance Scale")
    ],
    outputs=[
        gr.Audio(type="filepath", label="Processed Audio"),
        gr.Video(label="Processed Video ", height=240)
    ],
    title="ZeroSep: Training-free Sound Separation with Video/Audio Upload",
    description=(
        "Upload either a video file (to extract its audio) or an audio file directly, and specify a target text prompt (and optionally a source prompt) "  
    ),
    allow_flagging="never"
)

write_debug("Launching Gradio interface...")
iface.launch(share=True)
