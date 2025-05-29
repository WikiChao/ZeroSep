import sys
import torch
import torchaudio
import gradio as gr
import numpy as np
import tempfile
import moviepy.editor as mp
import warnings
import os

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
    from ddm_inversion.ddim_inversion import ddim_inversion, text2image_ldm_stable
    from models import load_model
    from utils import set_reproducability, load_audio, get_spec
    write_debug("Custom modules imported successfully.")
except Exception as e:
    write_debug("Error importing custom modules: " + str(e))
    raise e

# --- Default constant values (matching your working command-line version) ---
DEFAULT_MODEL_ID = "cvssp/audioldm-s-full-v2"
DEFAULT_NUM_DIFFUSION_STEPS = 50   # e.g., as in your original working config
DEFAULT_SOURCE_GUIDANCE = 1.0      # Guidance for forward (inversion) process
DEFAULT_TARGET_GUIDANCE = 1.0      # Guidance for reverse (editing) process
DEFAULT_ETA = 1.0
DEFAULT_FIX_ALPHA = 0.1
DEFAULT_SEED = 42

# --- CSS for styling the interface ---
custom_css = """
.container {
    margin: 0 auto;
    max-width: 1200px;
}
.header-container {
    text-align: center;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 2px solid rgba(0, 0, 0, 0.1);
}
.model-params {
    padding: 15px;
    border-radius: 10px;
    background: rgba(0, 0, 0, 0.03);
    margin-top: 10px;
}
.footer {
    text-align: center;
    margin-top: 20px;
    font-size: 0.8em;
    color: #666;
    padding: 10px;
}
.prompt-box {
    background: rgba(0, 144, 255, 0.05);
    border-radius: 8px;
    padding: 5px;
    margin-bottom: 10px;
}
.processing-btn {
    background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
    color: white;
    border-radius: 5px;
}
.examples-table {
    width: 100%;
    border-collapse: collapse;
}
.examples-table th, .examples-table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
.examples-table th {
    background-color: #f2f2f2;
}
.examples-table tr:nth-child(even) {
    background-color: #f9f9f9;
}
.output-container {
    background: rgba(0, 0, 0, 0.02);
    border-radius: 10px;
    padding: 15px;
    min-height: 300px;
}
.mode-selector {
    text-align: center;
    margin: 15px 0;
}
"""

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
    model_id, tstart, seed, num_diffusion_steps, target_guidance, source_guidance,
    mode="ddpm"
):
    try:
        write_debug("process_media called.")
        write_debug(f"Mode: {mode}")
        write_debug("Target prompt: " + target_prompt)
        write_debug("Source prompt: " + source_prompt)
        
        # Ensure tstart doesn't exceed num_diffusion_steps
        tstart = min(tstart, num_diffusion_steps)
        write_debug(f"Using tstart: {tstart} with {num_diffusion_steps} diffusion steps")
        
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
            raise gr.Error("Please provide either a video or an audio file.")
        
        if not target_prompt or target_prompt.strip() == "":
            raise gr.Error("Please enter a target prompt describing the sound you want to extract.")
        
        # Set reproducibility and device.
        write_debug("Setting reproducibility with seed " + str(seed))
        set_reproducability(seed)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        write_debug("Using device: " + device)
        
        # Create progress tracker
        progress = gr.Progress(track_tqdm=True)
        progress(0, desc="Preparing model...")
        
        # Load the diffusion model.
        write_debug("Loading model: " + model_id)
        ldm_stable = load_model(model_id, device, num_diffusion_steps)
        write_debug("Model loaded successfully.")
        
        # Load the audio from the chosen file.
        progress(0.1, desc="Loading audio...")
        write_debug("Loading audio using file: " + input_file)
        x0, sr, duration = load_audio(
            input_file, 
            ldm_stable.get_fn_STFT(), 
            device=device,
            stft=True,
            model_sr=ldm_stable.get_sr()
        )
        write_debug(f"Audio loaded. Shape: {x0.shape}, Sample rate: {sr}, Duration: {duration}")
        
        tstart_tensor = torch.tensor([tstart], dtype=torch.int)
        skip = num_diffusion_steps - tstart  # with tstart=50 and steps=50, skip is 0.
        write_debug("tstart: " + str(tstart_tensor.item()) + ", skip steps: " + str(skip))
        
        progress(0.2, desc="Processing audio...")
        with torch.inference_mode():
            write_debug("Encoding audio into latent space.")
            w0 = ldm_stable.vae_encode(x0)
            
            # Process based on mode
            if mode == "ddim":
                write_debug("Using DDIM inversion mode.")
                
                # Check for unsupported configurations in DDIM mode
                if skip != 0:
                    warnings.warn("Plain DDIM Inversion should be run with t_start == num_diffusion_steps. "
                                "You are now running partial DDIM inversion.", RuntimeWarning)
                
                forward_prompt = source_prompt.strip() if source_prompt.strip() != "" else ""
                target_prompt_text = target_prompt.strip()
                
                # Forward DDIM inversion process
                progress(0.3, desc="DDIM inversion forward process...")
                write_debug("Starting DDIM inversion forward process.")
                wT = ddim_inversion(ldm_stable, w0, forward_prompt, source_guidance,
                                   num_inference_steps=num_diffusion_steps, skip=skip)
                write_debug("DDIM inversion forward process complete.")
                
                # Reverse DDIM generation process with target prompt
                progress(0.6, desc="DDIM generation with target prompt...")
                write_debug("Starting DDIM text-to-image generation with target prompt.")
                w0_edited = text2image_ldm_stable(ldm_stable, target_prompt_text,
                                                num_diffusion_steps, target_guidance,
                                                wT, skip=skip)
                write_debug("DDIM generation process complete.")
                
            else:  # DDPM mode
                write_debug("Using custom DDPM inversion mode.")
                
                forward_prompt = [source_prompt] if source_prompt.strip() != "" else [""]
                write_debug("Starting forward inversion process.")
                
                progress(0.3, desc="DDPM forward inversion process...")
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
                
                progress(0.6, desc="DDPM separation with target prompt...")
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
            
            progress(0.8, desc="Decoding audio...")
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
        progress(0.9, desc="Saving results...")
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
            progress(1.0, desc="Complete!")
            return tmp_wav.name, tmp_video.name, "✅ Processing complete!"
        else:
            progress(1.0, desc="Complete!")
            return tmp_wav.name, None, "✅ Processing complete!"
            
    except Exception as e:
        write_debug("Error in process_media: " + str(e))
        return None, None, f"❌ Error: {str(e)}"

def get_example_file_path(filename):
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(os.path.dirname(script_dir), "examples")
    fp = os.path.join(examples_dir, filename)
    if os.path.isfile(fp):
        write_debug(f"Found example file: {fp}")
        return fp
    else:
        write_debug(f"Warning: Example file not found: {fp}")
        return None   # ← return None, not ""

# Add this function after your existing functions
def load_with_demo_audio(target, source, model, tstart, steps, t_guidance, s_guidance, mode, demo_file):
    """Load example configuration with a pre-selected demo audio file"""
    return [
        None,       # No video
        demo_file,  # Pre-selected audio file
        target, source, model, tstart, DEFAULT_SEED, 
        steps, t_guidance, s_guidance, mode
    ]

# Example data for demonstration - using relative paths
audio_examples = [
    # Example 1: Extract speech from background noise
    [get_example_file_path("BMayJId0X1s_120.wav"), 
     "man speech", "", 
     "cvssp/audioldm-s-full-v2", 50, DEFAULT_SEED, 50, 1.0, 1.0, "ddpm"],

]

video_examples = [
    # Example 1: Extract speech from background noise
    [get_example_file_path("Barack Obama's speech to graduates.mp4"), 
     "man speech", "", 
     "cvssp/audioldm-s-full-v2", 50, DEFAULT_SEED, 50, 1.0, 1.0, "ddpm"],
]

# Function to update tstart maximum based on num_diffusion_steps
def update_tstart_slider(num_steps):
    return gr.Slider(minimum=0, maximum=num_steps, step=1, value=num_steps, label="Start Timestep (tstart)")

# Function to update UI based on selected mode
def update_mode_ui(mode):
    if mode == "ddim":
        return [
            gr.update(value="For best DDIM results, set t_start equal to diffusion steps.", visible=True),
            gr.update(value=DEFAULT_NUM_DIFFUSION_STEPS)  # Set tstart to full steps for DDIM
        ]
    else:
        return [
            gr.update(value="", visible=False),
            gr.update()  # Keep current tstart value for DDPM
        ]

def get_recommended_steps(model_id):
    if "audioldm2" in model_id or "tango" in model_id:
        return 200
    elif "audioldm" in model_id:
        return 100
    else:
        return 50

def update_steps_for_model(model_id):
    recommended_steps = get_recommended_steps(model_id)
    return gr.update(value=recommended_steps), gr.update(maximum=recommended_steps, value=recommended_steps)

# --- Build the Gradio Interface ---
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="container"):
        # Header
        with gr.Column(elem_classes="header-container"):
            gr.Markdown("# 🎵 ZeroSep: Training-free Sound Separation")
            gr.Markdown(
                "Extract specific sounds from audio or video using text prompts. "
                "Powered by diffusion models."
            )
        
        # Main content - removed the tabs structure since we only have one tab now
        # Input column
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                with gr.Group():
                    gr.Markdown("Upload either a video or an audio file:")
                    video_input = gr.Video(label="Video Input (optional)", height=240)
                    audio_input = gr.Audio(type="filepath", label="Audio Input (optional)")
                
                with gr.Group(elem_classes="prompt-box"):
                    gr.Markdown("### 📝 Prompts")
                    target_prompt = gr.Textbox(
                        lines=2, 
                        placeholder="Enter target prompt (e.g., 'male speech', 'drums', 'piano music')", 
                        label="Target Sound to Extract (Required)",
                        info="Describe the sound you want to extract"
                    )
                    source_prompt = gr.Textbox(
                        lines=2, 
                        placeholder="Enter source prompt (e.g., 'speech with background music')", 
                        label="Source Description (Optional)",
                        info="Leave empty, defaults to empty string. Can help guide the model but is not required."
                    )
                
                with gr.Column(elem_classes="mode-selector"):
                    mode = gr.Radio(
                        choices=["ddpm", "ddim"], 
                        value="ddpm", 
                        label="Separation Mode",
                        info="Choose DDPM for default separation or DDIM for alternative method"
                    )
                    mode_warning = gr.Markdown(visible=False)
                
                process_button = gr.Button("🚀 Process Audio/Video", elem_classes="processing-btn")
                
            # Output column
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                with gr.Group(elem_classes="output-container"):
                    status_text = gr.Markdown("Upload an audio or video file and click Process")
                    audio_output = gr.Audio(type="filepath", label="Processed Audio", show_download_button=True)
                    video_output = gr.Video(label="Processed Video", height=240, visible=True)
        
        # Advanced options in an accordion
        with gr.Accordion("Advanced Options", open=False):
            with gr.Column(elem_classes="model-params"):
                gr.Markdown("### ⚙️ Model Parameters")
                with gr.Row():
                    with gr.Column(scale=1):
                        model_id = gr.Dropdown(
                            choices=[
                                "cvssp/audioldm-s-full-v2", 
                                "cvssp/audioldm-l-full", 
                                "cvssp/audioldm2",
                                "cvssp/audioldm2-large", 
                                "cvssp/audioldm2-music",
                                "declare-lab/tango-full-ft-audio-music-caps", 
                                "declare-lab/tango-full-ft-audiocaps"
                            ], 
                            value=DEFAULT_MODEL_ID, 
                            label="Model Type",
                            info="AudioLDM family models are recommended for best results"
                        )
                        seed = gr.Number(value=DEFAULT_SEED, label="Random Seed", precision=0)
                    
                    with gr.Column(scale=1):
                        num_diffusion_steps = gr.Slider(
                            minimum=10, maximum=200, step=1, 
                            value=DEFAULT_NUM_DIFFUSION_STEPS, 
                            label="Number of Diffusion Steps",
                            info="Higher values = better quality but slower"
                        )
                        tstart = gr.Slider(
                            minimum=0, maximum=DEFAULT_NUM_DIFFUSION_STEPS, 
                            step=1, value=DEFAULT_NUM_DIFFUSION_STEPS, 
                            label="Start Timestep (tstart)",
                            info="Controls separation strength (better to keep it equal to diffusion steps)"
                        )
                
                with gr.Row():
                    with gr.Column():
                        target_guidance = gr.Slider(
                            minimum=0.0, maximum=15.0, step=0.1, 
                            value=DEFAULT_TARGET_GUIDANCE, 
                            label="Target Guidance Scale",
                            info="1.0 is enough for most cases"
                        )
                    with gr.Column():
                        source_guidance = gr.Slider(
                            minimum=0.0, maximum=15.0, step=0.1, 
                            value=DEFAULT_SOURCE_GUIDANCE, 
                            label="Source Guidance Scale",
                            info="1.0 is enough for most cases"
                        )

        # Footer
        with gr.Row(elem_classes="footer"):
            gr.Markdown("📄 ZeroSep: Zero-shot Audio Source Separation using Text Prompts")
        
        # Event handlers - keep these
        num_diffusion_steps.change(
            fn=update_tstart_slider,
            inputs=num_diffusion_steps,
            outputs=tstart
        )
        
        mode.change(
            fn=update_mode_ui,
            inputs=mode,
            outputs=[mode_warning, tstart]
        )
        
        model_id.change(
            fn=update_steps_for_model,
            inputs=model_id,
            outputs=[num_diffusion_steps, tstart]
        )
        
        process_button.click(
            fn=process_media,
            inputs=[
                video_input, audio_input, target_prompt, source_prompt,
                model_id, tstart, seed, num_diffusion_steps, 
                target_guidance, source_guidance, mode
            ],
            outputs=[audio_output, video_output, status_text]
        )

    # Audio-only examples block
    gr.Examples(
        examples=audio_examples,
        fn=process_media,
        inputs=[
            audio_input,        # only the audio widget
            target_prompt, source_prompt,
            model_id, tstart, seed,
            num_diffusion_steps,
            target_guidance, source_guidance,
            mode
        ],
        outputs=[audio_output, video_output, status_text],
        label="🎯 Audio-only demos",
        examples_per_page=1,
        run_on_click=True,
    )

    # Video-only examples block
    gr.Examples(
        examples=video_examples,
        fn=process_media,
        inputs=[
            video_input,        # only the video widget
            target_prompt, source_prompt,
            model_id, tstart, seed,
            num_diffusion_steps,
            target_guidance, source_guidance,
            mode
        ],
        outputs=[audio_output, video_output, status_text],
        label="🎬 Video-only demos",
        examples_per_page=1,
        run_on_click=True,
    )

write_debug("Launching Gradio interface...")
demo.launch(share=True)
