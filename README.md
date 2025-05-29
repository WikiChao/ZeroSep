# ZeroSep: Separate Anything in Audio with Zero Training

[![Arxiv 2025](https://img.shields.io/badge/Arxiv-2025-blue)](https://wikichao.github.io/ZeroSep/)  
[**Project page**](https://wikichao.github.io/ZeroSep/) • [**Paper**](https://wikichao.github.io/ZeroSep/)  

ZeroSep is a **training-free** audio source separation framework that repurposes pre-trained text-guided diffusion models for zero-shot separation.  
No fine-tuning, no task-specific data—just latent inversion + text-conditioned denoising to isolate **any** sound you describe.

<div align="center">
  [![Demo video – click to play](assets/thumb.png)](assets/demo_v2.mp4)
  <p><i>Demo video: ZeroSep separates speech from a music-speech mix with a simple text prompt.</i></p>
</div>

---

## 🚀 Features

- **Zero-shot separation**: separate without any additional training  
- **Open-set**: isolate arbitrary sounds via natural‐language prompts  
- **Model‐agnostic**: works with AudioLDM, AudioLDM2, Tango, or any text-guided diffusion backbone  
- Flexible inversion: choose **DDIM** or **DDPM**  
- Built-in **Gradio** demo for quick interactive use

---

## 📦 Installation

1. **Clone** this repo  
   ```bash
   git clone https://github.com/WikiChao/ZeroSep.git
   cd ZeroSep

2. *(Optional)* Create & activate a Conda environment

   ```bash
   conda create -n zerosep python=3.9
   conda activate zerosep
   ```
3. **Install** dependencies

   ```bash
   pip install -r requirements.txt
   ```
4. **(If using private Hugging Face models)**

   ```bash
   huggingface-cli login
   ```
5.**(If using tango)**
  download tango model from https://github.com/declare-lab/tango
  and put them into /code folder

---

## 🛠️ Usage

### 1. Gradio Web App

Launch the interactive demo:

```bash
python demo.py
```

Then open [http://localhost:7860](http://localhost:7860) or the public link in your browser.
Upload an audio/video file, select your model & inversion strategy, enter a prompt (e.g. “dog bark”), and click **Run**.

### 2. Command-Line Interface

Separate a single audio file with one command:

```bash
python separate.py --input examples/BMayJId0X1s_120.wav --target "male speech"
```

#### Complete Example with All Parameters

```bash
python separate.py --input examples/BMayJId0X1s_120.wav \
                   --target "male speech" \
                   --source "man talking with background music" \
                   --model "cvssp/audioldm-s-full-v2" \
                   --mode "ddpm" \
                   --steps 50 \
                   --tstart 50 \
                   --seed 42 \
                   --target_guidance 1.0 \
                   --source_guidance 1.0 \
                   --output_dir results \
                   --output_name "extracted_speech"
```

#### Parameter Reference

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `--input` | `-i` | Input audio file | Required |
| `--target` | `-t` | Sound to extract | Required |
| `--source` | `-s` | Source description | "" |
| `--model` | `-m` | Diffusion model | "cvssp/audioldm-s-full-v2" |
| `--mode` | | Separation algorithm | "ddpm" |
| `--steps` | | Diffusion steps | 50 |
| `--tstart` | | Start timestep | Same as steps |
| `--target_guidance` | | Target CFG scale | 1.0 |
| `--source_guidance` | | Source CFG scale | 1.0 |
| `--output_dir` | `-o` | Output directory | "results" |

---

## 📖 Citation

If you use ZeroSep, please cite our paper:

```bibtex
@inproceedings{huang2025zerosep,
  title        = {ZeroSep: Separate Anything in Audio with Zero Training},
  author       = {Huang, Chao and Ma, Yuesheng and Huang, Junxuan and Liang, Susan and Tang, Yunlong and Bi, Jing and Liu, Wenqiang and Mesgarani, Nima and Xu, Chenliang},
  booktitle    = {Arxiv},
  year         = {2025},
}
```

---

## 📜 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

* **AudioLDM** & **AudioLDM2** (Liu et al.) for the diffusion backbones
* **Tango** (Ghosal et al.) for additional model support
* **Gradio** for the user-friendly web UI
* [**AudioEditingCode**](https://github.com/HilaManor/AudioEditingCode) - Our work is heavily built upon their implementation. Please consider giving their repository a star as well!
