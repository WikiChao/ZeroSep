# ZeroSep: Separate Anything in Audio with Zero Training

[![ArXiv 2025](https://img.shields.io/badge/ArXiv-2025-red)](https://arxiv.org/abs/2505.23625) <img src="https://visitor-badge.laobi.icu/badge?page_id=WikiChao.ZeroSep" alt="访客统计" /> <img src="https://img.shields.io/github/stars/WikiChao/ZeroSep?style=social" alt="GitHub stars" /> <img alt="Static Badge" src="https://img.shields.io/badge/license-MIT%20-blue.svg" />
  
[**Project page**](https://wikichao.github.io/ZeroSep/) • [**Paper**](https://arxiv.org/pdf/2505.23625)

ZeroSep is a **training-free** audio source separation framework that repurposes pre-trained text-guided diffusion models for zero-shot separation.  
No fine-tuning, no task-specific data—just latent inversion + text-conditioned denoising to isolate **any** sound you describe.

<div align="center">
  <a href="https://www.youtube.com/watch?v=0t9nA1EUFrQ" target="_blank">
    <img src="https://img.youtube.com/vi/0t9nA1EUFrQ/0.jpg" alt="ZeroSep Demo" width="600">
  </a>
  <p><i>Demo video: ZeroSep separates speech from a mix with a simple text prompt.</i></p>
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
cd code
python demo.py
```

Then open [http://localhost:7860](http://localhost:7860) or the public link in your browser.
Upload an audio/video file, select your model & inversion strategy, enter a prompt (e.g. “dog bark”), and click **Run**.

### 2. Command-Line Interface

Separate a single audio file with one command:

```bash
cd code
python separate.py --input examples/BMayJId0X1s_120.wav --target "man speech"
```

#### Complete Example with All Parameters

```bash
python separate.py --input examples/BMayJId0X1s_120.wav \
                   --target "man speech" \
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

## 🎵 Examples

We've included several sample video/audio files in the `examples` folder to help you get started.

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

## 🙏 Acknowledgments

This work would not have been possible without the contributions of several outstanding projects:

* **AudioLDM** & **AudioLDM2** (Liu et al.) for providing the foundational diffusion model architectures
* **Tango** (Ghosal et al.) for their audio generation framework and model support
* **Gradio** team for their excellent interactive UI framework enabling our demo
* [**AudioEditingCode**](https://github.com/HilaManor/AudioEditingCode) by Manor et al. - Our implementation builds substantially upon their codebase. We sincerely appreciate their work and encourage supporting their repository.

### Licensing Information
* Code adapted from [**AudioEditingCode**](https://github.com/HilaManor/AudioEditingCode), including inversion and forward processes, is used under their MIT license.
* AudioLDM and AudioLDM2 models are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa]. Any use of these model weights is subject to the same license terms.
* All other original code in this repository is released under the MIT license.

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
