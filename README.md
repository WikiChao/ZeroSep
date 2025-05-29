# ZeroSep: Separate Anything in Audio with Zero Training

[![Arxiv 2025](https://img.shields.io/badge/Arxiv-2025-blue)](https://wikichao.github.io/ZeroSep/)  
[**Project page**](https://wikichao.github.io/ZeroSep/) • [**Paper**](https://wikichao.github.io/ZeroSep/)  

ZeroSep is a **training-free** audio source separation framework that repurposes pre-trained text-guided diffusion models for zero-shot separation.  
No fine-tuning, no task-specific data—just latent inversion + text-conditioned denoising to isolate **any** sound you describe.

<div align="center">
  <video width="100%" controls>
    <source src="assets/zerosep_demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <p><i>Demo video: ZeroSep separating speech from a music-speech mix with a simple text prompt.</i></p>
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

Then open [http://localhost:7860](http://localhost:7860) in your browser.
Upload an audio/video file, select your model & inversion strategy, enter a prompt (e.g. “dog bark”), and click **Run**.

### 2. Command-Line Interface

Separate a single audio file with one command:

```bash
python separate.py \
  --input path/to/mix.wav \
  --prompt "dog bark" \
  --model audioldm2-l \
  --inversion ddim \
  --output path/to/output.wav
```

### 3. Python API

Integrate ZeroSep into your own scripts:

```python
from zerosep import ZeroSep

zs = ZeroSep(
    model_name="audioldm2-l",
    inversion="ddim",
    guidance_weight=1.0
)

separated = zs.separate("path/to/mix.wav", prompt="dog bark")
zs.save(separated, "dog_bark.wav")
```

---

## ⚙️ Configuration

All default settings live in `config.yaml`:

```yaml
model: audioldm2-l
inversion: ddim
guidance_weight: 1.0
num_steps: 50
device: cuda
```

Override any option via CLI flags or by editing this file.

---

## 📁 Examples

In the `examples/` folder you’ll find sample mixtures and YAML configs:

```bash
python separate.py --config examples/music_example.yaml
```

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
