# WorldCrafter: Dynamic Scene Generation from a Single Image with Geometric and Temporal Consistency

This repository provides the anonymous implementation of **WorldCrafter**, a framework designed for dynamic scene generation from a single image and a text prompt. The method supports various controllable camera motions (e.g., pan, zoom) while preserving both geometric and temporal consistency in the generated videos.

---

## 🔧 Installation

We recommend using `conda` for environment setup:

```
conda create -n worldcrafter python=3.10 -y
conda activate worldcrafter
pip install -r requirements.txt
```

---

## 🚀 Usage

The framework supports different camera motion types. You can either run the provided shell scripts or call the Python scripts directly.

### 🔹 Shell Scripts

```
bash run_panleftright.sh              # Pan left-right
bash run_panupdown.sh                 # Pan up-down
bash run_zoominout.sh                 # Zoom in/out
```

### 🔹 Python Scripts

```
python test_video_panleftright.py \
  --input_path path/to/image.jpg \
  --prompt "A fantasy landscape with waterfalls"
```

Ablation studies can be conducted using the `_wo_videodepth.py` or `_wo_videoinpaint.py` versions.

---

## 📂 Directory Structure

```
world_crafter/
├── .gitignore                          # Git ignore file
├── run_panleftright.sh                # Pan left-right generation script
├── run_panleftright_wo_videodepth.sh  # Ablation: without video depth
├── run_panleftright_wo_videoinpaint.sh# Ablation: without inpainting
├── run_panupdown.sh                   # Pan up-down generation script
├── run_zoominout.sh                   # Zoom in/out generation script
├── test_video_panleftright.py         # Main testing script
├── test_video_panleftright_wo_videodepth.py
├── test_video_panleftright_wo_videoinpaint.py
├── test_video_panupdown.py
├── test_video_zoominout.py
├── models/                            # Model definitions and wrappers
├── scene/                             # Scene rendering, depth, inpainting modules
├── syncdiffusion/                     # Diffusion-based video generation
├── util/                              # Utility functions
├── utils/                             # Additional helper functions
```

---

## 📊 Evaluation

We evaluate our method using the **VBench** protocol and CLIP-based metrics. The following aspects are measured:

- **Subject Consistency**: Measures appearance stability of the main subject across frames.
- **Background Consistency**: Evaluates temporal coherence of the background.
- **Temporal Stability**: Captures motion smoothness across video frames.
- **CLIP Similarity**: Assesses semantic alignment between video frames and input prompts.

User study results and additional quantitative metrics are reported in the supplementary material.

---

## 📄 License

This code is released for **academic research use only**. 