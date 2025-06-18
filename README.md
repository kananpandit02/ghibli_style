# ✨ Ghibli Style Transfer using CycleGAN 🎨
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cyclegan-app-6cvc3wgympvy9tskshngmf.streamlit.app/)



> 🧠 Transform real-world images into Studio Ghibli-style illustrations using a CycleGAN model built from scratch in PyTorch.  
> 🚀 Developed as a deep learning project by the **Fusion Eyes** team.

---

## 📌 Table of Contents

- [📖 Overview](#-overview)
- [🎯 Objectives](#-objectives)
- [🧠 Motivation](#-motivation)
- [📁 Dataset](#-dataset)
- [🛠 Preprocessing](#-preprocessing)
- [🧰 Model Architecture](#-model-architecture)
- [🧪 How to Run](#-how-to-run)
- [📦 Requirements](#-requirements)
- [📸 Results](#-results)
- [📊 Challenges & Discussion](#-challenges--discussion)
- [🔮 Future Work](#-future-work)
- [📄 Resources & Report](#-resources--report)
- [📬 Contact](#-contact)

---

## 📖 Overview

Studio Ghibli animations are renowned for their warm color tones, fantasy landscapes, and emotional depth. This project leverages CycleGAN to bring that hand-drawn magic to real-world images — all without paired data or pre-trained models.

**Project Title:** *Ghibli Style Transfer using CycleGAN*  
**Team Name:** *Fusion Eyes*  
**Team Members:**
- 👤 Kanan Pandit (B2430051)
- 👤 Partha Mete (B2430052)

---

## 🎯 Objectives

- 🧾 Transform real-world photographs into Studio Ghibli-style images.
- ♻️ Use **CycleGAN** for unpaired image-to-image translation.
- 🧱 Preserve semantic structures (e.g., faces, scenes).
- 💻 Implement the model **from scratch** for deeper understanding.

---

## 🧠 Motivation

> Why Ghibli?

Studio Ghibli films evoke nostalgia, warmth, and fantasy. Their distinct visual style is challenging for AI to replicate — making them perfect for testing the limits of deep learning in aesthetic transformation.

---

## 📁 Dataset

📂 **Dataset Name**: [Real to Ghibli Image Dataset (Kaggle)](https://www.kaggle.com/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images)  
🖼️ **Images**: 5,000  
🧾 **Unpaired** — no 1:1 correspondence

### 🔍 `trainA` – Real Domain:
- Human portraits, nature, cities, rivers, forests

### 🎨 `trainB` – Ghibli Domain:
- Fantasy villages, hand-drawn characters, magic landscapes

---

## 🛠 Preprocessing

All images are transformed as follows:

| Step              | Purpose                                    |
|-------------------|--------------------------------------------|
| Resize to 286×286 | Uniform dimensions                         |
| Random crop 256×256 | Variation in training patches            |
| Horizontal flip   | Augmentation                               |
| Normalize to [-1, 1] | GAN-friendly input scaling              |

Implemented using a custom `Dataset` class in PyTorch.

---

## 🧰 Model Architecture

### 🌀 CycleGAN Structure

| Component  | Role                                  |
|------------|----------------------------------------|
| `G_AB`     | Converts Real → Ghibli-style image     |
| `G_BA`     | Converts Ghibli → Real-world image     |
| `D_A`      | Discriminator for real-world images    |
| `D_B`      | Discriminator for Ghibli-style images  |

### 🧠 Generator (ResNet-based)

- Reflection padding
- 6 × Residual Blocks
- InstanceNorm, ReLU, Tanh

### 🛡️ Discriminator (PatchGAN)

- Focuses on `70x70` patches
- Preserves texture and style details

### 📦 Loss Functions

- **Adversarial Loss (MSELoss)**  
- **Cycle Consistency Loss (L1 Loss)**  
- **Identity Loss (L1 Loss)**

---

## 🧪 How to Run

### ✅ Google Colab (Recommended)

1. Open: [`FiNAL_CV_GHIBHLI_STYLE_TRANSFER.ipynb`](FiNAL_CV_GHIBHLI_STYLE_TRANSFER.ipynb)
2. Select **GPU Runtime**
3. Upload dataset and run cells

### 🖥️ Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/ghibli-style-transfer.git
cd ghibli-style-transfer

# Install dependencies
pip install -r requirements.txt

# Download dataset manually from Kaggle & place it in:
# ./dataset/trainA and ./dataset/trainB

# Run training
python train.py --dataset_dir ./dataset --epochs 50 --batch_size 1

####📊 Challenges & Discussion
Challenge	Notes
⚠️ GAN Instability	Oscillating loss values; mode collapse occasionally
🐌 Slow Convergence	Due to batch size = 1 (per CycleGAN recommendation)
⚙️ Hyperparameter tuning	Needed to stabilize training across 50 epochs

Despite challenges, meaningful stylization was achieved.

####🔮 Future Work
🧠 Add attention mechanisms (UGATIT, Self-Attn GAN)

🎨 Use perceptual loss (VGG-based)

🖼️ Train on higher-res datasets

💡 Create a Gradio or Streamlit app for real-time Ghibli-ization


