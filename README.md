
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

| Step               | Purpose                          |
|--------------------|----------------------------------|
| Resize to 286×286  | Uniform dimensions               |
| Random crop 256×256| Introduce variation              |
| Horizontal flip    | Data augmentation                |
| Normalize to [-1, 1]| GAN-friendly input scaling       |

---

## 🧰 Model Architecture

### 🌀 CycleGAN Structure

| Component | Role                                 |
|-----------|---------------------------------------|
| `G_AB`    | Generator: Real → Ghibli              |
| `G_BA`    | Generator: Ghibli → Real              |
| `D_A`     | Discriminator: Real domain            |
| `D_B`     | Discriminator: Ghibli domain          |

### 🧠 Generator (ResNet-based)
- Reflection padding  
- 6 × Residual Blocks  
- InstanceNorm, ReLU, Tanh

### 🛡️ Discriminator (PatchGAN)
- Operates on 70×70 patches  
- Preserves local texture details

### 📦 Loss Functions
- **Adversarial Loss (MSELoss)**  
- **Cycle Consistency Loss (L1)**  
- **Identity Loss (L1)**

---

## 🧪 How to Run

### ✅ Google Colab (Recommended)

1. Open the notebook: [`FiNAL_CV_GHIBHLI_STYLE_TRANSFER.ipynb`](FiNAL_CV_GHIBHLI_STYLE_TRANSFER.ipynb)  
2. Set runtime to **GPU**  
3. Upload the dataset and run all cells

---

### 🖥️ Local Setup (Optional)

```bash
git clone https://github.com/kananpandit02/ghibli-style-transfer.git
cd ghibli-style-transfer
pip install -r requirements.txt
python train.py --dataset_dir ./dataset --epochs 50 --batch_size 1
```

---

## 📦 Requirements

```
torch
torchvision
numpy
Pillow
matplotlib
tqdm
```

---

## 📸 Results

> Real → Ghibli style image transformation

- Identity preserved using identity loss
- Texture and style successfully transferred
- Results improve as training progresses

| Input (Real) | Ghibli Output | Reconstructed |
|--------------|---------------|----------------|
| ![](samples/input.jpg) | ![](samples/output.jpg) | ![](samples/recon.jpg) |

---

## 📊 Challenges & Discussion

| Challenge | Notes |
|----------|-------|
| ⚠️ GAN Instability | Oscillating losses and mode collapse observed occasionally |
| 🐌 Slow Convergence | Used batch size = 1 (as recommended for CycleGAN) |
| ⚙️ Hyperparameter tuning | Required manual tuning for loss balance and stability |

Despite these challenges, the model achieved high-quality artistic translation.

---

## 🔮 Future Work

- 🧠 Add attention mechanisms (UGATIT, Self-Attention)  
- 🎨 Introduce perceptual loss (VGG-based)  
- 🖼️ Train with higher-resolution images  
- 🌐 Extend or enhance the existing [Streamlit app](https://cyclegan-app-6cvc3wgympvy9tskshngmf.streamlit.app/) for more interactivity or batch processing

---

## 📄 Resources & Report

- 📘 Report: [`CV_PROJECT_FINAL_REPORT_FUSION_EYES_.pdf`](CV_PROJECT_FINAL_REPORT_FUSION_EYES_.pdf)
- 📓 Notebook: `FiNAL_CV_GHIBHLI_STYLE_TRANSFER.ipynb`
- 📚 [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- 🔗 [Official CycleGAN GitHub](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

---

## 📬 Contact

- 📧 kananpandot02@gmail.com
- 🧑‍💻 GitHub: [kananpandit02](https://github.com/kananpandit02/)

---

## 📢 Citation

```
@misc{fusioneyes2025ghibli,
  author = {Kanan Pandit and Partha Mete},
  title = {Ghibli Style Transfer using CycleGAN},
  year = {2025},
  url = {https://github.com/kananpandit02/ghibli_style}
}
```

⭐️ If you found this helpful, don’t forget to **star** the repository!
