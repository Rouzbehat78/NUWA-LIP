
# NUWA‑LIP 🚀  
_A Next-Generation Framework for Language-Guided Image Inpainting and Generation_

---

## 🌟 Overview

**NUWA‑LIP** is a cutting-edge, multi-modal framework designed for masked image-text modeling and conditional image generation.  
It seamlessly integrates large language-vision models (like CLIP), discrete visual token models (like DF-VQGAN), and advanced transformer architectures (like Megatron-LR) to deliver high-quality, scalable, and customizable visual generation pipelines.

Whether you’re working on text-conditioned inpainting, spatially masked reconstruction, or fine-tuned creative generation on custom datasets, NUWA-LIP provides an efficient, modular, and research-ready foundation.

---

## 🔑 Core Contributions

✅ **Unified Masked Image-Text Framework**  
→ Supports both Vision-Masked Language Modeling (VMLM) and standard Masked Language Modeling (MLM).

✅ **Advanced Visual Tokenization**  
→ Uses DF‑VQGAN to discretize visual spaces, making transformer modeling over images efficient.

✅ **Scalable, Modular Training**  
→ Built-in support for single-node and distributed PyTorch DDP; easy-to-extend configs and hooks.

✅ **Custom Dataset Integration**  
→ Pre-configured pipelines for MSCOCO, Conceptual Captions, and easy extension to your own datasets.

✅ **Plug-and-Play Decoder Options**  
→ Flexible second-stage decoders (MP-S2S, Mamba, MaskGIT, etc.) for balancing quality vs. inference speed.

✅ **Rich Utility Layer**  
→ I/O, logging, checkpoints, LMDB management, SFTP syncing, and custom data handling all built-in.

---

## 📦 Repository Structure

```
NUWA-LIP/
├── aim/                  # Mamba-based fast transformer modules
├── collector/            # MP-S2S inference/training components
├── config/               # Configurations for various datasets and pipelines
├── wutils.py             # Utilities for logging, file ops, and data loading
├── finetune.py           # Fine-tuning pipeline
├── run.sh                # Sample shell script for execution
├── README.md             # This file!
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Rouzbehat78/NUWA-LIP.git
cd NUWA-LIP
pip install -r requirements.txt
```

Optional (for GPU acceleration):
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Usage

### Train or Fine-Tune
```bash
bash run.sh --config config/lip4maskcoco/base.py --train
```

### Run Inference
```bash
python collector/mps_inference.py --input your_input_image.jpg --prompt "A sunset over the mountains"
```

### Replace Decoder with Faster Mamba Variant
```bash
python aim/mamba_training.py --config config/your_config.py
```

---

## 📊 Results

| **Model Variant**         | **Min Val Loss ↓** | **Final Loss ↓** | **Latency (ms) ↓** |
|---------------------------|---------------------|-------------------|---------------------|
| NUWA-LIP Original         | N/A                 | N/A               | 204.7               |
| NUWA-LIP w/ Mamba (Ours)  | **0.89**           | 0.91              | **73.9**            |
| Ablation – No Refinement  | 2.47                | 2.50              | **0.0**             |

> 💡 **Key takeaway:** Our Mamba-based decoder achieves a ~3x speedup with minimal performance degradation.

---

## 🤝 Contributions
TODO
