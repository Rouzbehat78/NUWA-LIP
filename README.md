
# AR IMAGE INPAINTING WITH GUIDED TEXT: VQ-GANS + CLIP + MAMBA
This repository implements a two-stage, discrete-latent inpainting system:

1. **Defect-Aware VQGAN**  
   Trains a VQGAN to ignore masked regions during encoding, producing “clean” discrete tokens for known image patches.

2. **Text-Conditioned Mamba AR Decoder**  
   Adapts the Mamba state-space sequence model to autoregressively predict missing tokens given (a) CLIP ViT-B/32 text embeddings (projected + AdaLN) and (b) the surrounding decoded tokens.  
   Classifier-free guidance is used at generation time to bias outputs toward the prompt.

This design avoids iterative diffusion, focuses compute on masked areas only, and scales linearly with token sequence length—enabling faster, high-resolution inpainting.

##  Key Features

- **Defect-Free Tokenization**  
  Mask-aware  DF- VQGAN encoder that zeroes out masked pixels in convolutions & attention, and normalizes over unmasked features only.
- **Discrete-Latent Inpainting**  
  Autoregressive filling of missing latent tokens in raster order; unmasked tokens are preserved via teacher forcing.
- **Multimodal Conditioning**  
  CLIP ViT-B/32 embeddings injected as a “prefix token” and via adaptive layer normalization (AdaLN) across Mamba layers.
- **Classifier-Free Guidance**  
  Interpolates between conditional and unconditional decoders at inference (guidance = 1.5) for stronger prompt adherence.
- **Modular & Extensible**  
  Swap in alternate tokenizers, conditioning methods, or AR decoders with minimal changes.
- **Mamba Decoder Image Generation**
  Using Mamba decoder image generation for linear sequence lenght computation complexity. Adopted from AiM pre-trained Mamba Baseline.


---

## 🗂 Repository Layout

```
NUWA‑LIP/
├── aim/                          # Mamba‐related code and notebooks
│   ├── __init__.py
│   ├── aim.py                    # Top‐level Mamba model class
│   ├── block.py                  # State‐space / attention blocks
│   ├── generation.py             # AR sampling & CFG logic
│   ├── mixer_seq_simple.py       # Simplified sequence mixer
│   └── notebooks/
│       └── mamba_training.ipynb  # Experiments & prototyping
│
├── config/            # Model configurations & dataset definitions
│   ├─ mps4cc          # Our Mamba based model for inpainting using the NUWA-LIP DF-VQGAN
│   └─ dfvqgan.py      # DF‑VQGAN module & codebook interfaces
├── collector/         # Hooks for inference & training collectors
├── wutils.py          # Utilities: I/O, trainers, checkpointing, SFTP, etc.
├── finetune.py        # Entry point for training & evaluation
├── run.sh             # Platform‑agnostic launch script
└── README.md          
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
