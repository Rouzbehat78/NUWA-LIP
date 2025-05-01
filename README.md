# NUWA‑LIP 🚀

**NUWA‑LIP** is a state‑of‑the‑art, multi‑modal framework for masked image‑text modeling and conditional image generation.  
Powered by CLIP, VQGAN, and Megatron‑LR, it supports a suite of pretraining and finetuning pipelines on MSCOCO, Conceptual Captions, custom datasets, and more.

---

## 🔥 Key Features

- **Masked Vision‑Language Modeling**  
  ‣ Generate and apply spatial/textual masks with `MaskingGenerator`  
  ‣ Support for VMLM (_Vision-Masked Language Modeling_) & MLM  
- **Conditional Image Generation**  
  ‣ Leverage a pretrained painter network _(mps4coco / lip4maskcoco / lip4custom)_  
  ‣ Seamless integration with DF‑VQGAN for discrete visual tokens  
- **Scalable Training & Inference**  
  ‣ Single‑node or distributed via PyTorch DDP / Megatron  
  ‣ Customizable pipelines: `run.sh`, `finetune.py`, `collector/` hooks  
- **Utility Arsenal**  
  ‣ `wutils.py` for I/O, logging, checkpoints, LMDB, SFTP, JSON/TSV handling  
  ‣ Built‑in DataLoaderX, Trainer, Meter, and more  

---

## 🗂 Repository Layout

```
NUWA‑LIP/
├── config/            # Model configurations & dataset definitions
│   ├─ mps4coco/       # MVLM + conditional painter
│   ├─ lip4maskcoco/   # Lip‑based masked modeling on COCO
│   ├─ lip4custom/     # Custom downstream tasks
│   └─ dfvqgan.py      # DF‑VQGAN module & codebook interfaces
├── collector/         # Hooks for inference & training collectors
├── wutils.py          # Utilities: I/O, trainers, checkpointing, SFTP, etc.
├── finetune.py        # Entry point for training & evaluation
├── run.sh             # Platform‑agnostic launch script
└── README.md          
```

---

## ⚙️ Installation

1. Clone this repo:  
   ```bash
   git clone https://github.com/your-org/NUWA-LIP.git
   cd NUWA-LIP
   ```
2. Create & activate your environment (Conda/Pipenv/venv):  
   ```bash
   pip install -r requirements.txt
   ```
3. Download pretrained CLIP & VQGAN weights:  
   ```bash
   # ensure CLIP/VQGAN files under $ROOT/CLIP and $ROOT/checkpoint/DF‑VQGAN/
   bash scripts/download_pretrained.sh
   ```

---

## 🚀 Quick Start

### 1. Pretraining on MSCOCO (MVLM)
```bash
bash run.sh \
  --config config/mps4coco/base.py \
  --action train \
  --platform local \
  --train_local_batch_size 4 \
  --eval_local_batch_size 1
```

### 2. Finetuning for Masked Text Generation
```bash
python finetune.py \
  --config config/lip4maskcoco/base.py \
  --do_train \
  --dist False \
  --local_train_batch_size 8
```

### 3. Inference & Visualization
```bash
python finetune.py \
  --config config/lip4custom/base.py \
  --do_eval_visu \
  --visu_split val
```

---

## 📖 Configuration

All hyperparameters and file paths are exposed via `Args` classes in `config/*/base.py`.  
Easily override:
```python
args = Args()
args.learning_rate = 3e-4
args.epochs = 1000
```

---

## ⭐️ Citation

If you use **NUWA‑LIP** in your research, please cite:

```
@inproceedings{NUWA-LIP2023,
  title={NUWA‑LIP: Unified Large‑scale Image-Text Pretraining},
  author={Your Name and Collaborators},
  year={2023},
  booktitle={Conference/Journal}
}
```
