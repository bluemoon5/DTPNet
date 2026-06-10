# DTPNet: Temporal Difference Prompt Fusion Network

> **Remote Sensing Change Detection via Vision Foundation Model Adaptation**

[![Paper](https://img.shields.io/badge/Paper-Master%20Thesis-blue)](./assets/thesis.pdf)
[![Framework](https://img.shields.io/badge/Framework-OpenCD%20%2B%20MMSeg-orange)]()
[![Backbone](https://img.shields.io/badge/Backbone-DINOv2%20ViT--B%2F14-green)]()

Official implementation of my Master's thesis *"Research on Remote Sensing Image Change Detection Based on Multi-Feature Fusion"* (基于多特征融合的遥感图像变化检测方法研究).

**DTPNet** achieves **SOTA binary change detection** on LEVIR-CD (F1=92.34%, IoU=85.77%) and S2Looking (F1=65.83%, IoU=49.06%), using only **40% of the parameters** of comparable large-model methods.

**DTPNet-SCD** extends the framework to **semantic change detection**, achieving mIoU=74.89% on SECOND and 82.89% on Landsat-SCD.

---

## Key Innovations

| Module | Problem Solved | Approach |
|--------|---------------|----------|
| **DLP** (Domain-specific Learnable Prompt) | Domain gap between natural images (DINOv2 pre-training) and remote sensing | Freeze DINOv2, insert learnable prompt tokens into each Transformer layer to guide feature adaptation |
| **TDA T** (Temporal Difference Attention Transformer) | Insufficient modeling of subtle temporal differences | Dual-branch differential attention with λ-reparameterization for explicit difference modeling |
| **Decoupled SCD** | C×C class space explosion in semantic change detection | Decompose joint prediction into change detection + dual-temporal semantic segmentation |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    DTPNet Pipeline                    │
├─────────────────────────────────────────────────────┤
│                                                       │
│   T1 Image ──► DINOv2 + DLP ──┐                       │
│                                ├──► TDA T ──► BAN     │
│   T2 Image ──► DINOv2 + DLP ──┘    (Diff       Mask  │
│    (frozen)    (learnable         Attention)  Decoder)│
│               prompts)                                │
│                                          ↓            │
│                                    Change Map         │
└─────────────────────────────────────────────────────┘
```

- **Encoder**: Frozen DINOv2 ViT-B/14 + Per-layer learnable prompt tokens (DLP)
- **Decoder**: 3-layer Temporal Difference Attention Transformer (TDA T) with SwiGLU FFN
- **Head**: BAN BIT Head for binary/semantic mask prediction
- **Only 41.21M parameters** vs ChangeCLIP (185.60M) and BAN (45.60M)

---

## Results

### Binary Change Detection (Chapter 3)

| Method | LEVIR-CD F1↑ | LEVIR-CD IoU↑ | S2Looking F1↑ | S2Looking IoU↑ | Params↓ |
|--------|:------------:|:-------------:|:-------------:|:-------------:|:-------:|
| FC-EF | 69.37 | 53.11 | 7.65 | - | 1.35M |
| BIT-CD | 84.41 | 73.02 | 63.76 | 46.80 | 3.55M |
| ChangeFormer | 87.98 | 78.54 | 63.02 | 45.99 | 31.20M |
| ChangeCLIP | 90.54 | 82.72 | 63.87 | 46.96 | 185.60M |
| RemoteCLIP | 90.89 | 83.30 | 64.12 | 47.18 | 178.40M |
| BAN | 91.50 | 84.33 | 65.17 | 48.35 | 45.60M |
| **DTPNet (ours)** | **92.34** | **85.77** | **65.83** | **49.06** | **41.21M** |

### Semantic Change Detection (Chapter 4)

| Method | SECOND mIoU↑ | SECOND Fscd↑ | Landsat-SCD mIoU↑ | Landsat-SCD Fscd↑ |
|--------|:------------:|:------------:|:-----------------:|:-----------------:|
| Bi-SRNet | 68.32 | 58.45 | 78.12 | 69.87 |
| SCanNet | 72.15 | 63.28 | 80.34 | 72.15 |
| **DTPNet-SCD (ours)** | **74.89** | **66.12** | **82.89** | **75.34** |

### Ablation Study

| DLP | TDA T | LEVIR-CD F1 | LEVIR-CD IoU | S2Looking F1 | S2Looking IoU |
|:---:|:-----:|:-----------:|:------------:|:------------:|:-------------:|
| ✗ | ✗ | 91.85 | 84.93 | 65.44 | 48.63 |
| ✓ | ✗ | 91.95 | 85.10 | 65.48 | 48.67 |
| ✓ | ✓ | **92.34** | **85.77** | **65.83** | **49.06** |

---

## Project Structure

```
DTPNet/
├── README.md              # This file
├── models/
│   ├── dinov2_backbone.py # DINOv2 ViT + DLP prompt embeddings
│   ├── tdat_diff_transformer.py  # TDA T decoder (MultiHeadDifferentialAttention)
│   ├── ban_head.py        # BAN mask decoder head
│   └── segmentor.py       # DualSiamEncoderDecoder segmentor
├── configs/
│   ├── dtpnet_levircd.py       # DTPNet training config (LEVIR-CD)
│   └── dtpnet_scd_levircd.py   # DTPNet-SCD training config
├── train.py               # Distributed training entrypoint
├── test.py                # Distributed testing entrypoint
└── assets/
    └── DTPNet_architecture.png # Model architecture diagram (Fig 3.1)
```

---

## Quick Start

### Requirements

- PyTorch 2.0+
- MMSegmentation + OpenCD
- 1× NVIDIA GPU with ≥24GB VRAM (A10/3090/4090)
- Ubuntu 20.04 (tested)

### Installation

```bash
pip install torch==2.0.1 torchvision==0.15.2
pip install mmcv-full mmsegmentation
git clone https://github.com/likyoo/open-cd.git
cd open-cd && pip install -e .
```

### Training

```bash
# Binary change detection on LEVIR-CD
bash dist_train.sh configs/dtpnet_levircd.py 2

# Semantic change detection on SECOND
bash dist_train.sh configs/dtpnet_scd_levircd.py 2
```

### Inference

```bash
python test.py configs/dtpnet_levircd.py work_dirs/dtpnet_levircd/iter_40000.pth
```

> **Note:** DTPNet uses frozen DINOv2 ViT-B/14 with per-layer learnable prompts. Full inference requires ~20GB VRAM. The pretrained DINOv2 checkpoint is auto-downloaded from Meta.

---

## Design Decisions

**Why freeze DINOv2?**
- Full fine-tuning of 86M ViT parameters + domain gap → overfitting on limited remote sensing data
- DLP achieves domain adaptation with only ~0.06% additional trainable parameters

**Why differential attention instead of cross-attention?**
- Cross-attention mixes features between time steps, losing temporal identity
- Differential attention preserves per-timestep representations while explicitly modeling change

**Why decouple SCD instead of C×C joint prediction?**
- C×C class space suffers from combinatorial explosion (e.g., 7 land cover classes → 49 change classes)
- ~80% of class combinations have zero or near-zero samples (long-tail distribution)
- Decoupling into change detection + semantic segmentation avoids this issue entirely

---

## Citation

```bibtex
@mastersthesis{liu2026dtpnet,
  title     = {Research on Remote Sensing Image Change Detection Based on Multi-Feature Fusion},
  author    = {Yuanlong Liu},
  school    = {Dalian Minzu University},
  year      = {2026},
  type      = {Master's Thesis},
  advisor   = {Associate Professor Mingliang Xue}
}
```

---

## License

This project is released under the MIT License.

## Acknowledgments

- [OpenCD](https://github.com/likyoo/open-cd) - Change detection toolbox based on MMSegmentation
- [DINOv2](https://github.com/facebookresearch/dinov2) - Self-supervised vision transformer by Meta AI
- [BAN](https://github.com/likyoo/BAN) - Bi-temporal Adapter Network baseline
