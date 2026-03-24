# E2E Sparse Neural Networks for Jet Classification

**GSoC 2025 — ML4SCI | CMS Experiment**

Self-supervised pretraining with Masked Autoencoders (MAE) on sparse CMS detector images, followed by fine-tuning for quark vs. gluon jet classification.

**Author:** Arjun Bhandary  
**Organization:** ML4SCI  
**Mentors:** Sergei Gleyzer, Emanuele Usai, Eric Reinhardt (University of Alabama)

---

## Problem

Jet classification (quark vs. gluon) is a key task in high-energy physics. CMS detector data is naturally **sparse** — only ~5–10% of the 125×125×8 image pixels contain non-zero energy deposits. Standard dense CNNs waste computation on empty regions. This project explores whether **sparse convolutions** and **sparse transformers** can exploit that sparsity for faster, more efficient classification, using self-supervised pretraining on 60k unlabelled jets followed by fine-tuning on 10k labelled samples.

## Approach

All models follow a two-phase pipeline:

**Phase 1 — Self-Supervised Pretraining (MAE):** 75% of active jet tokens are masked. The encoder processes the visible 25%, the decoder reconstructs the masked tokens, and the model learns rich representations without labels.

<p align="center">
  <img src="assets/pretraining_mae_pipeline.png" alt="Pretraining Pipeline" width="850"/>
</p>

**Phase 2 — Supervised Fine-Tuning:** The pretrained encoder is frozen for 5 epochs (head-only training), then unfrozen for end-to-end fine-tuning with differential learning rates.

<p align="center">
  <img src="assets/finetuning_pipeline.png" alt="Fine-tuning Pipeline" width="850"/>
</p>

## Results

| Model | AUC | Accuracy | F1 | 1/FPR @ TPR=0.7 | Best Epoch |
|:------|:---:|:--------:|:--:|:----------------:|:----------:|
| **Sparse ResNet MAE (recon only)** | **0.9609** | **0.904** | **0.908** | **27.4** | 9 |
| Sparse ResNet MAE (recon + occupancy) | 0.9566 | 0.890 | 0.894 | 22.6 | 9 |
| Sparse ViT MAE | 0.9426 | 0.878 | 0.881 | 15.4 | 15 |
| Sparse ResNet-SE MAE | 0.9420 | 0.876 | 0.881 | 17.8 | 6 |
| Sparse Autoencoder (L1 + KL) | 0.9341 | 0.869 | 0.871 | 14.1 | 20 |

The **Sparse ResNet with reconstruction-only MAE** achieves the best AUC of **0.9609**, slightly outperforming the variant with an additional occupancy prediction head. The ViT and SE-attention variants perform reasonably but don't surpass the simpler ResNet architecture on this dataset.

## Repository Structure

```
E2E/
├── EDA.ipynb                          # Exploratory data analysis
├── SparseConvolutions/
│   ├── ResNet_based/
│   │   ├── sparse_ResNet/             # Best model — ResNet MAE (recon only)
│   │   ├── sparse_ResNet_occupancy/   # ResNet MAE (recon + occupancy head)
│   │   └── sparse_ResNet_se/          # ResNet with squeeze-and-excitation
│   ├── ViT_based/                     # Sparse Vision Transformer MAE
│   └── SparseAutoencoder/             # Hybrid sparse autoencoder variant
├── SparseAutoencoder/
│   ├── Sparse_ResNet/                 # Sparse autoencoder with L1 + KL
│   └── dense_resnet_sae/              # Dense baseline autoencoder
├── assets/                            # Architecture diagrams
├── requirements.txt
└── install.sh
```

Each experiment directory contains `pretrain.py`, `finetune.py`, training history JSONs, and metric plots (loss curves, ROC curves, confusion matrices).

See the sub-directory READMEs for detailed architecture descriptions:
- [`SparseConvolutions/README.md`](SparseConvolutions/README.md) — Sparse convolution models (ResNet, ResNet-SE, ViT)
- [`SparseAutoencoder/README.md`](SparseAutoencoder/README.md) — Sparse autoencoder models (L1 + KL regularization)

## Dataset

| Split | Samples | Labels |
|:------|:-------:|:------:|
| Unlabelled (pretraining) | 60,000 | — |
| Labelled (fine-tuning) | 10,000 | Quark (0) / Gluon (1) |

Each sample is a **125 × 125 × 8** image from simulated pp collisions in the CMS detector. The 8 channels represent different detector sub-systems (ECAL, HCAL, tracks, etc.). Typical sparsity is ~90–95% (most pixels are zero).

## Setup

```bash
# Clone
git clone https://github.com/Arjun-bhandary/E2E.git
cd E2E

# Install (requires CUDA 12.x)
bash install.sh
```

**Requirements:** Python 3.10+, PyTorch 2.1+ (CUDA 12.1), spconv-cu120, scikit-learn, matplotlib.

## Key Findings

1. **Reconstruction-only MAE outperforms reconstruction + occupancy** — adding an occupancy prediction head during pretraining slightly hurts downstream classification (AUC 0.9566 vs. 0.9609), suggesting the model benefits from focusing entirely on feature reconstruction.

2. **Sparse convolutions > Sparse transformers** on this dataset — the ViT variant (AUC 0.9426) underperforms the ResNet (AUC 0.9609), likely because the 125×125 spatial structure is well-suited to local convolutional operations.

3. **SE-attention adds complexity without benefit** — Squeeze-and-Excitation blocks (AUC 0.9420) don't improve over the simpler ResNet, suggesting channel recalibration isn't critical for this particular feature space.

4. **Traditional sparse autoencoders underperform MAE** — the L1 + KL regularized autoencoder (AUC 0.9341) falls behind masked autoencoding, confirming that masking-based pretext tasks learn more transferable representations.

## References

- Graham & van der Maaten, [Submanifold Sparse Convolutional Networks](https://arxiv.org/abs/1706.01307), 2017
- Yang et al., [3D Object Detection with Sparse Convolutions](https://arxiv.org/abs/1712.07262), 2017
- Andrews et al., [End-to-End Jet Classification](https://arxiv.org/abs/1902.08276), 2019
- He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), 2021
- Reinhardt, GSoC 2023 (prior work)

## License

This project was developed as part of Google Summer of Code 2025 with ML4SCI.
