# E2E Sparse Neural Networks for Jet Classification

**GSoC 2026 — ML4SCI**

Self-supervised pretraining with Masked Autoencoders (MAE) on sparse detector images, followed by fine-tuning for binary jet classification.

**Author:** Arjun Bhandary

---

## Problem

Jet classification is a key task in high-energy physics. CMS detector data is naturally **sparse** — only ~5–10% of the 125×125×8 image pixels contain non-zero energy deposits. Standard dense CNNs waste computation on empty regions. This project explores whether **sparse convolutions** and **sparse transformers** can exploit that sparsity for faster, more efficient classification, using self-supervised pretraining on 60k unlabelled jets followed by fine-tuning on 10k labelled samples.

<p align="center">
  <img src="assets/sparse_tensor_visualization.png" alt="Sparse Tensor Visualization — CMS jet image showing ~90-95% sparsity" width="700"/>
  <br/>
  <em>A CMS jet image as a sparse tensor: only the coloured pixels carry non-zero energy deposits across 8 detector channels. The vast majority of the 125×125 grid is empty.</em>
</p>

## Approach

All models follow a two-phase pipeline:

### Phase 1 — Self-Supervised Pretraining (MAE)

75% of active jet tokens are masked. The encoder processes the visible 25%, the decoder reconstructs the masked tokens, and the model learns rich representations without labels.

<p align="center">
  <img src="assets/pretraining_mae_pipeline.png" alt="Pretraining Pipeline" width="850"/>
</p>

### Phase 2 — Supervised Fine-Tuning

The pretrained encoder is frozen for 5 epochs (head-only training), then unfrozen for end-to-end fine-tuning with differential learning rates.

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

The **Sparse ResNet with reconstruction-only MAE** achieves the best AUC of **0.9609**, outperforming all other variants including the occupancy-augmented model, the ViT, and the traditional sparse autoencoder.

<p align="center">
  <img src="https://raw.githubusercontent.com/Arjun-bhandary/E2E/main/SparseConvolutions/ResNet_based/sparse_ResNet/roc_curve.jpg" alt="ROC Curve — Sparse ResNet MAE (best model, AUC = 0.9609)" width="550"/>
  <br/>
  <em>ROC curve for the best model (Sparse ResNet MAE, reconstruction only). AUC = 0.9609.</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Arjun-bhandary/E2E/main/SparseConvolutions/ResNet_based/sparse_ResNet/confusion_matrix.jpg" alt="Confusion Matrix — Sparse ResNet MAE" width="450"/>
  <br/>
  <em>Confusion matrix on the validation set (Sparse ResNet MAE, reconstruction only).</em>
</p>

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
├── pruning/
│   ├── sparse_resnet/                 # Pruning analysis — Sparse ResNet MAE
│   ├── sparse_vit/                    # Pruning analysis — Sparse ViT MAE
│   └── sparse_ae/                     # Pruning analysis — Sparse Autoencoder
├── assets/                            # Architecture diagrams and result plots
├── requirements.txt
└── install.sh
```

Each experiment directory contains `pretrain.py`, `finetune.py`, training history JSONs, and metric plots (loss curves, ROC curves, confusion matrices).

See the sub-directory READMEs for detailed architecture descriptions:
- [`SparseConvolutions/README.md`](SparseConvolutions/README.md) — Sparse convolution models (ResNet, ResNet-SE, ViT)
- [`SparseAutoencoder/README.md`](SparseAutoencoder/README.md) — Sparse autoencoder models (L1 + KL regularization)
- [`pruning/README.md`](pruning/README.md) — Post-training pruning analysis and Error vs. FLOPS trade-offs

## Dataset

| Split | Samples | Labels |
|:------|:-------:|:------:|
| Unlabelled (pretraining) | 60,000 | — |
| Labelled (fine-tuning) | 10,000 | Quark (0) / Gluon (1) |

Each sample is a **125 × 125 × 8** image from simulated pp collisions in the CMS detector. The 8 channels represent different detector sub-systems (ECAL, HCAL, tracks, etc.). Typical sparsity is ~90–95% (most pixels are zero).

## Compute Infrastructure

All experiments were run on a shared university DGX-1 server accessed via SSH.

| Resource | Details |
|:---------|:--------|
| **GPU** | NVIDIA Tesla V100-SXM2 — 32 GB HBM2, 5120 CUDA cores, 640 Tensor Cores |
| **Server** | DGX-1 (8× V100-SXM2-32GB, NVLink interconnect) |
| **CUDA Version** | 12.0 |
| **Driver** | NVIDIA 525.60.13 |
| **PyTorch** | 2.1.0+cu121 |
| **Sparse Backend** | spconv-cu120 v2.3.6 |

### Why spconv instead of MinkowskiEngine?

[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) is a popular sparse convolution library, but its pre-built pip wheels are compiled for **CUDA ≤ 11.x** (10.2, 11.1, 11.3). Building from source on CUDA 12.x requires manual patches to the thrust/CUB API and specific GCC versions — the project has not been actively maintained since 2023 ([see issues](https://github.com/NVIDIA/MinkowskiEngine/issues/543)). This made MinkowskiEngine unusable on:

- **Our university server** — CUDA 12.0 (driver 525.60.13)
- **Kaggle notebooks** — CUDA 12.1+

[spconv](https://github.com/traveller59/spconv) (v2.x) provides equivalent sparse convolution primitives (`SubMConv2d`, `SparseConv2d`, `SparseInverseConv2d`) with pre-built wheels for CUDA 12.0 (`spconv-cu120`), active maintenance, and tensor core support. All models in this repository use spconv as the sparse convolution backend.

## Setup

```bash
# Clone
git clone https://github.com/Arjun-bhandary/E2E.git
cd E2E

# Install (requires CUDA 12.x, Python 3.10+)
bash install.sh
```

The install script sets up PyTorch 2.1.0 (CUDA 12.1), spconv-cu120, and all other dependencies. See `requirements.txt` for the full list.

## Post-Training Pruning

After fine-tuning, we apply **global unstructured magnitude pruning** (`torch.nn.utils.prune.L1Unstructured`) to three models — Sparse ResNet MAE, Sparse ViT MAE, and Sparse Autoencoder — at 11 sparsity levels (0% to 95%). This measures how much each model can be compressed before classification quality degrades, and how FLOPS scale with weight sparsity.

Pruning is applied globally across all convolutional (`SubMConv2d`, `SparseConv2d`) and linear layers, ranking all weights by absolute magnitude and zeroing the smallest ones. The pruning masks are then made permanent (weights are actually set to zero, not just masked).

<p align="center">
  <img src="assets/error_vs_flops_all_models.png" alt="Error vs FLOPS — all pruned models" width="700"/>
  <br/>
  <em>Error (1 − Accuracy) vs. estimated GFLOPS at different pruning ratios for all three models.</em>
</p>

**FLOPS estimation** accounts for the dual sparsity in these models: sparse convolutions already operate only on active sites (not the full 125×125 grid), and pruning further reduces computation proportionally to the fraction of non-zero weights remaining. See [`pruning/README.md`](pruning/README.md) for detailed results, per-model Error vs. FLOPS plots, and the FLOPS estimation methodology.

