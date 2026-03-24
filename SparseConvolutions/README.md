# Sparse Convolutions for Jet Classification

This directory contains all sparse convolution-based models: ResNet variants (with and without occupancy, with SE-attention) and a Sparse Vision Transformer.

---

## Sparse Tensor Representation

CMS jet images are 125×125×8, but ~90–95% of pixels are zero. Instead of storing the full dense grid, we represent each jet as a **sparse tensor** — only the N active (non-zero) pixels are stored, each with its spatial coordinates and 8-channel feature vector.

<p align="center">
  <img src="../assets/sparse_tensor_data_structure.png" alt="Sparse Tensor Data Structure" width="550"/>
</p>

This representation is the input to all sparse convolution layers via `spconv.SparseConvTensor`.

## Sparse Convolution Primitives

Three types of sparse convolution operations are used throughout:

### SubMConv2d — Submanifold Sparse Convolution

Computes convolution **only at existing active sites**, preserving the sparsity pattern exactly. The output has the same set of non-zero coordinates as the input.

<p align="center">
  <img src="../assets/submanifold_conv2d.png" alt="SubMConv2d" width="700"/>
</p>

### SparseConv2d — Regular Sparse Convolution (Downsampling)

A strided convolution that evaluates at **any output site whose kernel covers an active input site**. This changes the sparsity pattern and halves spatial dimensions (stride=2), used for downsampling between encoder stages.

<p align="center">
  <img src="../assets/regular_sparse_conv2d.png" alt="SparseConv2d" width="700"/>
</p>

### SparseInverseConv2d — Inverse Sparse Convolution (Upsampling)

Reverses a previous `SparseConv2d` by reusing its cached index mappings. Used in the MAE decoder to upsample back to the original spatial resolution.

<p align="center">
  <img src="../assets/sparse_inverse_conv2d.png" alt="SparseInverseConv2d" width="700"/>
</p>

---

## Architectures

### SparseResBlock

The basic building block: two 3×3 `SubMConv2d` layers with BatchNorm and ReLU, plus a 1×1 skip connection when channel dimensions change (pre-activation ResNet style).

<p align="center">
  <img src="../assets/sparse_resblock.png" alt="SparseResBlock" width="750"/>
</p>

### Sparse ResNet Encoder

A 4-stage encoder with 2 `SparseResBlock`s per stage, interleaved with `SparseConv2d` downsamplers (stride 2). Channel progression: 8 → 64 → 128 → 256 → 512.

<p align="center">
  <img src="../assets/sparse_resnet_encoder.png" alt="Sparse ResNet Encoder" width="800"/>
</p>

---

## Experiments

### ResNet_based/

| Variant | Pretraining | AUC | Accuracy | F1 | 1/FPR @ 0.7 |
|:--------|:------------|:---:|:--------:|:--:|:-----------:|
| `sparse_ResNet/` | MAE (reconstruction only) | **0.9609** | **0.904** | **0.908** | **27.4** |
| `sparse_ResNet_occupancy/` | MAE (recon + occupancy) | 0.9566 | 0.890 | 0.894 | 22.6 |
| `sparse_ResNet_se/` | MAE + SE blocks (3 blocks/stage) | 0.9420 | 0.876 | 0.881 | 17.8 |

**`sparse_ResNet/`** — The best-performing model. Pretrained with reconstruction-only MAE (75% masking, MSE loss on masked tokens). No occupancy head during pretraining. Uses the standard 2-block-per-stage encoder.

**`sparse_ResNet_occupancy/`** — Adds an occupancy prediction head during pretraining: the decoder predicts both the feature values of masked tokens AND a binary mask of which spatial locations are active. The occupancy loss is weighted at 0.5× relative to reconstruction. This ablation shows the occupancy head slightly hurts downstream performance (ΔAUC = −0.0043).

**`sparse_ResNet_se/`** — Adds Squeeze-and-Excitation (SE) channel attention to each residual block and increases to 3 blocks per stage. Despite more parameters, it underperforms the simpler architecture (ΔAUC = −0.0189 vs. baseline).

### ViT_based/

| Variant | AUC | Accuracy | F1 | 1/FPR @ 0.7 |
|:--------|:---:|:--------:|:--:|:-----------:|
| Sparse ViT (6L, 256dim, 8 heads) | 0.9426 | 0.878 | 0.881 | 15.4 |

Treats each active pixel as a token, adds 2D sinusoidal positional encoding, and processes through a 6-layer transformer encoder with a CLS token. Uses a lightweight MLP decoder (not a transformer decoder) for speed. Tokens are capped at 1024 per sample.

The ViT underperforms the ResNet variants, likely because the 125×125 spatial structure benefits more from local convolutional inductive biases than global attention.

---

## Training Details

All models share the same training recipe:

**Pretraining:**
- 60,000 unlabelled samples, 80/20 train-val split
- AdamW optimizer, lr=1e-3, weight_decay=1e-4
- Cosine annealing to 1e-6, early stopping (patience=7–10)
- 75% masking ratio, MSE reconstruction loss

**Fine-tuning:**
- 10,000 labelled samples, 80/20 train-val split
- Phase 1 (epochs 1–5): Encoder frozen, head lr=1e-3
- Phase 2 (epochs 6+): Full model, encoder lr=5e-5, head lr=1e-3
- Dropout 0.5, weight_decay=1e-3, early stopping (patience=6)
- BCEWithLogitsLoss

## File Structure

Each experiment directory contains:

```
├── pretrain.py              # Self-supervised pretraining script
├── finetune.py              # Supervised fine-tuning script
├── pretrain_history.json    # Per-epoch pretraining metrics
├── finetune_history.json    # Per-epoch fine-tuning metrics
├── finetune_metrics.json    # Final evaluation summary
└── *.jpg                    # Training plots (loss, AUC, ROC, confusion matrix)
```
