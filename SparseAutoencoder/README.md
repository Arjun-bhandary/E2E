# Sparse Autoencoder for Jet Classification

This directory explores an alternative to Masked Autoencoding: a traditional **sparse autoencoder** with L1 sparsity and KL divergence regularization on the latent space.

---

## Approach

Instead of masking tokens and reconstructing them (MAE), the sparse autoencoder:

1. **Encodes the full jet image** (no masking) into a latent representation
2. **Decodes** back to reconstruct the original
3. Adds **L1 sparsity** and **KL divergence** penalties on the latent activations to encourage a sparse, distributed internal representation

<p align="center">
  <img src="../assets/sparse_autoencoder_pipeline.png" alt="Sparse Autoencoder Pipeline" width="800"/>
  <br/>
  <em>Sparse Autoencoder: the full input is encoded (no masking), reconstructed, and regularized with L1 + KL penalties on the latent activations.</em>
</p>

### Loss Function

The total pretraining loss combines three terms:

```
L_total = L_recon (MSE) + λ_L1 · ||z||₁ + λ_KL · KL(ρ || ρ̂)
```

| Term | Purpose | Typical magnitude (epoch 1) |
|:-----|:--------|:---------------------------:|
| **L_recon (MSE)** | Faithful reconstruction of input features | ~0.014 |
| **λ_L1 · ‖z‖₁** | Encourages sparse latent activations (few active units) | ~1e-6 |
| **λ_KL · KL(ρ ‖ ρ̂)** | Pushes average activation ρ̂ toward target sparsity ρ | ~4e-6 |

Where **ρ** is a target sparsity level (e.g., 0.05 — meaning each latent unit should be active ~5% of the time) and **ρ̂** is the observed average activation of each latent unit across the batch.

### Architecture

The encoder and decoder use the **same Sparse ResNet architecture** as the convolution-based models (see [`SparseConvolutions/README.md`](../SparseConvolutions/README.md)), but the pretraining objective is different:

```
                   MAE (SparseConvolutions/)          Sparse Autoencoder (this directory)
                   ──────────────────────────         ────────────────────────────────────
Input:             25% of active tokens (75% masked)  100% of active tokens (no masking)
Pretext task:      Reconstruct masked tokens          Reconstruct all tokens
Regularization:    None (implicit via masking)         L1 + KL on latent activations
Loss:              MSE on masked positions only        MSE on all positions + L1 + KL
```

```
Encoder (identical to Sparse ResNet encoder)
┌─────────────────────────────────────────────────────────────────────┐
│ Input: SparseConvTensor (N × 8, spatial 125×125)                    │
│   ▼ SubMConv2d(8 → 64, 3×3) + BN + ReLU                            │
│   ▼ Stage 1: 2× ResBlock(64)                                       │
│   ▼ SparseConv2d(64 → 128, stride=2)   → 63×63                     │
│   ▼ Stage 2: 2× ResBlock(128)                                      │
│   ▼ SparseConv2d(128 → 256, stride=2)  → 32×32                     │
│   ▼ Stage 3: 2× ResBlock(256)                                      │
│   ▼ SparseConv2d(256 → 512, stride=2)  → 16×16                     │
│   ▼ Stage 4: 2× ResBlock(512)                                      │
│   ▼                                                                 │
│   Latent z (N' × 512)  ← L1 and KL penalties applied here          │
└─────────────────────────────────────────────────────────────────────┘

Decoder (mirror of encoder using SparseInverseConv2d)
┌─────────────────────────────────────────────────────────────────────┐
│   ▼ SparseInverseConv2d(512 → 256)  → 32×32                        │
│   ▼ 2× ResBlock(256)                                               │
│   ▼ SparseInverseConv2d(256 → 128)  → 63×63                        │
│   ▼ 2× ResBlock(128)                                               │
│   ▼ SparseInverseConv2d(128 → 64)   → 125×125                      │
│   ▼ 2× ResBlock(64)                                                │
│   ▼ SubMConv2d(64 → 8, 1×1)                                        │
│   ▼                                                                 │
│   Reconstructed output (125×125 × 8)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Experiments

### Sparse_ResNet/

Uses the Sparse ResNet encoder pretrained with the autoencoder objective (L1 + KL) instead of MAE.

| Metric | Value |
|:-------|:-----:|
| **AUC** | 0.9341 |
| **Accuracy** | 0.869 |
| **F1** | 0.871 |
| **1/FPR @ TPR=0.7** | 14.1 |

This is the weakest-performing model in the comparison (**ΔAUC = −0.0268** vs. the best MAE model), suggesting that masking-based pretext tasks learn more transferable representations than reconstruction with sparsity regularization.

### dense_resnet_sae/

A **dense (non-sparse) ResNet autoencoder** baseline for comparison. Uses standard `nn.Conv2d` instead of `spconv.SubMConv2d`, operating on the full 125×125×8 grid including zero pixels.

```
Dense Autoencoder (baseline)
┌─────────────────────────────────────────────────────┐
│ Input: Dense tensor (B × 8 × 125 × 125)             │
│   ▼ nn.Conv2d(8 → 64, 3×3) + BN + ReLU              │
│   ▼ Standard ResBlocks + MaxPool2d (stride=2)        │
│   ▼ ...                                              │
│   Latent → Decoder (ConvTranspose2d for upsampling)  │
│   ▼ Reconstructed output (B × 8 × 125 × 125)        │
└─────────────────────────────────────────────────────┘

Key difference: processes ALL 15,625 pixels per sample,
including the ~90% that are zero — wasting computation.
```

---

## Key Observation

The sparse autoencoder's L1 and KL losses are **orders of magnitude smaller** than the reconstruction loss:

```
Epoch 1 losses:
  L_recon ≈ 0.014       (dominant)
  L1      ≈ 0.000001    (negligible)
  KL      ≈ 0.000004    (negligible)
```

This suggests the regularization terms may be **too weak** to meaningfully shape the learned representations. The λ coefficients were not extensively tuned — increasing `λ_L1` and `λ_KL` by 2–3 orders of magnitude could potentially improve results and make the sparse autoencoder more competitive with MAE. This is a promising direction for future work.

### Why MAE outperforms this approach

| Aspect | MAE | Sparse Autoencoder |
|:-------|:----|:-------------------|
| **Information bottleneck** | Strong — only 25% of tokens visible | Weak — full input available |
| **Pretext difficulty** | Hard — must infer missing structure | Easy — input ≈ output |
| **Representation quality** | Learns contextual, predictive features | Learns identity-like mappings |
| **Regularization** | Implicit (masking forces generalization) | Explicit but undertuned (L1 + KL) |

The masking in MAE creates a much stronger information bottleneck, forcing the encoder to learn meaningful spatial and channel relationships. The sparse autoencoder, with its full-input-available setup and weak regularization, tends toward learning near-identity mappings that don't transfer as well to classification.

---

## File Structure

```
├── Sparse_ResNet/
│   ├── pretrain.py              # Autoencoder pretraining (L1 + KL)
│   ├── finetune.py              # Supervised fine-tuning
│   ├── pretrain_history.json    # Per-epoch losses (total, recon, L1, KL)
│   ├── finetune_history.json
│   ├── finetune_metrics.json
│   └── *.jpg                    # Training plots
└── dense_resnet_sae/
    ├── pretrain.py              # Dense autoencoder baseline
    └── finetune.py
```
