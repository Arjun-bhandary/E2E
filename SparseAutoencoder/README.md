# Sparse Autoencoder for Jet Classification

This directory explores an alternative to Masked Autoencoding: a **traditional sparse autoencoder** with L1 sparsity and KL divergence regularization on the latent space.

---

## Approach

Instead of masking tokens and reconstructing them (MAE), the sparse autoencoder:

1. Encodes the **full** jet image (no masking) into a latent representation
2. Decodes back to reconstruct the original
3. Adds **L1 sparsity** and **KL divergence** penalties on the latent activations to encourage a sparse, distributed internal representation

The total pretraining loss is:

```
L_total = L_recon (MSE) + λ_L1 · ||z||₁ + λ_KL · KL(ρ || ρ̂)
```

where ρ is a target sparsity level and ρ̂ is the average activation of each latent unit.

## Experiments

### Sparse_ResNet/

Uses the same Sparse ResNet encoder architecture as the convolution-based models, but pretrained with the autoencoder objective (L1 + KL) instead of MAE.

| Metric | Value |
|:-------|:-----:|
| AUC | 0.9341 |
| Accuracy | 0.869 |
| F1 | 0.871 |
| 1/FPR @ TPR=0.7 | 14.1 |

This is the weakest-performing model in the comparison (ΔAUC = −0.0268 vs. the best MAE model), suggesting that masking-based pretext tasks learn more transferable representations than reconstruction with sparsity regularization.

### dense_resnet_sae/

A dense (non-sparse) ResNet autoencoder baseline for comparison. Uses standard `nn.Conv2d` instead of `spconv.SubMConv2d`, operating on the full 125×125×8 grid including zero pixels.

## Key Observation

The sparse autoencoder's L1 and KL losses are orders of magnitude smaller than the reconstruction loss (L1 ≈ 1e-6, KL ≈ 4e-6 vs. recon ≈ 0.014 at epoch 1), suggesting the regularization terms may be too weak to meaningfully shape the learned representations. Tuning λ_L1 and λ_KL could improve results.

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
