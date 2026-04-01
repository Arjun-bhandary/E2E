# Sparse Autoencoder for Jet Classification

This directory explores an alternative to Masked Autoencoding: a traditional **sparse autoencoder** with L1 sparsity and KL divergence regularization on the latent space.

---

## Approach

Instead of masking tokens and reconstructing them (MAE), the sparse autoencoder encodes the **full** jet image (no masking), decodes to reconstruct the original, and adds **L1 + KL** penalties on the latent activations.

```mermaid
flowchart LR
    A["Full Sparse Jet\n125×125×8\nNO masking"] --> B["Sparse ResNet\nEncoder\n8→64→...→512"]
    B --> C["Latent z\nN'×512"]
    C --> D["Sparse Decoder\nSparseInverseConv\n512→...→8"]
    D --> E(("L_total\nMSE + λ₁‖z‖₁\n+ λ_KL · KL(ρ‖ρ̂)"))
    A -.->|"reconstruction target"| E
    C -.->|"L1 + KL\npenalties"| E

    style A fill:#2a2a2a,stroke:#888,color:#ccc
    style B fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style C fill:#3d2d4a,stroke:#a78bfa,color:#d8cce8
    style D fill:#4a3b1f,stroke:#b8960f,color:#fce8b2
    style E fill:#4a2a2a,stroke:#8a4a4a,color:#f0c0c0
```

### Loss Function

```
L_total = L_recon (MSE) + λ_L1 · ||z||₁ + λ_KL · KL(ρ || ρ̂)
```

| Term | Purpose | Typical magnitude (epoch 1) |
|:-----|:--------|:---------------------------:|
| **L_recon (MSE)** | Faithful reconstruction | ~0.014 |
| **λ_L1 · ‖z‖₁** | Sparse latent activations | ~1e-6 |
| **λ_KL · KL(ρ ‖ ρ̂)** | Push avg activation toward target ρ | ~4e-6 |

### MAE vs. Sparse Autoencoder

```mermaid
flowchart LR
    subgraph MAE["MAE (SparseConvolutions/)"]
        M1["25% visible\n75% masked"] --> M2["Encoder"] --> M3["Decoder"] --> M4["MSE on\nmasked only"]
    end

    subgraph SAE["Sparse Autoencoder (this dir)"]
        S1["100% visible\nno masking"] --> S2["Encoder"] --> S3["Decoder"] --> S4["MSE on all\n+ L1 + KL"]
    end

    style MAE fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style SAE fill:#3d2d4a,stroke:#a78bfa,color:#d8cce8
    style M1 fill:#2a2a2a,stroke:#888,color:#ccc
    style M2 fill:#2d4a2d,stroke:#4a7a4a,color:#d4edda
    style M3 fill:#4a3b1f,stroke:#b8960f,color:#fce8b2
    style M4 fill:#4a2a2a,stroke:#8a4a4a,color:#f0c0c0
    style S1 fill:#2a2a2a,stroke:#888,color:#ccc
    style S2 fill:#2d4a2d,stroke:#4a7a4a,color:#d4edda
    style S3 fill:#4a3b1f,stroke:#b8960f,color:#fce8b2
    style S4 fill:#4a2a2a,stroke:#8a4a4a,color:#f0c0c0
```

---

## Experiments

### Sparse_ResNet/

Uses the same Sparse ResNet encoder as the convolution-based models, pretrained with the autoencoder objective (L1 + KL) instead of MAE.

| Metric | Value |
|:-------|:-----:|
| **AUC** | 0.9341 |
| **Accuracy** | 0.869 |
| **F1** | 0.871 |
| **1/FPR @ TPR=0.7** | 14.1 |

This is the weakest-performing model (**ΔAUC = −0.0268** vs. the best MAE model), suggesting masking-based pretext tasks learn more transferable representations.

### dense_resnet_sae/

A **dense (non-sparse) ResNet autoencoder** baseline. Uses standard `nn.Conv2d` instead of `spconv.SubMConv2d`, operating on the full 125×125×8 grid including zero pixels.

```mermaid
flowchart LR
    A["Dense Input\nB×8×125×125\n(all 15,625 pixels)"] --> B["nn.Conv2d\nResBlocks\n+ MaxPool2d"]
    B --> C["Latent"] --> D["ConvTranspose2d\nUpsampling"]
    D --> E(("MSE + L1 + KL"))

    style A fill:#2a2a2a,stroke:#888,color:#ccc
    style B fill:#4a3b1f,stroke:#b8960f,color:#fce8b2
    style C fill:#3d2d4a,stroke:#a78bfa,color:#d8cce8
    style D fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style E fill:#4a2a2a,stroke:#8a4a4a,color:#f0c0c0
```

---

## Key Observation

The L1 and KL losses are **orders of magnitude smaller** than reconstruction loss (~1e-6 vs. ~0.014), suggesting the regularization terms are too weak to meaningfully shape representations. Increasing `λ_L1` and `λ_KL` by 2–3 orders of magnitude could improve results.

### Why MAE Outperforms

| Aspect | MAE | Sparse Autoencoder |
|:-------|:----|:-------------------|
| **Information bottleneck** | Strong — only 25% visible | Weak — full input |
| **Pretext difficulty** | Hard — infer missing structure | Easy — input ≈ output |
| **Representation quality** | Contextual, predictive features | Identity-like mappings |
| **Regularization** | Implicit (masking) | Explicit but undertuned |

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
