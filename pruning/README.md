# Post-Training Pruning Analysis

Pruning experiments for three fine-tuned models, measuring the trade-off between model compression and classification performance.

---

## Method

**Global unstructured magnitude pruning** via `torch.nn.utils.prune.L1Unstructured`. All prunable weights are pooled, ranked by |w|, and the smallest are zeroed out. No retraining after pruning.

```mermaid
flowchart LR
    A["Fine-tuned Model\nSparse ResNet / ViT / AE"] --> B["Collect All Weights\nSubMConv2d + SparseConv2d\n+ Linear layers"]
    B --> C["Global L1 Ranking\nSort all weights by |w|"]
    C --> D["Zero Bottom P%\nP ∈ 0,10,...,90,95"]
    D --> E["Make Permanent\nRemove mask\nweights = 0"]
    E --> F["Evaluate Val Set\nAUC, Accuracy, Error"]
    E --> G["Estimate FLOPS\nSpatial × Weight\nsparsity"]
    F --> H(("Error vs\nFLOPS"))
    G --> H

    style A fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style B fill:#2a2a2a,stroke:#888,color:#ccc
    style C fill:#4a3b1f,stroke:#b8960f,color:#fce8b2
    style D fill:#4a2a2a,stroke:#8a4a4a,color:#f0c0c0
    style E fill:#2a2a2a,stroke:#888,color:#ccc
    style F fill:#2d4a2d,stroke:#4a7a4a,color:#d4edda
    style G fill:#2d4a2d,stroke:#4a7a4a,color:#d4edda
    style H fill:#3d2d4a,stroke:#a78bfa,color:#d8cce8
```

### Sparsity Levels Tested

```
Pruning ratios: [0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 95%]
```

---

## Models Pruned

| Model | Unpruned AUC | Unpruned Accuracy | Directory |
|:------|:------------:|:-----------------:|:----------|
| **Sparse ResNet MAE** | 0.9609 | 0.904 | `sparse_resnet/` |
| **Sparse ViT MAE** | 0.9426 | 0.878 | `sparse_vit/` |
| **Sparse Autoencoder** | 0.9341 | 0.869 | `sparse_ae/` |

---

## FLOPS Estimation

FLOPS account for **two levels of sparsity**:

**Level 1 — Spatial sparsity** (inherent): sparse convolutions compute only at active sites (~1,500 of 15,625 pixels). Already ~10× cheaper than dense.

**Level 2 — Weight sparsity** (from pruning): zeroed weights skip computation. FLOPS scale linearly with density.

```
FLOPS_layer = 2 × K² × C_in × C_out × N_active × (1 − sparsity)
```

Active sites decrease through downsampling: 125² → ~1500, 63² → ~900, 32² → ~450, 16² → ~180.

---

## Results

### Sparse ResNet MAE (Best Model)

<p align="center">
  <img src="../assets/sparse_resnet_error_vs_flops.png" alt="Sparse ResNet — Error vs FLOPS" width="600"/>
</p>

### Sparse ViT MAE

<p align="center">
  <img src="../assets/sparse_vit_error_vs_flops.png" alt="Sparse ViT — Error vs FLOPS" width="600"/>
</p>

### Sparse Autoencoder (L1 + KL)

<p align="center">
  <img src="../assets/sparse_ae_error_vs_flops.png" alt="Sparse AE — Error vs FLOPS" width="600"/>
</p>

### Combined Comparison

<p align="center">
  <img src="https://raw.githubusercontent.com/Arjun-bhandary/E2E/main/assests/combined_result.png" alt="All Models — Error vs FLOPS" width="700"/>
  <br/>
  <em>Error (1 − Accuracy) vs. estimated GFLOPS across pruning ratios. Sparse ResNet MAE dominates at every FLOPS budget.</em>
</p>

---

## Key Observations

1. **High tolerance to pruning** — all models maintain near-baseline accuracy up to ~50% weight sparsity.

2. **Graceful degradation** — performance drops smoothly, sharpest between 80–95% sparsity.

3. **Double sparsity** — spatial sparsity (sparse convs on ~10% active pixels) + weight sparsity (pruning) means a 50%-pruned model uses ~20× fewer FLOPS than a dense unpruned equivalent.

4. **No retraining** — all results are one-shot pruning. Iterative pruning + retraining could improve further.

---

## File Structure

```
pruning/
├── README.md
├── sparse_resnet/
│   ├── prune_sparse_resnet.py
│   ├── sparse_resnet_pruning_results.json
│   └── sparse_resnet_error_vs_flops.png
├── sparse_vit/
│   ├── prune_sparse_vit.py
│   ├── sparse_vit_pruning_results.json
│   └── sparse_vit_error_vs_flops.png
└── sparse_ae/
    ├── prune_sparse_ae.py
    ├── sparse_ae_pruning_results.json
    └── sparse_ae_error_vs_flops.png
```

### Running

```bash
python pruning/sparse_resnet/prune_sparse_resnet.py
python pruning/sparse_vit/prune_sparse_vit.py
python pruning/sparse_ae/prune_sparse_ae.py
```

Requires GPU and access to fine-tuned checkpoints + dataset.
