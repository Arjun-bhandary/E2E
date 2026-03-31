# Post-Training Pruning Analysis

This directory contains pruning experiments for three fine-tuned models, measuring the trade-off between model compression (weight sparsity / FLOPS reduction) and classification performance.

---

## Method

We apply **global unstructured magnitude pruning** using PyTorch's `torch.nn.utils.prune.L1Unstructured`. All prunable weights across the entire model (sparse convolution layers + linear classifier layers) are pooled together, ranked by absolute magnitude, and the smallest weights are zeroed out.

```
Global L1 Unstructured Pruning
──────────────────────────────
1. Collect all weight tensors from SubMConv2d, SparseConv2d, and Linear layers
2. Concatenate into a single vector, sort by |w|
3. Zero out the bottom P% (the pruning ratio)
4. Make permanent: remove mask, set weights to zero
5. Evaluate on the validation set (2,000 samples)
```

This is applied **post-training** — no retraining or fine-tuning is done after pruning. The goal is to characterize the inherent redundancy in each model architecture.

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

FLOPS for sparse convolution models are estimated analytically, accounting for **two levels of sparsity**:

### Level 1 — Spatial Sparsity (inherent to sparse convolutions)

Sparse convolutions only compute at **active sites** (non-zero pixels), not the full 125×125 grid. A typical jet has ~1,500 active pixels out of 15,625 total (~10% occupancy). This already gives a ~10× FLOPS reduction over dense convolutions.

### Level 2 — Weight Sparsity (from pruning)

After magnitude pruning, zeroed weights contribute no multiply-accumulate operations. FLOPS scale linearly with the density (fraction of non-zero weights):

```
FLOPS_layer = 2 × K² × C_in × C_out × N_active × (1 − sparsity)

Where:
  K         = kernel size (3 for most layers)
  C_in      = input channels
  C_out     = output channels
  N_active  = number of active sites at this resolution
  sparsity  = fraction of zero weights after pruning
```

Active sites decrease through the encoder due to strided `SparseConv2d` downsampling:

```
Resolution    Active sites (approx.)
125×125       ~1500  (N₀)
63×63         ~900   (0.6 × N₀)
32×32         ~450   (0.3 × N₀)
16×16         ~180   (0.12 × N₀)
```

The classifier head FLOPS (Linear layers: 512→256→64→1) are small relative to the encoder and scale the same way with weight pruning.

---

## Results

### Sparse ResNet MAE (Best Model)

<p align="center">
  <img src="https://raw.githubusercontent.com/Arjun-bhandary/E2E/main/pruning/sparse_resnet/sparse_resnet_error_vs_flops.png" alt="Sparse ResNet MAE — Error vs FLOPS" width="650"/>
  <br/>
  <em>Error (1 − Accuracy) vs. estimated GFLOPS for the Sparse ResNet MAE at various pruning ratios.</em>
</p>

### Sparse ViT MAE

<p align="center">
  <img src="https://raw.githubusercontent.com/Arjun-bhandary/E2E/main/pruning/sparse_resnet/sparse_vit_error_vs_flops.png" alt="Sparse ViT MAE — Error vs FLOPS" width="650"/>
  <br/>
  <em>Error vs. GFLOPS for the Sparse ViT MAE.</em>
</p>


### Sparse Autoencoder (L1 + KL)

<p align="center">
  <img src="https://raw.githubusercontent.com/Arjun-bhandary/E2E/main/pruning/sparse_resnet/sparse_ae_error_vs_flops.png" alt="Sparse Autoencoder — Error vs FLOPS" width="650"/>
  <br/>
  <em>Error vs. GFLOPS for the Sparse Autoencoder.</em>
</p>


---

## Combined Comparison

<p align="center">
  <img src="../assets/error_vs_flops_all_models.png" alt="Error vs FLOPS — All Models" width="700"/>
  <br/>
  <em>Error vs. GFLOPS for all three models across pruning ratios. The Sparse ResNet MAE dominates — lower error at any given FLOPS budget.</em>
</p>

---

## File Structure

```
pruning/
├── README.md
├── sparse_resnet/
│   ├── prune_sparse_resnet.py           # Pruning script
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

Each script loads the fine-tuned checkpoint, applies pruning at all 11 sparsity levels, evaluates on the validation set, estimates FLOPS, and generates the Error vs. FLOPS plot.

### Running

```bash
# From repo root
python pruning/sparse_resnet/prune_sparse_resnet.py
python pruning/sparse_vit/prune_sparse_vit.py
python pruning/sparse_ae/prune_sparse_ae.py
```

Requires a GPU and access to the fine-tuned checkpoints and dataset.
