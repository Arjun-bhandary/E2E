# Mermaid Diagrams for E2E Repo
# Copy each ```mermaid block into the appropriate README

---

## 1. MAE Pretraining Pipeline (Main README)

```mermaid
flowchart LR
    subgraph INPUT["🔹 Input"]
        A["Sparse Jet Image\n125×125×8\n~1500 active pixels"]
    end

    subgraph MASK["🎭 Masking (75%)"]
        B["Active Tokens\nN ≈ 1500"]
        C["Visible 25%\n~375 tokens"]
        D["Masked 75%\n~1125 tokens"]
        B --> C
        B --> D
    end

    subgraph ENCODER["🔷 Sparse ResNet Encoder"]
        E["4-Stage Encoder\n8→64→128→256→512\nSubMConv2d + SparseConv2d"]
    end

    subgraph DECODER["🔶 Decoder"]
        F["Mirror Encoder\nSparseInverseConv2d\n512→256→128→64→8"]
    end

    subgraph LOSS["📉 Loss"]
        G["MSE on Masked\nTokens Only"]
    end

    A --> B
    C --> E
    E --> F
    F --> G
    D -.->|"target"| G

    style INPUT fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style MASK fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style ENCODER fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style DECODER fill:#3b1f0b,stroke:#f59e0b,color:#e2e8f0
    style LOSS fill:#1a2e1a,stroke:#22c55e,color:#e2e8f0
```

---

## 2. Fine-tuning Pipeline (Main README)

```mermaid
flowchart LR
    subgraph INPUT["🔹 Input"]
        A["Sparse Jet Image\n125×125×8"]
    end

    subgraph PHASE1["❄️ Phase 1 — Frozen (Epochs 1-5)"]
        B["Pretrained Encoder\n🔒 Frozen\nlr = 0"]
    end

    subgraph HEAD["🟢 Classification Head"]
        C["GAP → Dense 512→256→64→1\nlr = 1e-3\nDropout 0.5"]
    end

    subgraph PHASE2["🔥 Phase 2 — Unfrozen (Epochs 6+)"]
        D["Full Model\n🔓 Encoder lr = 5e-5\n🟢 Head lr = 1e-3"]
    end

    subgraph OUT["📊 Output"]
        E["Quark vs Gluon\nBCEWithLogitsLoss"]
    end

    A --> B --> C --> E
    A -.->|"after epoch 5"| D --> E

    style INPUT fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style PHASE1 fill:#1e293b,stroke:#60a5fa,color:#93c5fd
    style HEAD fill:#1a2e1a,stroke:#22c55e,color:#e2e8f0
    style PHASE2 fill:#3b1f0b,stroke:#f59e0b,color:#e2e8f0
    style OUT fill:#1e293b,stroke:#a78bfa,color:#e2e8f0
```

---

## 3. Sparse Tensor Data Structure (SparseConvolutions README)

```mermaid
flowchart TD
    subgraph DENSE["Dense Image (125×125×8)"]
        A["15,625 pixels × 8 channels\n~90-95% are zero"]
    end

    subgraph EXTRACT["Extract Non-Zero"]
        B["Find active pixels\nwhere sum(channels) > 0"]
    end

    subgraph SPARSE["spconv.SparseConvTensor"]
        C["indices: N×2\n(row, col) of active pixels"]
        D["features: N×8\n8-channel values"]
        E["spatial_shape: (125, 125)"]
        F["batch_size: B"]
    end

    DENSE --> EXTRACT
    EXTRACT --> C
    EXTRACT --> D
    C --> SPARSE
    D --> SPARSE
    E --> SPARSE
    F --> SPARSE

    style DENSE fill:#1e293b,stroke:#ef4444,color:#e2e8f0
    style EXTRACT fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style SPARSE fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
```

---

## 4. Sparse Convolution Primitives (SparseConvolutions README)

```mermaid
flowchart LR
    subgraph SUB["SubMConv2d"]
        direction TB
        S1["Input sparse tensor"]
        S2["3×3 conv at active sites ONLY"]
        S3["Output: same sparsity pattern"]
        S1 --> S2 --> S3
    end

    subgraph STRIDED["SparseConv2d (stride=2)"]
        direction TB
        T1["Input sparse tensor"]
        T2["3×3 conv, stride 2"]
        T3["Output: new sparsity pattern\nspatial dims halved"]
        T1 --> T2 --> T3
    end

    subgraph INVERSE["SparseInverseConv2d"]
        direction TB
        U1["Latent sparse tensor"]
        U2["Reuse cached index maps"]
        U3["Output: original resolution\nrestored"]
        U1 --> U2 --> U3
    end

    style SUB fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style STRIDED fill:#3b1f0b,stroke:#f59e0b,color:#e2e8f0
    style INVERSE fill:#1a2e1a,stroke:#22c55e,color:#e2e8f0
```

---

## 5. SparseResBlock (SparseConvolutions README)

```mermaid
flowchart TD
    IN["Input\n(C_in channels)"] --> BN1["BatchNorm1d"] --> RELU1["ReLU"]
    RELU1 --> CONV1["SubMConv2d\nC_in → C_out, 3×3"]
    CONV1 --> BN2["BatchNorm1d"] --> RELU2["ReLU"]
    RELU2 --> CONV2["SubMConv2d\nC_out → C_out, 3×3"]
    CONV2 --> ADD["➕ Add"]

    IN -->|"skip connection\n(1×1 SubMConv2d\nif C_in ≠ C_out)"| SKIP["1×1 Conv / Identity"]
    SKIP --> ADD

    ADD --> RELU3["ReLU"] --> OUT["Output\n(C_out channels)"]

    style IN fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style CONV1 fill:#1e3a5f,stroke:#60a5fa,color:#e2e8f0
    style CONV2 fill:#1e3a5f,stroke:#60a5fa,color:#e2e8f0
    style SKIP fill:#3b1f0b,stroke:#f59e0b,color:#fcd34d
    style ADD fill:#1a2e1a,stroke:#22c55e,color:#e2e8f0
    style OUT fill:#1e293b,stroke:#a78bfa,color:#e2e8f0
    style BN1 fill:#1e293b,stroke:#64748b,color:#e2e8f0
    style BN2 fill:#1e293b,stroke:#64748b,color:#e2e8f0
    style RELU1 fill:#1e293b,stroke:#64748b,color:#e2e8f0
    style RELU2 fill:#1e293b,stroke:#64748b,color:#e2e8f0
    style RELU3 fill:#1e293b,stroke:#64748b,color:#e2e8f0
```

---

## 6. Full Sparse ResNet Encoder (SparseConvolutions README)

```mermaid
flowchart TD
    IN["Input\nSparseConvTensor\nN×8, 125×125"] --> STEM["Stem\nSubMConv2d 8→64\nBN + ReLU"]

    STEM --> S1["Stage 1\n2× SparseResBlock\n64 → 64"]
    S1 --> D1["⬇ SparseConv2d\n64→128, stride 2\n→ 63×63"]

    D1 --> S2["Stage 2\n2× SparseResBlock\n128 → 128"]
    S2 --> D2["⬇ SparseConv2d\n128→256, stride 2\n→ 32×32"]

    D2 --> S3["Stage 3\n2× SparseResBlock\n256 → 256"]
    S3 --> D3["⬇ SparseConv2d\n256→512, stride 2\n→ 16×16"]

    D3 --> S4["Stage 4\n2× SparseResBlock\n512 → 512"]

    S4 --> OUT["Latent Features\nN'×512, 16×16"]

    style IN fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style STEM fill:#1e3a5f,stroke:#60a5fa,color:#e2e8f0
    style S1 fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style S2 fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style S3 fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style S4 fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style D1 fill:#3b1f0b,stroke:#f59e0b,color:#fcd34d
    style D2 fill:#3b1f0b,stroke:#f59e0b,color:#fcd34d
    style D3 fill:#3b1f0b,stroke:#f59e0b,color:#fcd34d
    style OUT fill:#1a2e1a,stroke:#22c55e,color:#e2e8f0
```

---

## 7. Sparse Autoencoder Pipeline (SparseAutoencoder README)

```mermaid
flowchart LR
    subgraph INPUT["🔹 Input"]
        A["Full Sparse Jet\n125×125×8\nNO masking"]
    end

    subgraph ENCODER["🔷 Sparse ResNet Encoder"]
        B["4-Stage Encoder\n8→64→128→256→512"]
    end

    subgraph LATENT["📌 Latent z"]
        C["Sparse features\nN'×512"]
        D["L1 penalty: λ₁·‖z‖₁\nKL penalty: λ_KL·KL(ρ‖ρ̂)"]
    end

    subgraph DECODER["🔶 Decoder"]
        E["Mirror Encoder\nSparseInverseConv2d\n512→...→8"]
    end

    subgraph LOSS["📉 Total Loss"]
        F["L_recon (MSE)\n+ λ₁·‖z‖₁\n+ λ_KL·KL(ρ‖ρ̂)"]
    end

    A --> B --> C
    C --> D
    C --> E --> F
    A -.->|"reconstruction\ntarget"| F

    style INPUT fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style ENCODER fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style LATENT fill:#2d1b4e,stroke:#a78bfa,color:#e2e8f0
    style DECODER fill:#3b1f0b,stroke:#f59e0b,color:#e2e8f0
    style LOSS fill:#1a2e1a,stroke:#22c55e,color:#e2e8f0
```

---

## 8. Pruning Methodology Flow (pruning README)

```mermaid
flowchart TD
    A["Fine-tuned Model\n(Sparse ResNet / ViT / AE)"] --> B["Collect All Weights\nSubMConv2d + SparseConv2d\n+ Linear layers"]

    B --> C["Global L1 Ranking\nSort all weights by |w|"]

    C --> D["Zero Bottom P%\nP ∈ {0, 10, 20, ..., 90, 95}"]

    D --> E["Make Permanent\nRemove mask,\nweights = 0"]

    E --> F["Evaluate on Val Set\nAUC, Accuracy, Error"]

    E --> G["Estimate FLOPS\nSpatial sparsity × Weight sparsity"]

    F --> H["📊 Error vs FLOPS\nPlot"]
    G --> H

    style A fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style B fill:#1e293b,stroke:#60a5fa,color:#e2e8f0
    style C fill:#3b1f0b,stroke:#f59e0b,color:#fcd34d
    style D fill:#3b1f0b,stroke:#ef4444,color:#fcd34d
    style E fill:#1e293b,stroke:#64748b,color:#e2e8f0
    style F fill:#1a2e1a,stroke:#22c55e,color:#e2e8f0
    style G fill:#1a2e1a,stroke:#22c55e,color:#e2e8f0
    style H fill:#2d1b4e,stroke:#a78bfa,color:#e2e8f0
```
