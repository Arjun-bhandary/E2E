

import os, json, copy, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import torch, torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
import spconv.pytorch as spconv


# Hardcoded Paths
CHECKPOINT_PATH    = "/raid/home/dgx1736/Arush1/sparse_resnet_no_occupancy/finetuned_classifier.pt"
DATA_PATH          = "/raid/home/dgx1736/Arush1/Dataset_Specific_labelled.h5"
OUTPUT_DIR         = "/raid/home/dgx1736/Arush1/pruning/sparse_resnet"

# Hyperparameters & Settings
SPATIAL_SIZE       = [125, 125]
IN_CHANNELS        = 8
ENCODER_DIM        = 512
SEED               = 42
LABELED_JET_OFFSET = 2048
LABELED_Y_OFFSET   = 5000002048
LABELED_N_SAMPLES  = 10000
VAL_SPLIT          = 0.2
BATCH_SIZE         = 64

PRUNING_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

def seed_everything(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LabelledSparseJetDataset(Dataset):
    def __init__(self, path, threshold=0.0):
        self.path, self.threshold = path, threshold
        self.n_samples = LABELED_N_SAMPLES
        self.sample_bytes = 125 * 125 * 8 * 4
        with open(path, 'rb') as f:
            f.seek(LABELED_Y_OFFSET)
            self.Y = np.frombuffer(f.read(self.n_samples * 4), dtype=np.float32).copy()

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        with open(self.path, 'rb') as f:
            f.seek(LABELED_JET_OFFSET + idx * self.sample_bytes)
            raw = f.read(self.sample_bytes)
        img = np.frombuffer(raw, dtype=np.float32).reshape(125, 125, 8).copy() / 255.0
        y = self.Y[idx]
        coords = np.argwhere(img.sum(axis=-1) > self.threshold)
        if len(coords) == 0:
            coords = np.array([[0, 0]], dtype=np.int32)
            feats = np.zeros((1, IN_CHANNELS), dtype=np.float32)
        else:
            feats = img[coords[:, 0], coords[:, 1], :]
            coords = coords.astype(np.int32)
        return {'coords': torch.IntTensor(coords),
                'feats': torch.FloatTensor(feats),
                'label': torch.FloatTensor([y])}

def collate_fn(batch):
    cl, fl, ll = [], [], []
    for i, s in enumerate(batch):
        n = s['coords'].shape[0]
        cl.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), s['coords']], 1))
        fl.append(s['feats']); ll.append(s['label'])
    return torch.cat(cl), torch.cat(fl), torch.cat(ll)



class SparseResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, indice_key=None):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch); self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = spconv.SubMConv2d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(out_ch); self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv2d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key)
        self.skip = (spconv.SubMConv2d(in_ch, out_ch, 1, bias=False, indice_key=indice_key+'_skip')
                     if in_ch != out_ch else None)
    def forward(self, x):
        identity = x
        out = x.replace_feature(self.relu1(self.bn1(x.features)))
        out = self.conv1(out)
        out = out.replace_feature(self.relu2(self.bn2(out.features)))
        out = self.conv2(out)
        if self.skip: identity = self.skip(identity)
        return out.replace_feature(out.features + identity.features)

class SparseDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, indice_key=None):
        super().__init__()
        self.conv = spconv.SparseConv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False, indice_key=indice_key)
        self.bn = nn.BatchNorm1d(out_ch); self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x); return x.replace_feature(self.relu(self.bn(x.features)))

class SparseResNetEncoder(nn.Module):
    def __init__(self, in_channels=8, base_ch=64):
        super().__init__()
        self.stem = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, base_ch, 3, padding=1, bias=False, indice_key='stem'),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True))
        self.stage1 = nn.ModuleList([SparseResBlock(base_ch, base_ch, 's1'),
                                     SparseResBlock(base_ch, base_ch, 's1')])
        self.down1 = SparseDownsample(base_ch, base_ch, 'd1')
        self.stage2 = nn.ModuleList([SparseResBlock(base_ch, base_ch*2, 's2'),
                                     SparseResBlock(base_ch*2, base_ch*2, 's2b')])
        self.down2 = SparseDownsample(base_ch*2, base_ch*2, 'd2')
        self.stage3 = nn.ModuleList([SparseResBlock(base_ch*2, base_ch*4, 's3'),
                                     SparseResBlock(base_ch*4, base_ch*4, 's3b')])
        self.down3 = SparseDownsample(base_ch*4, base_ch*4, 'd3')
        self.stage4 = nn.ModuleList([SparseResBlock(base_ch*4, base_ch*8, 's4'),
                                     SparseResBlock(base_ch*8, base_ch*8, 's4b')])
    def forward(self, x):
        x = self.stem(x)
        for b in self.stage1: x = b(x)
        x = self.down1(x)
        for b in self.stage2: x = b(x)
        x = self.down2(x)
        for b in self.stage3: x = b(x)
        x = self.down3(x)
        for b in self.stage4: x = b(x)
        return x

class SparseMAEClassifier(nn.Module):
    def __init__(self, in_channels=8, enc_dim=512, dropout=0.5):
        super().__init__()
        self.encoder = SparseResNetEncoder(in_channels, enc_dim // 8)
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1))

    def forward(self, coords, feats, batch_size):
        x = spconv.SparseConvTensor(feats, coords, SPATIAL_SIZE, batch_size)
        x = self.encoder(x)
        features, batch_idx = x.features, x.indices[:, 0].long()
        pooled = torch.zeros(batch_size, features.shape[1], device=features.device)
        counts = torch.zeros(batch_size, 1, device=features.device)
        pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(features), features)
        counts.scatter_add_(0, batch_idx.unsqueeze(1),
                            torch.ones(len(batch_idx), 1, device=features.device))
        return self.classifier(pooled / counts.clamp(min=1))


def estimate_flops(sparsity_ratio=0.0, avg_active_pixels=1500):
    """
    Analytical FLOPS estimate for the Sparse ResNet classifier.

    For sparse convolutions, FLOPS scale with:
      - Number of active sites (NOT the full 125x125 grid)
      - Non-zero weights after pruning

    FLOPS ≈ Σ_layer (2 × K² × C_in × C_out × N_active) × (1 - sparsity)
    """
    density = 1.0 - sparsity_ratio
    base_ch = 64
    K = 3

    # Active sites at each encoder stage (shrinks with downsampling)
    n0 = avg_active_pixels           # 125×125 resolution
    n1 = n0 * 0.6                    # after down1 (stride 2, sparse ~60% survive)
    n2 = n1 * 0.5                    # after down2
    n3 = n2 * 0.4                    # after down3

    flops = 0

    # Stem: SubMConv 3×3, 8 → 64
    flops += 2 * K*K * 8 * base_ch * n0

    # Stage 1: 2× ResBlock(64→64), each = 2× SubMConv3×3
    for _ in range(2):
        flops += 2 * (2 * K*K * base_ch * base_ch * n0)

    # Down1: SparseConv2d 64→64 stride 2
    flops += 2 * K*K * base_ch * base_ch * n0

    # Stage 2: ResBlock(64→128) + ResBlock(128→128)
    flops += 2 * K*K * base_ch * base_ch*2 * n1        # conv1
    flops += 2 * K*K * base_ch*2 * base_ch*2 * n1      # conv2
    flops += 2 * 1 * base_ch * base_ch*2 * n1            # 1×1 skip
    flops += 2 * (2 * K*K * base_ch*2 * base_ch*2 * n1) # second block

    # Down2: SparseConv2d 128→128 stride 2
    flops += 2 * K*K * base_ch*2 * base_ch*2 * n1

    # Stage 3: ResBlock(128→256) + ResBlock(256→256)
    flops += 2 * K*K * base_ch*2 * base_ch*4 * n2
    flops += 2 * K*K * base_ch*4 * base_ch*4 * n2
    flops += 2 * 1 * base_ch*2 * base_ch*4 * n2
    flops += 2 * (2 * K*K * base_ch*4 * base_ch*4 * n2)

    # Down3: SparseConv2d 256→256 stride 2
    flops += 2 * K*K * base_ch*4 * base_ch*4 * n2

    # Stage 4: ResBlock(256→512) + ResBlock(512→512)
    flops += 2 * K*K * base_ch*4 * base_ch*8 * n3
    flops += 2 * K*K * base_ch*8 * base_ch*8 * n3
    flops += 2 * 1 * base_ch*4 * base_ch*8 * n3
    flops += 2 * (2 * K*K * base_ch*8 * base_ch*8 * n3)

    # Classifier: Linear(512→256) + Linear(256→64) + Linear(64→1)
    flops += 2*512*256 + 2*256*64 + 2*64*1

    # Pruned weights → proportional FLOPS reduction
    flops *= density
    return flops



def get_prunable_parameters(model):
    """Return (module, 'weight') pairs for all conv and linear layers."""
    params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            params.append((module, 'weight'))
        elif isinstance(module, (spconv.SubMConv2d, spconv.SparseConv2d)):
            params.append((module, 'weight'))
    return params


def apply_pruning(model, ratio):
    """Deep-copy the model, apply global L1 unstructured pruning, make permanent."""
    pruned = copy.deepcopy(model)
    if ratio <= 0.0:
        return pruned
    params = get_prunable_parameters(pruned)
    if not params:
        print("  WARNING: no prunable parameters found"); return pruned
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=ratio)
    for module, pname in params:
        try: prune.remove(module, pname)
        except ValueError: pass
    return pruned


def measure_sparsity(model):
    """Fraction of zero-valued weight entries."""
    total, zeros = 0, 0
    for name, p in model.named_parameters():
        if 'weight' in name:
            total += p.numel()
            zeros += (p.data == 0).sum().item()
    return zeros / max(total, 1)



@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for coords, feats, labels in loader:
        coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
        logits = model(coords, feats, labels.shape[0])
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, (np.array(all_probs) >= 0.5).astype(int))
    return auc, acc, 1.0 - acc



def plot_results(results, output_dir):
    gflops = [r['gflops'] for r in results]
    errors = [r['error'] for r in results]
    ratios = [r['pruning_ratio'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gflops, errors, color='#2563eb', marker='o', markersize=9,
            linewidth=2.5, label='Sparse ResNet MAE', zorder=5)

    for i, (gf, er, pr) in enumerate(zip(gflops, errors, ratios)):
        offset = (7, 7) if i % 2 == 0 else (7, -14)
        ax.annotate(f'{pr:.0%}', (gf, er), textcoords='offset points',
                    xytext=offset, fontsize=8, color='#2563eb', alpha=0.85)

    ax.set_xlabel('GFLOPS (estimated)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Error (1 − Accuracy)', fontsize=13, fontweight='bold')
    ax.set_title('Sparse ResNet MAE — Pruning: Error vs FLOPS', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(output_dir, 'sparse_resnet_error_vs_flops.png')
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"Plot saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ────────────────────────────────────────────────────────
    model = SparseMAEClassifier(IN_CHANNELS, ENCODER_DIM, 0.5).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded Sparse ResNet MAE | Params: {n_params:,}")

    # ── Load data (val split only) ────────────────────────────────────────
    dataset = LabelledSparseJetDataset(DATA_PATH)
    n = len(dataset); nv = int(VAL_SPLIT * n); nt = n - nv
    _, val_ds = torch.utils.data.random_split(
        dataset, [nt, nv], generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False, collate_fn=collate_fn)
    print(f"Validation samples: {nv}\n")

    # ── Pruning loop ──────────────────────────────────────────────────────
    results = []
    print(f"{'Ratio':<8} {'Sparsity':<10} {'AUC':<8} {'Acc':<8} {'Error':<8} {'GFLOPS':<10}")
    print("-" * 55)

    for ratio in PRUNING_RATIOS:
        pruned = apply_pruning(model, ratio)
        pruned = pruned.to(DEVICE)

        sparsity = measure_sparsity(pruned)
        auc, acc, error = evaluate(pruned, val_loader, DEVICE)
        flops = estimate_flops(sparsity_ratio=sparsity)
        gflops = flops / 1e9

        results.append({
            'pruning_ratio': ratio,
            'actual_sparsity': round(sparsity, 4),
            'auc': round(auc, 4),
            'accuracy': round(acc, 4),
            'error': round(error, 4),
            'flops': flops,
            'gflops': round(gflops, 4),
        })
        print(f"{ratio:<8.2f} {sparsity:<10.4f} {auc:<8.4f} {acc:<8.4f} {error:<8.4f} {gflops:<10.4f}")

        del pruned
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, 'sparse_resnet_pruning_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    plot_results(results, OUTPUT_DIR)
    print("Done!")


if __name__ == '__main__':
    main()
