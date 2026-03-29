
import os, json, math, copy, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import torch, torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader


# Hardcoded Paths
CHECKPOINT_PATH    = "/raid/home/dgx1736/Arush1/sparse_vit/finetuned_classifier.pt"
DATA_PATH          = "/raid/home/dgx1736/Arush1/Dataset_Specific_labelled.h5"
OUTPUT_DIR         = "/raid/home/dgx1736/Arush1/pruning/sparse_vit"

# Hyperparameters & Settings
IN_CHANNELS        = 8
MAX_TOKENS         = 1024
EMBED_DIM          = 256
N_HEADS            = 8
ENC_LAYERS         = 6
FFN_DIM            = 1024
DROPOUT            = 0.1
DROPOUT_CLS        = 0.5
SEED               = 42
LABELED_JET_OFFSET = 2048
LABELED_Y_OFFSET   = 5000002048
LABELED_N_SAMPLES  = 10000
VAL_SPLIT          = 0.2
BATCH_SIZE         = 32

PRUNING_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

def seed_everything(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LabelledTokenDataset(Dataset):
    def __init__(self, path, threshold=0.0, max_tokens=1024):
        self.path, self.threshold, self.max_tokens = path, threshold, max_tokens
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
        coords = np.argwhere(img.sum(axis=-1) > self.threshold)
        n = len(coords)
        if n == 0:
            coords = np.array([[0, 0]], dtype=np.int32)
            feats = np.zeros((1, IN_CHANNELS), dtype=np.float32); n = 1
        else:
            feats = img[coords[:, 0], coords[:, 1], :]
        if n > self.max_tokens:
            sel = np.random.choice(n, self.max_tokens, replace=False)
            coords, feats = coords[sel], feats[sel]; n = self.max_tokens

        positions = np.zeros((self.max_tokens, 2), dtype=np.float32)
        features = np.zeros((self.max_tokens, IN_CHANNELS), dtype=np.float32)
        n_real = min(n, self.max_tokens)
        positions[:n_real] = coords[:n_real].astype(np.float32)
        features[:n_real] = feats[:n_real]
        pad_mask = np.ones(self.max_tokens, dtype=bool)
        pad_mask[:n_real] = False

        return (torch.FloatTensor(positions), torch.FloatTensor(features),
                torch.BoolTensor(pad_mask), torch.FloatTensor([self.Y[idx]]))

def collate_fn(batch):
    pos, feat, pad, labels = zip(*batch)
    return torch.stack(pos), torch.stack(feat), torch.stack(pad), torch.cat(labels)



class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        half_d = d_model // 2
        freq = torch.exp(torch.arange(0, half_d, 2).float() * -(math.log(10000.0) / half_d))
        self.register_buffer('freq', freq)
    def forward(self, positions):
        B, T, _ = positions.shape; half_d = self.d_model // 2
        row, col = positions[:, :, 0:1], positions[:, :, 1:2]
        row_enc = torch.zeros(B, T, half_d, device=positions.device)
        row_enc[:, :, 0::2] = torch.sin(row * self.freq)
        row_enc[:, :, 1::2] = torch.cos(row * self.freq)
        col_enc = torch.zeros(B, T, half_d, device=positions.device)
        col_enc[:, :, 0::2] = torch.sin(col * self.freq)
        col_enc[:, :, 1::2] = torch.cos(col * self.freq)
        return torch.cat([row_enc, col_enc], dim=-1)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))
    def forward(self, x, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class SparseViTEncoder(nn.Module):
    def __init__(self, in_channels=8, d_model=256, n_heads=8, n_layers=6,
                 ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Sequential(nn.Linear(in_channels, d_model), nn.LayerNorm(d_model))
        self.pos_enc = SinusoidalPositionalEncoding2D(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, positions, features, pad_mask):
        B = features.shape[0]
        tokens = self.token_embed(features) + self.pos_enc(positions)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        full_pad = torch.cat([cls_pad, pad_mask], dim=1)
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=full_pad)
        return self.norm(tokens)

    def forward_cls(self, positions, features, pad_mask):
        return self.forward(positions, features, pad_mask)[:, 0, :]

class SparseViTClassifier(nn.Module):
    def __init__(self, in_channels=8, d_model=256, n_heads=8, n_layers=6,
                 ffn_dim=1024, dropout=0.1, dropout_cls=0.5):
        super().__init__()
        self.encoder = SparseViTEncoder(in_channels, d_model, n_heads, n_layers, ffn_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_cls),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(dropout_cls * 0.5),
            nn.Linear(64, 1))
    def forward(self, positions, features, pad_mask):
        cls_feat = self.encoder.forward_cls(positions, features, pad_mask)
        return self.classifier(cls_feat)



def estimate_flops(sparsity_ratio=0.0, avg_tokens=500):
    """
    Analytical FLOPS estimate for Sparse ViT classifier.

    Dominant costs:
      - Self-attention per layer: QKV projections + score computation + output proj
      - FFN per layer: two large linear layers
      - Token embed + classifier head
    """
    density = 1.0 - sparsity_ratio
    d = EMBED_DIM       # 256
    T = avg_tokens + 1  # +1 for CLS
    ffn = FFN_DIM        # 1024

    flops = 0

    # Token embedding: Linear(8→256) applied to T tokens
    flops += 2 * IN_CHANNELS * d * T

    # Per transformer layer (× 6):
    for _ in range(ENC_LAYERS):
        # QKV projections: 3 × (2 × T × d × d)
        flops += 3 * 2 * T * d * d
        # Attention scores: Q@Kᵀ → (T×T×d), Attn@V → (T×T×d)
        flops += 2 * T * T * d
        # Output projection: Linear(d→d) × T
        flops += 2 * T * d * d
        # FFN: Linear(d→ffn) + Linear(ffn→d), each × T
        flops += 2 * T * d * ffn + 2 * T * ffn * d

    # Classifier: Linear(256→128) + Linear(128→64) + Linear(64→1)
    flops += 2*256*128 + 2*128*64 + 2*64*1

    flops *= density
    return flops


def get_prunable_parameters(model):
    """Return (module, 'weight') for all Linear layers + attention projections."""
    params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            params.append((module, 'weight'))
        # nn.MultiheadAttention stores in_proj_weight internally
        elif isinstance(module, nn.MultiheadAttention):
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                params.append((module, 'in_proj_weight'))
            if hasattr(module, 'out_proj'):
                params.append((module.out_proj, 'weight'))
    return params


def apply_pruning(model, ratio):
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
    for pos, feat, pad, labels in loader:
        pos, feat, pad, labels = pos.to(device), feat.to(device), pad.to(device), labels.to(device)
        logits = model(pos, feat, pad).squeeze()
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
    ax.plot(gflops, errors, color='#dc2626', marker='s', markersize=9,
            linewidth=2.5, label='Sparse ViT MAE', zorder=5)

    for i, (gf, er, pr) in enumerate(zip(gflops, errors, ratios)):
        offset = (7, 7) if i % 2 == 0 else (7, -14)
        ax.annotate(f'{pr:.0%}', (gf, er), textcoords='offset points',
                    xytext=offset, fontsize=8, color='#dc2626', alpha=0.85)

    ax.set_xlabel('GFLOPS (estimated)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Error (1 − Accuracy)', fontsize=13, fontweight='bold')
    ax.set_title('Sparse ViT MAE — Pruning: Error vs FLOPS', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(output_dir, 'sparse_vit_error_vs_flops.png')
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"Plot saved: {path}")



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ────────────────────────────────────────────────────────
    model = SparseViTClassifier(IN_CHANNELS, EMBED_DIM, N_HEADS, ENC_LAYERS,
                                 FFN_DIM, DROPOUT, DROPOUT_CLS).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded Sparse ViT MAE | Params: {n_params:,}")

    # ── Load data ─────────────────────────────────────────────────────────
    dataset = LabelledTokenDataset(DATA_PATH, max_tokens=MAX_TOKENS)
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
    json_path = os.path.join(OUTPUT_DIR, 'sparse_vit_pruning_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    plot_results(results, OUTPUT_DIR)
    print("Done!")


if __name__ == '__main__':
    main()