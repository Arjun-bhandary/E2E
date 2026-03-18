import os, json, time, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import spconv.pytorch as spconv

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
UNLABELED_PATH = '/raid/home/dgx1736/Arush1/Dataset_Specific_Unlabelled.h5'
SAVE_DIR       = '/raid/home/dgx1736/Arush1/sparse_resnet_no_occupancy'
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED           = 42
THRESHOLD      = 0.0
BATCH_SIZE     = 64
EPOCHS         = 30
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
MASK_RATIO     = 0.75
SPATIAL_SIZE   = [125, 125]
IN_CHANNELS    = 8
ENCODER_DIM    = 512
DECODER_DIM    = 256
VAL_SPLIT      = 0.2
NUM_WORKERS    = 0
PATIENCE       = 7

UNLABELED_JET_OFFSET = 2048
UNLABELED_N_SAMPLES  = 60000

os.makedirs(SAVE_DIR, exist_ok=True)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(SEED)
print(f"Device: {DEVICE} | Seed: {SEED} | Mask: {MASK_RATIO}")
print(f"Ablation: NO occupancy head (reconstruction only)")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET — lazy per-sample reader
# ═══════════════════════════════════════════════════════════════════════════════
class UnlabelledSparseJetDataset(Dataset):
    def __init__(self, path, threshold=0.0):
        self.path, self.threshold = path, threshold
        self.n_samples = UNLABELED_N_SAMPLES
        self.sample_bytes = 125 * 125 * 8 * 4
        print(f"Dataset: {self.n_samples} samples (lazy reader)")

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        with open(self.path, 'rb') as f:
            f.seek(UNLABELED_JET_OFFSET + idx * self.sample_bytes)
            raw = f.read(self.sample_bytes)
        img = np.frombuffer(raw, dtype=np.float32).reshape(125, 125, 8).copy()
        img /= 255.0
        return img


def mae_collate_fn(batch, mask_ratio=MASK_RATIO, threshold=THRESHOLD):
    """Collate with masking. No occupancy sampling — only visible + masked."""
    vis_coords_list, vis_feats_list = [], []
    mask_coords_list, mask_feats_list = [], []

    for i, img in enumerate(batch):
        active_mask = img.sum(axis=-1) > threshold
        coords = np.argwhere(active_mask)
        n_active = len(coords)
        if n_active == 0:
            coords = np.array([[0, 0]], dtype=np.int32)
            feats = np.zeros((1, IN_CHANNELS), dtype=np.float32)
            n_active = 1
        else:
            feats = img[coords[:, 0], coords[:, 1], :]
            coords = coords.astype(np.int32)

        n_mask = max(1, int(n_active * mask_ratio))
        perm = np.random.permutation(n_active)
        vis_idx, mask_idx = perm[:n_active - n_mask], perm[n_active - n_mask:]

        bc = np.full((len(vis_idx), 1), i, dtype=np.int32)
        vis_coords_list.append(np.hstack([bc, coords[vis_idx]]))
        vis_feats_list.append(feats[vis_idx])

        bc_m = np.full((len(mask_idx), 1), i, dtype=np.int32)
        mask_coords_list.append(np.hstack([bc_m, coords[mask_idx]]))
        mask_feats_list.append(feats[mask_idx])

    return {
        'vis_coords':  torch.from_numpy(np.vstack(vis_coords_list)).int(),
        'vis_feats':   torch.from_numpy(np.vstack(vis_feats_list)).float(),
        'mask_coords': torch.from_numpy(np.vstack(mask_coords_list)).int(),
        'mask_feats':  torch.from_numpy(np.vstack(mask_feats_list)).float(),
        'batch_size':  len(batch),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL — Sparse ResNet Encoder (identical to full version)
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DECODER + MAE (reconstruction only — no occupancy)
# ═══════════════════════════════════════════════════════════════════════════════
class SparseMAEDecoder(nn.Module):
    def __init__(self, enc_dim=512, dec_dim=256, out_channels=8):
        super().__init__()
        self.up1 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(enc_dim, dec_dim, 3, indice_key='d3'),
            nn.BatchNorm1d(dec_dim), nn.ReLU(inplace=True))
        self.dec1 = spconv.SparseSequential(
            spconv.SubMConv2d(dec_dim, dec_dim, 3, padding=1, bias=False, indice_key='dec1'),
            nn.BatchNorm1d(dec_dim), nn.ReLU(inplace=True))
        self.up2 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(dec_dim, dec_dim//2, 3, indice_key='d2'),
            nn.BatchNorm1d(dec_dim//2), nn.ReLU(inplace=True))
        self.dec2 = spconv.SparseSequential(
            spconv.SubMConv2d(dec_dim//2, dec_dim//2, 3, padding=1, bias=False, indice_key='dec2'),
            nn.BatchNorm1d(dec_dim//2), nn.ReLU(inplace=True))
        self.up3 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(dec_dim//2, dec_dim//4, 3, indice_key='d1'),
            nn.BatchNorm1d(dec_dim//4), nn.ReLU(inplace=True))
        self.dec3 = spconv.SparseSequential(
            spconv.SubMConv2d(dec_dim//4, dec_dim//4, 3, padding=1, bias=False, indice_key='dec3'),
            nn.BatchNorm1d(dec_dim//4), nn.ReLU(inplace=True))
        # only reconstruction head — no occupancy head
        self.recon_head = nn.Linear(dec_dim//4, out_channels)

    def forward(self, x):
        x = self.up1(x); x = self.dec1(x)
        x = self.up2(x); x = self.dec2(x)
        x = self.up3(x); x = self.dec3(x)
        return x


def _vectorized_lookup(dec_indices, dec_feats, query_coords, device):
    dec_hash = dec_indices[:, 0].long() * 15625 + dec_indices[:, 1].long() * 125 + dec_indices[:, 2].long()
    query_hash = query_coords[:, 0].long() * 15625 + query_coords[:, 1].long() * 125 + query_coords[:, 2].long()
    max_hash = BATCH_SIZE * 15625 + 1
    lookup = torch.full((max_hash,), -1, dtype=torch.long, device=device)
    lookup[dec_hash.clamp(0, max_hash-1)] = torch.arange(len(dec_hash), device=device)
    matched = lookup[query_hash.clamp(0, max_hash-1)]
    found_mask = matched >= 0
    found_qi = torch.where(found_mask)[0]
    return dec_feats[matched[found_mask]], found_mask, found_qi


class SparseMAE_ReconOnly(nn.Module):
    """Sparse MAE with reconstruction loss only. No occupancy prediction."""
    def __init__(self, in_channels=8, enc_dim=512, dec_dim=256):
        super().__init__()
        self.encoder = SparseResNetEncoder(in_channels, base_ch=enc_dim//8)
        self.decoder = SparseMAEDecoder(enc_dim, dec_dim, in_channels)
        self.mask_token = nn.Parameter(torch.randn(1, in_channels) * 0.02)

    def forward(self, vis_coords, vis_feats, mask_coords, mask_feats, batch_size):
        n_masked = mask_coords.shape[0]
        all_coords = torch.cat([vis_coords, mask_coords], dim=0)
        all_feats = torch.cat([vis_feats, self.mask_token.expand(n_masked, -1)], dim=0)

        x = spconv.SparseConvTensor(all_feats, all_coords, SPATIAL_SIZE, batch_size)
        z = self.encoder(x)
        decoded = self.decoder(z)

        # reconstruction loss only
        rf, _, rqi = _vectorized_lookup(decoded.indices, decoded.features,
                                         mask_coords, vis_feats.device)
        if rf.shape[0] > 0:
            recon_preds = self.decoder.recon_head(rf)
            recon_targets = mask_feats[rqi]
            recon_loss = nn.functional.mse_loss(recon_preds, recon_targets)
        else:
            recon_loss = torch.tensor(0.0, device=vis_feats.device)

        return recon_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer):
    model.train(); tot, n = 0, 0
    for batch in tqdm(loader, desc='Train', leave=False):
        vc, vf = batch['vis_coords'].to(DEVICE), batch['vis_feats'].to(DEVICE)
        mc, mf = batch['mask_coords'].to(DEVICE), batch['mask_feats'].to(DEVICE)
        optimizer.zero_grad()
        loss = model(vc, vf, mc, mf, batch['batch_size'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item(); n += 1
    return tot / max(n, 1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); tot, n = 0, 0
    for batch in tqdm(loader, desc='Val  ', leave=False):
        vc, vf = batch['vis_coords'].to(DEVICE), batch['vis_feats'].to(DEVICE)
        mc, mf = batch['mask_coords'].to(DEVICE), batch['mask_feats'].to(DEVICE)
        loss = model(vc, vf, mc, mf, batch['batch_size'])
        tot += loss.item(); n += 1
    return tot / max(n, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
def save_recon_loss_plot(history, path):
    epochs = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_loss'] for h in history], 'b-', label='Train')
    ax.plot(epochs, [h['va_loss'] for h in history], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Sparse ResNet MAE (No Occ): Reconstruction Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    dataset = UnlabelledSparseJetDataset(UNLABELED_PATH, THRESHOLD)
    n = len(dataset); nv = int(VAL_SPLIT * n); nt = n - nv
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [nt, nv], generator=torch.Generator().manual_seed(SEED))
    print(f"Train: {nt}  Val: {nv}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=mae_collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=mae_collate_fn, drop_last=False)

    model = SparseMAE_ReconOnly(IN_CHANNELS, ENCODER_DIM, DECODER_DIM).to(DEVICE)
    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ne_ = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Total params: {np_:,} | Encoder: {ne_:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"\n{'Ep':<5} {'TrLoss':<12} {'VaLoss':<12} {'LR':<10}")
    print("-" * 40)

    best_val, history, pat = float('inf'), [], 0
    for epoch in range(1, EPOCHS + 1):
        tl = train_one_epoch(model, train_loader, optimizer)
        vl = evaluate(model, val_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        history.append(dict(epoch=epoch, tr_loss=tl, va_loss=vl, lr=lr))
        mk = ''
        if vl < best_val:
            best_val = vl; pat = 0
            torch.save(model.encoder.state_dict(), os.path.join(SAVE_DIR, 'sparse_mae_encoder.pt'))
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'sparse_mae_full.pt'))
            mk = ' ← best'
        else: pat += 1
        print(f"{epoch:<5} {tl:<12.6f} {vl:<12.6f} {lr:<10.2e}{mk}")
        if pat >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}"); break

    print(f"\nBest Val Loss: {best_val:.6f} | Time: {(time.time()-t0)/3600:.2f}h")
    with open(os.path.join(SAVE_DIR, 'pretrain_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    save_recon_loss_plot(history, os.path.join(SAVE_DIR, 'pretrain_recon_loss.jpg'))
    print(f"Encoder saved to: {os.path.join(SAVE_DIR, 'sparse_mae_encoder.pt')}")
    print("Done.")
