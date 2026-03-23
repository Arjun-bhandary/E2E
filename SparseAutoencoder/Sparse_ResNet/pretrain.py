#!/usr/bin/env python3
"""
pretrain_sparse_ae.py — Sparse Autoencoder Pretraining
=======================================================
Alternative architecture for bonus task:
  A standard autoencoder with L1 sparsity penalty on the latent
  bottleneck, using sparse convolutions (spconv) for jet image data.

Comparison:
  Baseline = Sparse ResNet MAE (masked autoencoder)
  This     = Sparse Autoencoder (L1-regularised bottleneck)

Data loading is identical to the baseline code.

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 pretrain_sparse_ae.py
"""

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
SAVE_DIR       = '/raid/home/dgx1736/Arush1/sparse_autoencoder'
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED           = 42
THRESHOLD      = 0.0
BATCH_SIZE     = 64
EPOCHS         = 60
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
SPATIAL_SIZE   = [125, 125]
IN_CHANNELS    = 8
ENCODER_DIM    = 512
DECODER_DIM    = 256
VAL_SPLIT      = 0.2
NUM_WORKERS    = 0
PATIENCE       = 10

# Sparsity hyperparameters — with warmup schedule
# Final values after warmup (moderate strength)
SPARSITY_LAMBDA = 5e-4     # L1 penalty weight (final)
SPARSITY_TARGET = 0.05     # target activation rate
SPARSITY_KL_BETA = 5e-4    # KL divergence penalty weight (final)
SPARSITY_WARMUP = 10       # epochs of pure reconstruction before sparsity kicks in
SPARSITY_RAMP   = 10       # epochs to linearly ramp from 0 to full penalty

UNLABELED_JET_OFFSET = 2048
UNLABELED_N_SAMPLES  = 60000

os.makedirs(SAVE_DIR, exist_ok=True)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(SEED)
print(f"Device: {DEVICE} | Seed: {SEED}")
print(f"Architecture: Sparse Autoencoder (L1 + KL sparsity on bottleneck)")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET — identical lazy per-sample reader
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


def ae_collate_fn(batch, threshold=THRESHOLD):
    """Collate for standard autoencoder: input = target (no masking)."""
    coords_list, feats_list = [], []

    for i, img in enumerate(batch):
        active_mask = img.sum(axis=-1) > threshold
        coords = np.argwhere(active_mask)
        n_active = len(coords)
        if n_active == 0:
            coords = np.array([[0, 0]], dtype=np.int32)
            feats = np.zeros((1, IN_CHANNELS), dtype=np.float32)
        else:
            feats = img[coords[:, 0], coords[:, 1], :]
            coords = coords.astype(np.int32)

        bc = np.full((len(coords), 1), i, dtype=np.int32)
        coords_list.append(np.hstack([bc, coords]))
        feats_list.append(feats)

    return {
        'coords': torch.from_numpy(np.vstack(coords_list)).int(),
        'feats':  torch.from_numpy(np.vstack(feats_list)).float(),
        'batch_size': len(batch),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL — Sparse Autoencoder with sparsity-penalised bottleneck
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
    """Identical encoder to baseline for fair comparison."""
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


class SparseAEDecoder(nn.Module):
    """Decoder that reconstructs input features from bottleneck."""
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
        self.recon_head = nn.Linear(dec_dim//4, out_channels)

    def forward(self, x):
        x = self.up1(x); x = self.dec1(x)
        x = self.up2(x); x = self.dec2(x)
        x = self.up3(x); x = self.dec3(x)
        return x


def _vectorized_lookup(dec_indices, dec_feats, query_coords, device):
    """Look up decoded features at query coordinate locations."""
    dec_hash = dec_indices[:, 0].long() * 15625 + dec_indices[:, 1].long() * 125 + dec_indices[:, 2].long()
    query_hash = query_coords[:, 0].long() * 15625 + query_coords[:, 1].long() * 125 + query_coords[:, 2].long()
    max_hash = BATCH_SIZE * 15625 + 1
    lookup = torch.full((max_hash,), -1, dtype=torch.long, device=device)
    lookup[dec_hash.clamp(0, max_hash-1)] = torch.arange(len(dec_hash), device=device)
    matched = lookup[query_hash.clamp(0, max_hash-1)]
    found_mask = matched >= 0
    found_qi = torch.where(found_mask)[0]
    return dec_feats[matched[found_mask]], found_mask, found_qi


def kl_divergence_sparsity(activations, target_rho=0.05):
    """KL divergence sparsity penalty on average activations."""
    rho_hat = activations.mean(dim=0).clamp(1e-6, 1-1e-6)  # average activation per neuron
    rho = torch.full_like(rho_hat, target_rho)
    kl = rho * torch.log(rho / rho_hat) + \
         (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return kl.mean()  # mean over neurons, not sum (sum explodes with 512 dims)


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder: standard autoencoder (no masking) with sparsity
    constraints on the bottleneck representation.

    Uses warmup schedule: pure reconstruction first, then gradually
    increases sparsity penalties to avoid killing features early.
    """
    def __init__(self, in_channels=8, enc_dim=512, dec_dim=256,
                 sparsity_lambda=5e-4, sparsity_target=0.05, sparsity_kl_beta=5e-4,
                 warmup_epochs=10, ramp_epochs=10):
        super().__init__()
        self.encoder = SparseResNetEncoder(in_channels, base_ch=enc_dim // 8)
        self.decoder = SparseAEDecoder(enc_dim, dec_dim, in_channels)
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target
        self.sparsity_kl_beta = sparsity_kl_beta
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

    def _sparsity_weight(self, epoch):
        """0 during warmup, linear ramp to 1.0 after."""
        if epoch <= self.warmup_epochs:
            return 0.0
        elif epoch <= self.warmup_epochs + self.ramp_epochs:
            return (epoch - self.warmup_epochs) / self.ramp_epochs
        return 1.0

    def forward(self, coords, feats, batch_size, epoch=1):
        x = spconv.SparseConvTensor(feats, coords, SPATIAL_SIZE, batch_size)
        z = self.encoder(x)

        w = self._sparsity_weight(epoch)

        # ── sparsity penalties on bottleneck (with warmup) ──
        if w > 0:
            # L1 on raw features
            l1_loss = w * self.sparsity_lambda * z.features.abs().mean()
            # KL on sigmoid of features — WITH gradient flow to encoder
            z_act = torch.sigmoid(z.features)
            kl_loss = w * self.sparsity_kl_beta * kl_divergence_sparsity(z_act, self.sparsity_target)
        else:
            l1_loss = torch.tensor(0.0, device=feats.device)
            kl_loss = torch.tensor(0.0, device=feats.device)

        # ── decode ──
        decoded = self.decoder(z)

        # ── reconstruction loss: look up decoded features at input locations ──
        rf, found_mask, found_qi = _vectorized_lookup(
            decoded.indices, decoded.features, coords, feats.device)

        if rf.shape[0] > 0:
            recon_preds = self.decoder.recon_head(rf)
            recon_targets = feats[found_qi]
            recon_loss = nn.functional.mse_loss(recon_preds, recon_targets)
        else:
            recon_loss = torch.tensor(0.0, device=feats.device)

        total_loss = recon_loss + l1_loss + kl_loss

        return total_loss, recon_loss.item(), l1_loss.item() if torch.is_tensor(l1_loss) else l1_loss, \
               kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, epoch=1):
    model.train()
    tot, tot_r, tot_l1, tot_kl, n = 0, 0, 0, 0, 0
    for batch in tqdm(loader, desc='Train', leave=False):
        coords = batch['coords'].to(DEVICE)
        feats = batch['feats'].to(DEVICE)
        bs = batch['batch_size']

        optimizer.zero_grad()
        loss, r, l1, kl = model(coords, feats, bs, epoch=epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot += loss.item(); tot_r += r; tot_l1 += l1; tot_kl += kl; n += 1
    return tot/max(n,1), tot_r/max(n,1), tot_l1/max(n,1), tot_kl/max(n,1)


@torch.no_grad()
def evaluate(model, loader, epoch=1):
    model.eval()
    tot, tot_r, tot_l1, tot_kl, n = 0, 0, 0, 0, 0
    for batch in tqdm(loader, desc='Val  ', leave=False):
        coords = batch['coords'].to(DEVICE)
        feats = batch['feats'].to(DEVICE)
        bs = batch['batch_size']

        loss, r, l1, kl = model(coords, feats, bs, epoch=epoch)
        tot += loss.item(); tot_r += r; tot_l1 += l1; tot_kl += kl; n += 1
    return tot/max(n,1), tot_r/max(n,1), tot_l1/max(n,1), tot_kl/max(n,1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
def save_loss_plots(history, save_dir):
    epochs = [h['epoch'] for h in history]

    # Total loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_total'] for h in history], 'b-', label='Train')
    ax.plot(epochs, [h['va_total'] for h in history], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
    ax.set_title('Sparse Autoencoder: Total Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'pretrain_total_loss.jpg'), dpi=150); plt.close()

    # Reconstruction loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_recon'] for h in history], 'b-', label='Train')
    ax.plot(epochs, [h['va_recon'] for h in history], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Sparse Autoencoder: Reconstruction Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'pretrain_recon_loss.jpg'), dpi=150); plt.close()

    # Sparsity losses
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_l1'] for h in history], 'b-', label='Train L1')
    ax.plot(epochs, [h['tr_kl'] for h in history], 'g-', label='Train KL')
    ax.plot(epochs, [h['va_l1'] for h in history], 'b--', label='Val L1')
    ax.plot(epochs, [h['va_kl'] for h in history], 'g--', label='Val KL')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Sparsity Penalty')
    ax.set_title('Sparse Autoencoder: Sparsity Losses')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'pretrain_sparsity_loss.jpg'), dpi=150); plt.close()

    print(f"Saved plots to {save_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
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
                              collate_fn=ae_collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=ae_collate_fn, drop_last=False)

    model = SparseAutoencoder(
        IN_CHANNELS, ENCODER_DIM, DECODER_DIM,
        sparsity_lambda=SPARSITY_LAMBDA,
        sparsity_target=SPARSITY_TARGET,
        sparsity_kl_beta=SPARSITY_KL_BETA,
        warmup_epochs=SPARSITY_WARMUP,
        ramp_epochs=SPARSITY_RAMP
    ).to(DEVICE)

    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ne_ = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Total params: {np_:,} | Encoder: {ne_:,}")
    print(f"Sparsity schedule: warmup {SPARSITY_WARMUP}ep → ramp {SPARSITY_RAMP}ep → full")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"\n{'Ep':<5} {'Total':<10} {'Recon':<10} {'L1':<10} {'KL':<10} {'VaTotal':<10} {'VaRecon':<10} {'SpW':<5}")
    print("-" * 75)

    best_val, history, pat = float('inf'), [], 0
    for epoch in range(1, EPOCHS + 1):
        tt, tr, tl1, tkl = train_one_epoch(model, train_loader, optimizer, epoch=epoch)
        vt, vr, vl1, vkl = evaluate(model, val_loader, epoch=epoch)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        sw = model._sparsity_weight(epoch)

        history.append(dict(epoch=epoch, tr_total=tt, tr_recon=tr, tr_l1=tl1, tr_kl=tkl,
                            va_total=vt, va_recon=vr, va_l1=vl1, va_kl=vkl, lr=lr,
                            sparsity_weight=sw))
        mk = ''
        if vr < best_val:
            best_val = vr; pat = 0
            torch.save(model.encoder.state_dict(),
                       os.path.join(SAVE_DIR, 'sparse_ae_encoder.pt'))
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, 'sparse_ae_full.pt'))
            mk = ' ← best'
        elif epoch > SPARSITY_WARMUP + SPARSITY_RAMP:
            pat += 1

        print(f"{epoch:<5} {tt:<10.6f} {tr:<10.6f} {tl1:<10.6f} {tkl:<10.6f} "
              f"{vt:<10.6f} {vr:<10.6f} {sw:<5.2f}{mk}")

        if pat >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}"); break

    print(f"\nBest Val Recon Loss: {best_val:.6f} | Time: {(time.time()-t0)/3600:.2f}h")
    with open(os.path.join(SAVE_DIR, 'pretrain_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    save_loss_plots(history, SAVE_DIR)
    print(f"Encoder saved to: {os.path.join(SAVE_DIR, 'sparse_ae_encoder.pt')}")
    print("Done.")
