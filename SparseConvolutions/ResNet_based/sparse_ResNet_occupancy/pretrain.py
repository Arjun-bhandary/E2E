import os, json, time, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import spconv.pytorch as spconv

class Config:
  def __init__(self):
    self.unlabeled_path = '/raid/home/dgx1736/Arush1/Dataset_Specific_Unlabelled.h5'
    self.output_dir = '/raid/home/dgx1736/Arush1/sparse_resnet'
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.batch_size = 32
    self.epochs = 30
    self.lr = 1e-2
    self.weight_decay = 1e-4
    self.masking_ratio = 0.75
    self.occ_weight = 0.5
    self.val_split = 0.2
    self.patience = 7
    self.spatial_size = [125,125]
    self.input_channel = 8
    self.enc_dim = 512
    self.dec_dim = 256
    config.num_workers = 2



UNLABELED_JET_OFFSET = 2048
UNLABELED_N_SAMPLES  = 60000
config = Config()
os.makedirs(config.output_dir, exist_ok=True)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(config.seed)
print(f"Device: {config.device} | Seed: {config.seed} | Mask: {config.masking_ratio}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")


# Since the hdf5 file is storing contigiously 
# to avoid h5py data loading into RAM directly due to dependency issue
# uses lazy loading by reading one sample at a time via open() + seek() + np.frombuffer()
class pretrainingDataset(Dataset):
    def __init__(self, path, threshold=0.0):
        self.path = path
        self.threshold = threshold
        self.n_samples = UNLABELED_N_SAMPLES
        self.sample_bytes = 125 * 125 * 8 * 4
        print(f"Dataset: {self.n_samples} unlabeled samples")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with open(self.path, 'rb') as f:
            f.seek(UNLABELED_JET_OFFSET + idx * self.sample_bytes)
            raw = f.read(self.sample_bytes)
        img = np.frombuffer(raw, dtype=np.float32).reshape(125, 125, 8).copy()
        img /= 255.0 #noramlising data
        return img

# I have used the general format in which spconv also stores data
def collate_fn(batch, mask_ratio=config.masking_ratio, threshold=0.0):
    vis_coords, vis_features = [], [] # visible unmasked coords and feature vectors
    mask_coords, mask_features = [], [] # masked coords and feature vectors
    occ_coords, occ_labels = [], [] # occupancy coords and label vectors

    for i, img in enumerate(batch):
        active_mask = img.sum(axis=-1) > threshold
        coords = np.argwhere(active_mask)
        n_active = len(coords)
        if n_active == 0:
            coords = np.array([[0, 0]], dtype=np.int32)
            features = np.zeros((1, config.input_channel), dtype=np.float32)
            n_active = 1
        else:
            features = img[coords[:, 0], coords[:, 1], :]
            coords = coords.astype(np.int32)

        n_mask = max(1, int(n_active * mask_ratio))
        perm = np.random.permutation(n_active)
        vis_idx, mask_idx = perm[:n_active - n_mask], perm[n_active - n_mask:]

        bc = np.full((len(vis_idx), 1), i, dtype=np.int32)
        vis_coords.append(np.hstack([bc, coords[vis_idx]]))
        vis_features.append(features[vis_idx])

        bc_m = np.full((len(mask_idx), 1), i, dtype=np.int32)
        mask_coords.append(np.hstack([bc_m, coords[mask_idx]]))
        mask_features.append(features[mask_idx])

        # occupancy
        n_occ = min(n_active, 200)
        pos_idx = np.random.choice(n_active, size=n_occ, replace=(n_active < n_occ))
        active_set = set(map(tuple, coords.tolist()))
        neg_coords = []
        for _ in range(n_occ * 10):
            r, c = np.random.randint(0, 125), np.random.randint(0, 125)
            if (r, c) not in active_set:
                neg_coords.append([r, c])
            if len(neg_coords) >= n_occ: break
        while len(neg_coords) < n_occ:
            neg_coords.append([np.random.randint(0, 125), np.random.randint(0, 125)])
        neg_coords = np.array(neg_coords, dtype=np.int32)

        occ_c = np.vstack([coords[pos_idx], neg_coords])
        occ_l = np.concatenate([np.ones(n_occ), np.zeros(n_occ)]).astype(np.float32)
        bc_o = np.full((len(occ_c), 1), i, dtype=np.int32)
        occ_coords.append(np.hstack([bc_o, occ_c]))
        occ_labels.append(occ_l)

    return {
        'vis_coords':  torch.from_numpy(np.vstack(vis_coords)).int(),
        'vis_feats':   torch.from_numpy(np.vstack(vis_features)).float(),
        'mask_coords': torch.from_numpy(np.vstack(mask_coords)).int(),
        'mask_feats':  torch.from_numpy(np.vstack(mask_features)).float(),
        'occ_coords':  torch.from_numpy(np.vstack(occ_coords)).int(),
        'occ_labels':  torch.from_numpy(np.concatenate(occ_labels)).float(),
        'batch_size':  len(batch),
    }

# Resnet block with skip connections (channel dims are changed using 1x1 conv)
class SparseResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, indice_key=None):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = spconv.SubMConv2d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv2d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key)
        self.skip = (spconv.SubMConv2d(in_ch, out_ch, 1, bias=False, indice_key=indice_key + '_skip')
                     if in_ch != out_ch else None)

    def forward(self, x): 
        identity = x
        # x.feature -> (n_active,channel) x.indices -> (n_active,3)
        out = x.replace_feature(self.relu1(self.bn1(x.features)))
        out = self.conv1(out)
        out = out.replace_feature(self.relu2(self.bn2(out.features)))
        out = self.conv2(out)
        if self.skip: identity = self.skip(identity)
        return out.replace_feature(out.features + identity.features)

#Using sparseConv2D for downsampling
class SparseDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, indice_key=None):
        super().__init__()
        self.conv = spconv.SparseConv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False, indice_key=indice_key)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.relu(self.bn(x.features)))


class SparseEncoder(nn.Module):
    def __init__(self, in_channels=8, base_ch=64):
        super().__init__()
        self.stem = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, base_ch, 3, padding=1, bias=False, indice_key='stem'),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True))
        self.stage1 = nn.ModuleList([
            SparseResBlock(base_ch, base_ch, indice_key='s1'),
            SparseResBlock(base_ch, base_ch, indice_key='s1')])
        self.down1 = SparseDownsample(base_ch, base_ch, indice_key='d1')
        self.stage2 = nn.ModuleList([
            SparseResBlock(base_ch, base_ch*2, indice_key='s2'),
            SparseResBlock(base_ch*2, base_ch*2, indice_key='s2b')])
        self.down2 = SparseDownsample(base_ch*2, base_ch*2, indice_key='d2')
        self.stage3 = nn.ModuleList([
            SparseResBlock(base_ch*2, base_ch*4, indice_key='s3'),
            SparseResBlock(base_ch*4, base_ch*4, indice_key='s3b')])
        self.down3 = SparseDownsample(base_ch*4, base_ch*4, indice_key='d3')
        self.stage4 = nn.ModuleList([
            SparseResBlock(base_ch*4, base_ch*8, indice_key='s4'),
            SparseResBlock(base_ch*8, base_ch*8, indice_key='s4b')])

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


class SparseDecoder(nn.Module):
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
        self.occ_head = nn.Linear(dec_dim//4, 1)

    def forward(self, x):
        x = self.up1(x); x = self.dec1(x)
        x = self.up2(x); x = self.dec2(x)
        x = self.up3(x); x = self.dec3(x)
        return x


def _vectorized_lookup(dec_indices, dec_feats, query_coords, device):
    dec_hash = dec_indices[:, 0].long() * 15625 + dec_indices[:, 1].long() * 125 + dec_indices[:, 2].long()
    query_hash = query_coords[:, 0].long() * 15625 + query_coords[:, 1].long() * 125 + query_coords[:, 2].long()
    max_hash = config.batch_size * 15625 + 1
    lookup = torch.full((max_hash,), -1, dtype=torch.long, device=device)
    lookup[dec_hash.clamp(0, max_hash-1)] = torch.arange(len(dec_hash), device=device)
    matched = lookup[query_hash.clamp(0, max_hash-1)]
    found_mask = matched >= 0
    found_qi = torch.where(found_mask)[0]
    return dec_feats[matched[found_mask]], found_mask, found_qi


class ArjunModel(nn.Module):
    def __init__(self, in_channels=8, enc_dim=512, dec_dim=256):
        super().__init__()
        self.encoder = SparseEncoder(in_channels, base_ch=enc_dim//8)
        self.decoder = SparseDecoder(enc_dim, dec_dim, in_channels)
        self.mask_token = nn.Parameter(torch.randn(1, in_channels) * 0.02)

    def forward(self, vis_coords, vis_feats, mask_coords, mask_feats,
                occ_coords, occ_labels, batch_size):
        n_masked = mask_coords.shape[0]
        all_coords = torch.cat([vis_coords, mask_coords], dim=0)
        all_feats = torch.cat([vis_feats, self.mask_token.expand(n_masked, -1)], dim=0)

        x = spconv.SparseConvTensor(all_feats, all_coords, config.spatial_size, batch_size)
        z = self.encoder(x)
        decoded = self.decoder(z)
        di, df = decoded.indices, decoded.features

        # reconstruction
        rf, _, rqi = _vectorized_lookup(di, df, mask_coords, vis_feats.device)
        recon_loss = (nn.functional.mse_loss(self.decoder.recon_head(rf), mask_feats[rqi])
                      if rf.shape[0] > 0 else torch.tensor(0.0, device=vis_feats.device))

        # occupancy
        of, _, oqi = _vectorized_lookup(di, df, occ_coords, vis_feats.device)
        occ_logits = torch.full((occ_coords.shape[0],), -2.0, device=vis_feats.device)
        if of.shape[0] > 0:
            occ_logits[oqi] = self.decoder.occ_head(of).squeeze(-1)
        occ_loss = nn.functional.binary_cross_entropy_with_logits(occ_logits, occ_labels)

        total = recon_loss + config.occ_weight * occ_loss
        return total, recon_loss.item(), occ_loss.item()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer):
    model.train()
    tot, tr, to, n = 0, 0, 0, 0
    for batch in tqdm(loader, desc='Train', leave=False):
        vc, vf = batch['vis_coords'].to(config.device), batch['vis_feats'].to(config.device)
        mc, mf = batch['mask_coords'].to(config.device), batch['mask_feats'].to(config.device)
        oc, ol = batch['occ_coords'].to(config.device), batch['occ_labels'].to(config.device)
        optimizer.zero_grad()
        loss, rl, ol_ = model(vc, vf, mc, mf, oc, ol, batch['batch_size'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item(); tr += rl; to += ol_; n += 1
    return tot/max(n,1), tr/max(n,1), to/max(n,1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot, tr, to, n = 0, 0, 0, 0
    for batch in tqdm(loader, desc='Val  ', leave=False):
        vc, vf = batch['vis_coords'].to(config.device), batch['vis_feats'].to(config.device)
        mc, mf = batch['mask_coords'].to(config.device), batch['mask_feats'].to(config.device)
        oc, ol = batch['occ_coords'].to(config.device), batch['occ_labels'].to(config.device)
        loss, rl, ol_ = model(vc, vf, mc, mf, oc, ol, batch['batch_size'])
        tot += loss.item(); tr += rl; to += ol_; n += 1
    return tot/max(n,1), tr/max(n,1), to/max(n,1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INDIVIDUAL PLOT SAVING
# ═══════════════════════════════════════════════════════════════════════════════
def save_total_loss_plot(history, path):
    epochs = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_loss'] for h in history], 'b-', label='Train')
    ax.plot(epochs, [h['va_loss'] for h in history], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
    ax.set_title('Sparse ResNet MAE: Total Loss (Recon + λ·Occ)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

def save_recon_loss_plot(history, path):
    epochs = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_recon'] for h in history], 'b-', label='Train')
    ax.plot(epochs, [h['va_recon'] for h in history], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Sparse ResNet MAE: Reconstruction Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

def save_occ_loss_plot(history, path):
    epochs = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_occ'] for h in history], 'b-', label='Train')
    ax.plot(epochs, [h['va_occ'] for h in history], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Occupancy BCE')
    ax.set_title('Sparse ResNet MAE: Occupancy Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    dataset = pretrainingDataset(config.unlabeled_path)
    n = len(dataset); nv = int(config.val_split * n); nt = n - nv
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [nt, nv], generator=torch.Generator().manual_seed(config.seed))
    print(f"Train: {nt}  Val: {nv}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=collate_fn, drop_last=False)

    model = ArjunModel(config.input_channel, config.enc_dim, config.dec_dim).to(config.device)
    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ne_ = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Total params: {np_:,} | Encoder: {ne_:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    print(f"\n{'Ep':<5} {'TrLoss':<9} {'TrRec':<9} {'TrOcc':<9} "
          f"{'VaLoss':<9} {'VaRec':<9} {'VaOcc':<9} {'config.lr':<10}")
    print("-" * 75)

    best_val, history, pat = float('inf'), [], 0
    for epoch in range(1, config.epochs + 1):
        tl, tr, to = train_one_epoch(model, train_loader, optimizer)
        vl, vr, vo = evaluate(model, val_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        history.append(dict(epoch=epoch, tr_loss=tl, tr_recon=tr, tr_occ=to,
                            va_loss=vl, va_recon=vr, va_occ=vo, lr=lr))
        mk = ''
        if vl < best_val:
            best_val = vl; pat = 0
            torch.save(model.encoder.state_dict(), os.path.join(config.output_dir, 'sparse_mae_encoder.pt'))
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'sparse_mae_full.pt'))
            mk = ' ← best'
        else: pat += 1
        print(f"{epoch:<5} {tl:<9.4f} {tr:<9.4f} {to:<9.4f} "
              f"{vl:<9.4f} {vr:<9.4f} {vo:<9.4f} {lr:<10.2e}{mk}")
        if pat >= config.patience:
            print(f"\nEarly stopping at epoch {epoch}"); break

    elapsed = time.time() - t0
    print(f"\nBest Val Loss: {best_val:.6f} | Time: {elapsed/3600:.2f}h")

    with open(os.path.join(config.output_dir, 'pretrain_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    save_total_loss_plot(history, os.path.join(config.output_dir, 'pretrain_total_loss.jpg'))
    save_recon_loss_plot(history, os.path.join(config.output_dir, 'pretrain_recon_loss.jpg'))
    save_occ_loss_plot(history, os.path.join(config.output_dir, 'pretrain_occ_loss.jpg'))

    print(f"Encoder saved to: {os.path.join(config.output_dir, 'sparse_mae_encoder.pt')}")
    print("Done.")
