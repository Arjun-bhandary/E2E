import os, json, time, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import spconv.pytorch as spconv

class Config:
    def __init__(self):
        self.unlabeled_path = '/raid/home/dgx1736/Arush1/Dataset_Specific_Unlabelled.h5'
        self.output_dir = '/raid/home/dgx1736/Arush1/sparse_resnet'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        self.threshold = 0.0
        self.batch_size = 64
        self.epochs = 30
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.masking_ratio = 0.75
        self.occ_weight = 0.5
        self.val_split = 0.2
        self.patience = 7
        self.spatial_size = [125,125]
        self.input_channel = 8
        self.enc_dim = 512
        self.dec_dim = 256
        self.num_workers = 2

config = Config()
UNLABELED_JET_OFFSET = 2048
UNLABELED_N_SAMPLES  = 60000

os.makedirs(config.output_dir, exist_ok=True)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(config.seed)
print(f"Device: {config.device} | Seed: {config.seed} | Mask: {config.masking_ratio}")
print(f"Ablation: NO occupancy head (reconstruction only)")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")


# Since the hdf5 file is storing contigiously 
# to avoid h5py data loading into RAM directly due to dependency issue
# uses lazy loading by reading one sample at a time via open() + seek() + np.frombuffer()
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

# the collate function sparsifies the input by extracting active pixels.
# Constructs coordinate–feature pairs for spconv.
# Partitions them into visible and masked subsets for reconstruction training.
def mae_collate_fn(batch, mask_ratio=config.masking_ratio, threshold=config.threshold):
    vis_coords_list, vis_feats_list = [], []
    mask_coords_list, mask_feats_list = [], []

    for i, img in enumerate(batch):
        active_mask = img.sum(axis=-1) > threshold
        coords = np.argwhere(active_mask)
        n_active = len(coords)
        if n_active == 0:
            coords = np.array([[0, 0]], dtype=np.int32)
            feats = np.zeros((1, config.input_channel), dtype=np.float32)
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

# Resnet block with skip connections (channel dims are changed using 1x1 conv)
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

#Using sparseConv2D for downsampling
class SparseDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, indice_key=None):
        super().__init__()
        self.conv = spconv.SparseConv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False, indice_key=indice_key)
        self.bn = nn.BatchNorm1d(out_ch); self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x); return x.replace_feature(self.relu(self.bn(x.features)))

# This encoder takes a sparse tensor input and progressively compresses it spatially while increasing feature richness, producing a high-level sparse representation
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

# Upsamples sparse latent features back to higher resolution using inverse convolutions and refines them to reconstruct original inputs
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
        self.recon_head = nn.Linear(dec_dim//4, out_channels)

    def forward(self, x):
        x = self.up1(x); x = self.dec1(x)
        x = self.up2(x); x = self.dec2(x)
        x = self.up3(x); x = self.dec3(x)
        return x


class SparseMAE_ReconOnly(nn.Module):
    def __init__(self, in_channels=8, enc_dim=512, dec_dim=256):
        super().__init__()
        self.encoder = SparseResNetEncoder(in_channels, base_ch=enc_dim//8)
        self.decoder = SparseMAEDecoder(enc_dim, dec_dim, in_channels)
        self.mask_token = nn.Parameter(torch.randn(1, in_channels) * 0.02)

    def forward(self, vis_coords, vis_feats, mask_coords, mask_feats, batch_size):
        n_masked = mask_coords.shape[0]
        all_coords = torch.cat([vis_coords, mask_coords], dim=0)
        all_feats = torch.cat([vis_feats, self.mask_token.expand(n_masked, -1)], dim=0)

        x = spconv.SparseConvTensor(all_feats, all_coords, config.spatial_size, batch_size)
        z = self.encoder(x)
        decoded = self.decoder(z)

        rf, _, rqi = _vectorized_lookup(decoded.indices, decoded.features,
                                         mask_coords, vis_feats.device)
        if rf.shape[0] > 0:
            recon_preds = self.decoder.recon_head(rf)
            recon_targets = mask_feats[rqi]
            recon_loss = nn.functional.mse_loss(recon_preds, recon_targets)
        else:
            recon_loss = torch.tensor(0.0, device=vis_feats.device)

        return recon_loss


def train_one_epoch(model, loader, optimizer):
    model.train(); tot, n = 0, 0
    for batch in tqdm(loader, desc='Train', leave=False):
        vc, vf = batch['vis_coords'].to(config.device), batch['vis_feats'].to(config.device)
        mc, mf = batch['mask_coords'].to(config.device), batch['mask_feats'].to(config.device)
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
        vc, vf = batch['vis_coords'].to(config.device), batch['vis_feats'].to(config.device)
        mc, mf = batch['mask_coords'].to(config.device), batch['mask_feats'].to(config.device)
        loss = model(vc, vf, mc, mf, batch['batch_size'])
        tot += loss.item(); n += 1
    return tot / max(n, 1)


if __name__ == '__main__':
    t0 = time.time()
    dataset = UnlabelledSparseJetDataset(config.unlabeled_path, config.threshold)
    n = len(dataset); nv = int(config.val_split * n); nt = n - nv
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [nt, nv], generator=torch.Generator().manual_seed(config.seed))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True,
                              collate_fn=mae_collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=mae_collate_fn, drop_last=False)

    model = SparseMAE_ReconOnly(config.input_channel, config.enc_dim, config.dec_dim).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_val, history, pat = float('inf'), [], 0
    for epoch in range(1, config.epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer)
        vl = evaluate(model, val_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        history.append(dict(epoch=epoch, tr_loss=tl, va_loss=vl, lr=lr))
        if vl < best_val:
            best_val = vl; pat = 0
            torch.save(model.encoder.state_dict(), os.path.join(config.output_dir, 'sparse_mae_encoder.pt'))
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'sparse_mae_full.pt'))
        else: pat += 1
        if pat >= config.patience:
            break

    with open(os.path.join(config.output_dir, 'pretrain_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
