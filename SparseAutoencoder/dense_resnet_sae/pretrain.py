
import os, json, time, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# configs
UNLABELED_PATH = '/raid/home/dgx1736/Arush1/Dataset_Specific_Unlabelled.h5'
SAVE_DIR       = '/raid/home/dgx1736/Arush1/dense_resnet_sae'
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED           = 42
BATCH_SIZE     = 64
EPOCHS         = 60
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
SPATIAL_SIZE   = (125, 125)
IN_CHANNELS    = 8
BOTTLENECK_DIM = 512
VAL_SPLIT      = 0.2
NUM_WORKERS    = 4
PATIENCE       = 10

# Sparsity with warmup
SPARSITY_LAMBDA = 5e-4
SPARSITY_TARGET = 0.05
SPARSITY_KL_BETA = 5e-4
SPARSITY_WARMUP = 10       # pure recon epochs
SPARSITY_RAMP   = 10       # linear ramp epochs

UNLABELED_JET_OFFSET = 2048
UNLABELED_N_SAMPLES  = 60000

os.makedirs(SAVE_DIR, exist_ok=True)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(SEED)
print(f"Device: {DEVICE} | Seed: {SEED}")
print(f"Architecture: Dense ResNet-18 Sparse Autoencoder")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")


# Dataset 
class UnlabelledDenseJetDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.n_samples = UNLABELED_N_SAMPLES
        self.sample_bytes = 125 * 125 * 8 * 4
        print(f"Dataset: {self.n_samples} samples (dense, lazy reader)")

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        with open(self.path, 'rb') as f:
            f.seek(UNLABELED_JET_OFFSET + idx * self.sample_bytes)
            raw = f.read(self.sample_bytes)
        img = np.frombuffer(raw, dtype=np.float32).reshape(125, 125, 8).copy()
        img /= 255.0
        # HWC → CHW for Conv2d
        return torch.from_numpy(img).permute(2, 0, 1).float()


# ResNet-18 Encoder + Decoder with sparsity
class ResBlock(nn.Module):
    """Standard ResNet basic block."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 encoder for 125×125×8 input.
    Output: (B, 512) global average pooled feature vector.
    
    Stages: 8→64→64→128→256→512 with progressive downsampling.
    Spatial: 125→63→32→16→8→4 → GAP → (B, 512)
    """
    def __init__(self, in_channels=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))  # 125→63→32

        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)        # (B, 64, 32, 32)
        x = self.layer1(x)      # (B, 64, 32, 32)
        x = self.layer2(x)      # (B, 128, 16, 16)
        x = self.layer3(x)      # (B, 256, 8, 8)
        x = self.layer4(x)      # (B, 512, 4, 4)
        x = self.gap(x)         # (B, 512, 1, 1)
        return x.flatten(1)     # (B, 512)

    def forward_spatial(self, x):
        """Return spatial features before GAP (for decoder)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x                # (B, 512, 4, 4)


class ResNetDecoder(nn.Module):
    """
    Mirror decoder: (B, 512, 4, 4) → (B, 8, 125, 125)
    Uses transposed convolutions to upsample.
    """
    def __init__(self, out_channels=8):
        super().__init__()
        self.decoder = nn.Sequential(
            # 4→8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            # 8→16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # 16→32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # 32→64
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # 64→128 (slightly larger than 125, will crop)
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.decoder(x)  # (B, 8, 128, 128)
        # crop to 125×125
        x = x[:, :, :125, :125]
        return x


def kl_divergence_sparsity(activations, target_rho=0.05):
    """KL divergence sparsity penalty on average activations."""
    rho_hat = activations.mean(dim=0).clamp(1e-6, 1 - 1e-6)
    rho = torch.full_like(rho_hat, target_rho)
    kl = rho * torch.log(rho / rho_hat) + \
         (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return kl.mean()


class DenseResNetSAE(nn.Module):
    """
    Dense ResNet-18 Sparse Autoencoder.
    Encoder produces (B, 512) bottleneck with L1+KL sparsity.
    Decoder reconstructs (B, 8, 125, 125).
    """
    def __init__(self, in_channels=8, sparsity_lambda=5e-4,
                 sparsity_target=0.05, sparsity_kl_beta=5e-4,
                 warmup_epochs=10, ramp_epochs=10):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels)
        self.decoder = ResNetDecoder(in_channels)
        # project bottleneck back to spatial for decoder
        self.bottleneck_to_spatial = nn.Sequential(
            nn.Linear(512, 512 * 4 * 4),
            nn.ReLU(inplace=True))
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target
        self.sparsity_kl_beta = sparsity_kl_beta
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

    def _sparsity_weight(self, epoch):
        if epoch <= self.warmup_epochs:
            return 0.0
        elif epoch <= self.warmup_epochs + self.ramp_epochs:
            return (epoch - self.warmup_epochs) / self.ramp_epochs
        return 1.0

    def forward(self, x, epoch=1):
        # Encode to bottleneck
        z = self.encoder(x)  # (B, 512)

        w = self._sparsity_weight(epoch)

        # Sparsity penalties on bottleneck
        if w > 0:
            l1_loss = w * self.sparsity_lambda * z.abs().mean()
            z_act = torch.sigmoid(z)
            kl_loss = w * self.sparsity_kl_beta * kl_divergence_sparsity(z_act, self.sparsity_target)
        else:
            l1_loss = torch.tensor(0.0, device=x.device)
            kl_loss = torch.tensor(0.0, device=x.device)

        # Decode
        spatial = self.bottleneck_to_spatial(z).view(-1, 512, 4, 4)
        recon = self.decoder(spatial)

        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, x)
        total_loss = recon_loss + l1_loss + kl_loss

        return total_loss, recon_loss.item(), \
               l1_loss.item() if torch.is_tensor(l1_loss) else l1_loss, \
               kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss


# train and eval
def train_one_epoch(model, loader, optimizer, epoch=1):
    model.train()
    tot, tot_r, tot_l1, tot_kl, n = 0, 0, 0, 0, 0
    for batch in tqdm(loader, desc='Train', leave=False):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        loss, r, l1, kl = model(batch, epoch=epoch)
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
        batch = batch.to(DEVICE)
        loss, r, l1, kl = model(batch, epoch=epoch)
        tot += loss.item(); tot_r += r; tot_l1 += l1; tot_kl += kl; n += 1
    return tot/max(n,1), tot_r/max(n,1), tot_l1/max(n,1), tot_kl/max(n,1)


#plotting
def save_loss_plots(history, save_dir):
    epochs = [h['epoch'] for h in history]
    wu = SPARSITY_WARMUP + 0.5
    wr = SPARSITY_WARMUP + SPARSITY_RAMP + 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_recon'] for h in history], 'b-', label='Train')
    ax.plot(epochs, [h['va_recon'] for h in history], 'r-', label='Val')
    ax.axvline(wu, color='green', ls='--', alpha=0.5, label='Sparsity start')
    ax.axvline(wr, color='orange', ls='--', alpha=0.5, label='Full sparsity')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Dense ResNet-18 SAE: Reconstruction Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'pretrain_recon_loss.jpg'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_total'] for h in history], 'b-', label='Train')
    ax.plot(epochs, [h['va_total'] for h in history], 'r-', label='Val')
    ax.axvline(wu, color='green', ls='--', alpha=0.5, label='Sparsity start')
    ax.axvline(wr, color='orange', ls='--', alpha=0.5, label='Full sparsity')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
    ax.set_title('Dense ResNet-18 SAE: Total Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'pretrain_total_loss.jpg'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_l1'] for h in history], 'b-', label='Train L1')
    ax.plot(epochs, [h['tr_kl'] for h in history], 'g-', label='Train KL')
    ax.plot(epochs, [h['va_l1'] for h in history], 'b--', label='Val L1')
    ax.plot(epochs, [h['va_kl'] for h in history], 'g--', label='Val KL')
    ax.axvline(wu, color='green', ls='--', alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Sparsity Penalty')
    ax.set_title('Dense ResNet-18 SAE: Sparsity Losses')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'pretrain_sparsity_loss.jpg'), dpi=150); plt.close()

    print(f"Saved plots to {save_dir}")



if __name__ == '__main__':
    t0 = time.time()
    dataset = UnlabelledDenseJetDataset(UNLABELED_PATH)
    n = len(dataset); nv = int(VAL_SPLIT * n); nt = n - nv
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [nt, nv], generator=torch.Generator().manual_seed(SEED))
    print(f"Train: {nt}  Val: {nv}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)

    model = DenseResNetSAE(
        IN_CHANNELS,
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
                       os.path.join(SAVE_DIR, 'dense_sae_encoder.pt'))
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, 'dense_sae_full.pt'))
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
    print(f"Encoder saved to: {os.path.join(SAVE_DIR, 'dense_sae_encoder.pt')}")
    print("Done.")
