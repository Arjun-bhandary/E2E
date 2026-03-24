
import os, json, time, random, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Config:
    def __init__(self):
        self.config.unlabeled_path = '/raid/home/dgx1736/Arush1/Dataset_Specific_Unlabelled.h5'
        self.save_dir = '/raid/home/dgx1736/Arush1/sparse_vit'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        self.threshold = 0.0
        self.batch_size = 64
        self.epochs=50
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.mask_ratio = 0.75
        self.in_channel = 8
        self.max_tokens = 1024
        self.embed_dim = 256
        self.N_head = 8
        self.enc_layer = 6
        self.ffn_dim = 1024
        self.dropout = 0.1
        self.val_split = 0.2
        self.num_workers = 2
        self.patience = 10

UNLABELED_JET_OFFSET = 2048
UNLABELED_N_SAMPLES  = 60000
config = Config()
os.makedirs(config.save_dir, exist_ok=True)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(self.seed)
print(f"Device: {self.device} | Seed: {self.seed} | Mask: {self.mask_ratio}")
print(f"Architecture: Sparse ViT v2 ({self.enc_layer}L, dim={self.embed_dim}, max_tok={self.max_tokens})")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")


class UnlabelledTokenDataset(Dataset):
    def __init__(self, path, threshold=0.0, max_tokens=1024):
        self.path, self.threshold, self.max_tokens = path, threshold, max_tokens
        self.n_samples = UNLABELED_N_SAMPLES
        self.sample_bytes = 125 * 125 * 8 * 4
        print(f"Dataset: {self.n_samples} samples (max {max_tokens} tokens)")

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        with open(self.path, 'rb') as f:
            f.seek(UNLABELED_JET_OFFSET + idx * self.sample_bytes)
            raw = f.read(self.sample_bytes)
        img = np.frombuffer(raw, dtype=np.float32).reshape(125, 125, 8).copy() / 255.0

        coords = np.argwhere(img.sum(axis=-1) > self.threshold)
        n = len(coords)
        if n == 0:
            coords = np.array([[0, 0]], dtype=np.int32)
            feats = np.zeros((1, self.in_channel), dtype=np.float32); n = 1
        else:
            feats = img[coords[:, 0], coords[:, 1], :]

        if n > self.max_tokens:
            sel = np.random.choice(n, self.max_tokens, replace=False)
            coords, feats = coords[sel], feats[sel]; n = self.max_tokens

        positions = np.zeros((self.max_tokens, 2), dtype=np.float32)
        features = np.zeros((self.max_tokens, self.in_channel), dtype=np.float32)
        n_real = min(n, self.max_tokens)
        positions[:n_real] = coords[:n_real].astype(np.float32)
        features[:n_real] = feats[:n_real]

        return (torch.FloatTensor(positions), torch.FloatTensor(features),
                torch.tensor(n_real, dtype=torch.long))

def collate_fn(batch):
    pos, feat, lengths = zip(*batch)
    return torch.stack(pos), torch.stack(feat), torch.stack(lengths)


class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        half_d = d_model // 2
        freq = torch.exp(torch.arange(0, half_d, 2).float() * -(math.log(10000.0) / half_d))
        self.register_buffer('freq', freq)

    def forward(self, positions):
        B, T, _ = positions.shape
        half_d = self.d_model // 2
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
        """
        positions: (B, T, 2)
        features:  (B, T, 8)
        pad_mask:  (B, T) bool — True = IGNORE (padded)
        Returns: (B, T+1, D) all tokens including CLS
        """
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
        """Returns just the CLS token."""
        return self.forward(positions, features, pad_mask)[:, 0, :]


class SparseViTMAE(nn.Module):
    def __init__(self, in_channels=8, d_model=256, n_heads=8,
                 enc_layers=6, ffn_dim=1024, dropout=0.1, max_tokens=1024):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens

        self.encoder = SparseViTEncoder(in_channels, d_model, n_heads,
                                         enc_layers, ffn_dim, dropout)

        # lightweight MLP decoder (not transformer — much faster)
        self.decoder_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, in_channels))

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, positions, features, lengths, mask_ratio=0.75):
        """
        Fully vectorized — no per-sample Python loops.

        positions: (B, T, 2)
        features:  (B, T, 8)
        lengths:   (B,) number of real tokens per sample
        """
        B, T, C = features.shape
        device = features.device

        # build real-token mask: (B, T) — True where real
        idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        real_mask = idx < lengths.unsqueeze(1)              # (B, T)

        # for each sample, generate random noise and use it to select
        # which real tokens get masked
        noise = torch.rand(B, T, device=device)
        noise[~real_mask] = 2.0  # push pad tokens to end

        # sort noise — positions with smallest noise = visible
        sorted_indices = noise.argsort(dim=1)

        # number of visible tokens per sample
        n_vis = (lengths.float() * (1 - mask_ratio)).long().clamp(min=1)
        max_vis = n_vis.max().item()

        # visible token indices (first max_vis of sorted)
        vis_indices = sorted_indices[:, :max_vis]  # (B, max_vis)

        # gather visible tokens
        vis_pos = positions.gather(1, vis_indices.unsqueeze(-1).expand(-1, -1, 2))
        vis_feat = features.gather(1, vis_indices.unsqueeze(-1).expand(-1, -1, C))

        # pad mask for visible tokens: True = IGNORE
        vis_real = torch.arange(max_vis, device=device).unsqueeze(0) < n_vis.unsqueeze(1)
        vis_pad = ~vis_real  # (B, max_vis)

        # encode visible tokens only
        encoded = self.encoder(vis_pos, vis_feat, vis_pad)  # (B, max_vis+1, D)
        # skip CLS token at position 0
        vis_encoded = encoded[:, 1:, :]  # (B, max_vis, D)

        # now build full sequence for decoding
        # start with mask tokens everywhere
        all_tokens = self.mask_token.expand(B, T, -1).clone()

        # scatter visible encoded tokens back to their original positions
        scatter_idx = vis_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        # only scatter real visible tokens (not padding from max_vis)
        vis_mask_3d = vis_real.unsqueeze(-1).expand(-1, -1, self.d_model)
        vis_encoded_masked = vis_encoded * vis_mask_3d.float()
        all_tokens.scatter_(1, scatter_idx, vis_encoded_masked)

        # add positional encoding to all tokens
        all_tokens = all_tokens + self.encoder.pos_enc(positions)

        # decode with MLP head (no transformer decoder — fast)
        recon = self.decoder_head(all_tokens)  # (B, T, 8)

        # compute loss only at masked real positions
        # masked = real AND NOT visible
        vis_bool = torch.zeros(B, T, dtype=torch.bool, device=device)
        vis_bool.scatter_(1, vis_indices, vis_real)
        masked_real = real_mask & ~vis_bool  # (B, T) — True = masked real token

        if masked_real.any():
            pred = recon[masked_real]           # (N_masked, 8)
            target = features[masked_real]      # (N_masked, 8)
            loss = nn.functional.mse_loss(pred, target)
        else:
            loss = torch.tensor(0.0, device=device)

        return loss



def train_one_epoch(model, loader, optimizer):
    model.train(); tot, n = 0, 0
    for pos, feat, lengths in tqdm(loader, desc='Train', leave=False):
        pos, feat, lengths = pos.to(self.device), feat.to(self.device), lengths.to(self.device)
        optimizer.zero_grad()
        loss = model(pos, feat, lengths, self.mask_ratio)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item(); n += 1
    return tot / max(n, 1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); tot, n = 0, 0
    for pos, feat, lengths in tqdm(loader, desc='Val  ', leave=False):
        pos, feat, lengths = pos.to(self.device), feat.to(self.device), lengths.to(self.device)
        loss = model(pos, feat, lengths, self.mask_ratio)
        tot += loss.item(); n += 1
    return tot / max(n, 1)



def save_loss_plot(history, path):
    epochs = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['tr_loss'] for h in history], 'b-', label='Train')
    ax.plot(epochs, [h['va_loss'] for h in history], 'r-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Sparse ViT MAE: Reconstruction Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    t0 = time.time()
    dataset = UnlabelledTokenDataset(config.unlabeled_path, self.threshold, self.max_tokens)
    n = len(dataset); nv = int(self.val_split * n); nt = n - nv
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [nt, nv], generator=torch.Generator().manual_seed(self.seed))
    print(f"Train: {nt}  Val: {nv}")

    train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, pin_memory=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)

    model = SparseViTMAE(self.in_channel, self.embed_dim, self.N_head, self.enc_layer,
                          self.ffn_dim, self.dropout, self.max_tokens).to(self.device)
    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ne_ = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Total params: {np_:,} | Encoder: {ne_:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

    print(f"\n{'Ep':<5} {'TrLoss':<12} {'VaLoss':<12} {'self.lr':<10}")
    print("-" * 40)

    best_val, history, pat = float('inf'), [], 0
    for epoch in range(1, self.epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer)
        vl = evaluate(model, val_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        history.append(dict(epoch=epoch, tr_loss=tl, va_loss=vl, lr=lr))
        mk = ''
        if vl < best_val:
            best_val = vl; pat = 0
            torch.save(model.encoder.state_dict(),
                       os.path.join(config.save_dir, 'sparse_vit_encoder.pt'))
            torch.save(model.state_dict(),
                       os.path.join(config.save_dir, 'sparse_vit_mae_full.pt'))
            mk = ' ← best'
        else: pat += 1
        print(f"{epoch:<5} {tl:<12.6f} {vl:<12.6f} {lr:<10.2e}{mk}")
        if pat >= self.patience:
            print(f"\nEarly stopping at epoch {epoch}"); break

    print(f"\nBest Val Loss: {best_val:.6f} | Time: {(time.time()-t0)/3600:.2f}h")
    with open(os.path.join(config.save_dir, 'pretrain_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    save_loss_plot(history, os.path.join(config.save_dir, 'pretrain_recon_loss.jpg'))
    print(f"Encoder saved to: {os.path.join(config.save_dir, 'sparse_vit_encoder.pt')}")
    print("Done.")
