
import os, json, time, random, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Configs:
    labeled_path       = '/raid/home/dgx1736/Arush1/Dataset_Specific_labelled.h5'
    encoder_path       = '/raid/home/dgx1736/Arush1/sparse_vit/sparse_vit_encoder.pt'
    save_dir           = '/raid/home/dgx1736/Arush1/sparse_vit'
    device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed               = 42
    threshold          = 0.0
    batch_size         = 32
    epochs             = 40
    lr_head            = 1e-3
    lr_encoder         = 5e-5
    freeze_epochs      = 5
    weight_decay       = 1e-3
    dropout_cls        = 0.5
    in_channels        = 8
    max_tokens         = 1024
    embed_dim          = 256
    n_heads            = 8
    enc_layers         = 6
    ffn_dim            = 1024
    dropout            = 0.1
    val_split          = 0.2
    patience           = 6

    labeled_jet_offset = 2048
    labeled_y_offset   = 5000002048
    labeled_n_samples  = 10000

os.makedirs(Configs.save_dir, exist_ok=True)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(Configs.seed)
print(f"Device: {Configs.device} | Seed: {Configs.seed}")
print(f"Architecture: Sparse ViT v2 classifier (max_tok={Configs.max_tokens})")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")


class LabelledTokenDataset(Dataset):
    def __init__(self, path, threshold=0.0, max_tokens=1024):
        self.path, self.threshold, self.max_tokens = path, threshold, max_tokens
        self.n_samples = Configs.labeled_n_samples
        self.sample_bytes = 125 * 125 * 8 * 4
        with open(path, 'rb') as f:
            f.seek(Configs.labeled_y_offset)
            self.Y = np.frombuffer(f.read(self.n_samples * 4), dtype=np.float32).copy()
        print(f"Labels: {self.n_samples} | Balance: {self.Y.mean():.3f}")

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        with open(self.path, 'rb') as f:
            f.seek(Configs.labeled_jet_offset + idx * self.sample_bytes)
            raw = f.read(self.sample_bytes)
        img = np.frombuffer(raw, dtype=np.float32).reshape(125, 125, 8).copy() / 255.0
        coords = np.argwhere(img.sum(axis=-1) > self.threshold)
        n = len(coords)
        if n == 0:
            coords = np.array([[0, 0]], dtype=np.int32)
            feats = np.zeros((1, Configs.in_channels), dtype=np.float32); n = 1
        else:
            feats = img[coords[:, 0], coords[:, 1], :]

        if n > self.max_tokens:
            sel = np.random.choice(n, self.max_tokens, replace=False)
            coords, feats = coords[sel], feats[sel]; n = self.max_tokens

        positions = np.zeros((self.max_tokens, 2), dtype=np.float32)
        features = np.zeros((self.max_tokens, Configs.in_channels), dtype=np.float32)
        n_real = min(n, self.max_tokens)
        positions[:n_real] = coords[:n_real].astype(np.float32)
        features[:n_real] = feats[:n_real]

        # pad_mask: True = IGNORE
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
        self.encoder = SparseViTEncoder(in_channels, d_model, n_heads,
                                         n_layers, ffn_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_cls),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(dropout_cls * 0.5),
            nn.Linear(64, 1))

    def forward(self, positions, features, pad_mask):
        cls_feat = self.encoder.forward_cls(positions, features, pad_mask)
        return self.classifier(cls_feat)

    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False
        print("  → Encoder FROZEN")
    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True
        print("  → Encoder UNFROZEN")


def compute_metrics(labels, probs):
    auc = roc_auc_score(labels, probs)
    fpr, tpr, th = roc_curve(labels, probs)
    inv_fpr = None
    for i in range(len(tpr)):
        if tpr[i] >= 0.7: inv_fpr = 1.0 / max(fpr[i], 1e-10); break
    j = np.argmax(tpr - fpr); bt = th[j]
    preds = (np.array(probs) >= bt).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    cm = confusion_matrix(labels, preds)
    return dict(auc=auc, accuracy=acc, precision=prec, recall=rec, f1=f1,
                inv_fpr_at_tpr07=inv_fpr, best_threshold=float(bt),
                confusion_matrix=cm.tolist(), fpr=fpr, tpr=tpr)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train(); tot, ap, al = 0, [], []
    for pos, feat, pad, labels in tqdm(loader, desc='Train', leave=False):
        pos, feat, pad, labels = pos.to(Configs.device), feat.to(Configs.device), pad.to(Configs.device), labels.to(Configs.device)
        optimizer.zero_grad()
        logits = model(pos, feat, pad).squeeze()
        loss = criterion(logits, labels)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        tot += loss.item()
        ap.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        al.extend(labels.cpu().numpy().flatten())
    return tot/len(loader), roc_auc_score(al, ap), accuracy_score(al, (np.array(ap)>=0.5).astype(int))

@torch.no_grad()
def evaluate_full(model, loader, criterion):
    model.eval(); tot, ap, al = 0, [], []
    for pos, feat, pad, labels in tqdm(loader, desc='Val  ', leave=False):
        pos, feat, pad, labels = pos.to(Configs.device), feat.to(Configs.device), pad.to(Configs.device), labels.to(Configs.device)
        logits = model(pos, feat, pad).squeeze()
        tot += criterion(logits, labels).item()
        ap.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        al.extend(labels.cpu().numpy().flatten())
    return tot/len(loader), compute_metrics(al, ap), al, ap



def save_roc_plot(fpr, tpr, auc_val, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc_val:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    idx = np.argmin(np.abs(tpr - 0.7))
    ax.plot(fpr[idx], tpr[idx], 'ro', ms=8, label=f'TPR=0.7 → 1/FPR={1/max(fpr[idx],1e-10):.1f}')
    ax.axhline(0.7, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC — Sparse ViT'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_loss_plot(history, path):
    eps = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, [h['tr_loss'] for h in history], 'b-o', ms=3, label='Train')
    ax.plot(eps, [h['va_loss'] for h in history], 'r-o', ms=3, label='Val')
    ax.axvline(Configs.freeze_epochs+0.5, color='green', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss'); ax.set_title('Loss — Sparse ViT')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_auc_plot(history, path):
    eps = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, [h['tr_auc'] for h in history], 'b-o', ms=3, label='Train')
    ax.plot(eps, [h['va_auc'] for h in history], 'r-o', ms=3, label='Val')
    ax.axvline(Configs.freeze_epochs+0.5, color='green', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_xlabel('Epoch'); ax.set_ylabel('AUC'); ax.set_title('ROC-AUC — Sparse ViT')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_acc_plot(history, path):
    eps = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, [h['tr_acc'] for h in history], 'b-o', ms=3, label='Train')
    ax.plot(eps, [h['va_acc'] for h in history], 'r-o', ms=3, label='Val')
    ax.axvline(Configs.freeze_epochs+0.5, color='green', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Accuracy — Sparse ViT')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_cm_plot(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion Matrix — Sparse ViT')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Class 0','Class 1']); ax.set_yticklabels(['Class 0','Class 1'])
    for ii in range(2):
        for jj in range(2):
            c = 'white' if cm[ii][jj] > cm.max()/2 else 'black'
            ax.text(jj, ii, f'{cm[ii][jj]}', ha='center', va='center', fontsize=16, fontweight='bold', color=c)
    plt.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()



if __name__ == '__main__':
    t0 = time.time()
    dataset = LabelledTokenDataset(Configs.labeled_path, Configs.threshold, Configs.max_tokens)
    n = len(dataset); nv = int(Configs.val_split * n); nt = n - nv
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [nt, nv], generator=torch.Generator().manual_seed(Configs.seed))
    print(f"Train: {nt}  Val: {nv}")

    kw = dict(batch_size=Configs.batch_size, num_workers=0, pin_memory=False, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = SparseViTClassifier(Configs.in_channels, Configs.embed_dim, Configs.n_heads, Configs.enc_layers,
                                 Configs.ffn_dim, Configs.dropout, Configs.dropout_cls).to(Configs.device)
    if os.path.exists(Configs.encoder_path):
        ckpt = torch.load(Configs.encoder_path, map_location=Configs.device, weights_only=True)
        m, u = model.encoder.load_state_dict(ckpt, strict=False)
        print(f"Loaded Sparse ViT encoder | Missing: {len(m)} | Unexpected: {len(u)}")
        if not m and not u: print("  All keys matched.")
    else:
        print(f"WARNING: No encoder at {Configs.encoder_path}, training from scratch")

    criterion = nn.BCEWithLogitsLoss()
    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ne_ = sum(p.numel() for p in model.encoder.parameters())
    print(f"Total: {np_:,} | Encoder: {ne_:,}")

    history, best_auc, pat, phase = [], 0, 0, 'head-only'
    print(f"\n{'Ep':<5} {'TrL':<8} {'TrAUC':<8} {'TrAcc':<8} {'VaL':<8} {'VaAUC':<8} {'VaAcc':<8} {'Phase'}")
    print("-" * 70)

    for epoch in range(1, Configs.epochs + 1):
        if epoch == 1:
            model.freeze_encoder()
            opt = torch.optim.AdamW(model.classifier.parameters(), lr=Configs.lr_head, weight_decay=Configs.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Configs.freeze_epochs, eta_min=1e-5)
        if epoch == Configs.freeze_epochs + 1:
            model.unfreeze_encoder(); phase = 'full'; pat = 0
            opt = torch.optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': Configs.lr_encoder},
                {'params': model.classifier.parameters(), 'lr': Configs.lr_head}], weight_decay=Configs.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Configs.epochs-Configs.freeze_epochs, eta_min=1e-6)

        tl, ta, tac = train_one_epoch(model, train_loader, opt, criterion)
        vl, vm, _, _ = evaluate_full(model, val_loader, criterion)
        sched.step()
        va, vac = vm['auc'], vm['accuracy']
        history.append(dict(epoch=epoch, phase=phase, tr_loss=tl, tr_auc=ta, tr_acc=tac,
                            va_loss=vl, va_auc=va, va_acc=vac, va_f1=vm['f1'],
                            va_inv_fpr_07=vm['inv_fpr_at_tpr07']))
        mk = ''
        if va > best_auc:
            best_auc = va; pat = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_auc': va,
                        'val_metrics': {k:v for k,v in vm.items() if k not in ('fpr','tpr')}},
                       os.path.join(Configs.save_dir, 'finetuned_classifier.pt'))
            mk = ' ← best'
        elif epoch > Configs.freeze_epochs: pat += 1
        print(f"{epoch:<5} {tl:<8.4f} {ta:<8.4f} {tac:<8.4f} {vl:<8.4f} {va:<8.4f} {vac:<8.4f} {phase}{mk}")
        if pat >= Configs.patience and epoch > Configs.freeze_epochs:
            print(f"\nEarly stopping at epoch {epoch}"); break

    print("\n" + "=" * 50 + "\nFINAL EVALUATION — Sparse ViT\n" + "=" * 50)
    ckpt = torch.load(os.path.join(Configs.save_dir, 'finetuned_classifier.pt'), map_location=Configs.device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    _, fm, _, _ = evaluate_full(model, val_loader, criterion)

    print(f"  AUC: {fm['auc']:.4f} | Acc: {fm['accuracy']:.4f} | F1: {fm['f1']:.4f}")
    print(f"  Precision: {fm['precision']:.4f} | Recall: {fm['recall']:.4f}")
    print(f"  1/FPR@0.7: {fm['inv_fpr_at_tpr07']:.1f}")
    print(f"\n  Sparse ResNet MAE (best): AUC = 0.9609")
    print(f"  This (Sparse ViT):        AUC = {fm['auc']:.4f}")
    print(f"  Δ:                         {fm['auc'] - 0.9609:+.4f}")

    save_roc_plot(fm['fpr'], fm['tpr'], fm['auc'], os.path.join(Configs.save_dir, 'roc_curve.jpg'))
    save_loss_plot(history, os.path.join(Configs.save_dir, 'loss_curve.jpg'))
    save_auc_plot(history, os.path.join(Configs.save_dir, 'auc_curve.jpg'))
    save_acc_plot(history, os.path.join(Configs.save_dir, 'accuracy_curve.jpg'))
    save_cm_plot(np.array(fm['confusion_matrix']), os.path.join(Configs.save_dir, 'confusion_matrix.jpg'))

    with open(os.path.join(Configs.save_dir, 'finetune_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    summary = dict(architecture='Sparse ViT v2 (6L, 256dim, 8heads, 1024tok)',
                   best_epoch=ckpt['epoch'], auc=fm['auc'], accuracy=fm['accuracy'],
                   f1=fm['f1'], inv_fpr_07=fm['inv_fpr_at_tpr07'],
                   confusion_matrix=fm['confusion_matrix'])
    with open(os.path.join(Configs.save_dir, 'finetune_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min | All saved to {Configs.save_dir}")
    print("Done.")
