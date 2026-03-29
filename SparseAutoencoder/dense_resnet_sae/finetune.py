

import os, json, time, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configs
LABELED_PATH   = '/raid/home/dgx1736/Arush1/Dataset_Specific_labelled.h5'
ENCODER_PATH   = '/raid/home/dgx1736/Arush1/dense_resnet_sae/dense_sae_encoder.pt'
SAVE_DIR       = '/raid/home/dgx1736/Arush1/dense_resnet_sae'
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED           = 42
BATCH_SIZE     = 64
EPOCHS         = 40
LR_HEAD        = 1e-3
LR_ENCODER     = 5e-5
FREEZE_EPOCHS  = 5
WEIGHT_DECAY   = 1e-3
DROPOUT        = 0.5
IN_CHANNELS    = 8
VAL_SPLIT      = 0.2
PATIENCE       = 6

LABELED_JET_OFFSET = 2048
LABELED_Y_OFFSET   = 5000002048
LABELED_N_SAMPLES  = 10000

os.makedirs(SAVE_DIR, exist_ok=True)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(SEED)
print(f"Device: {DEVICE} | Seed: {SEED}")
print(f"Architecture: Dense ResNet-18 SAE (L1+KL pretrained)")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")


# Dataset with lazy loader
class LabelledDenseJetDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.n_samples = LABELED_N_SAMPLES
        self.sample_bytes = 125 * 125 * 8 * 4
        with open(path, 'rb') as f:
            f.seek(LABELED_Y_OFFSET)
            self.Y = np.frombuffer(f.read(self.n_samples * 4), dtype=np.float32).copy()
        print(f"Labels loaded: {self.n_samples} | Balance: {self.Y.mean():.3f}")

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        with open(self.path, 'rb') as f:
            f.seek(LABELED_JET_OFFSET + idx * self.sample_bytes)
            raw = f.read(self.sample_bytes)
        img = np.frombuffer(raw, dtype=np.float32).reshape(125, 125, 8).copy() / 255.0
        # HWC → CHW
        x = torch.from_numpy(img).permute(2, 0, 1).float()
        y = torch.FloatTensor([self.Y[idx]])
        return x, y


# ResNet-18 encoder + classifier head
class ResBlock(nn.Module):
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
    def __init__(self, in_channels=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return x.flatten(1)  # (B, 512)


class DenseResNetClassifier(nn.Module):
    def __init__(self, in_channels=8, dropout=0.5):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1))

    def forward(self, x):
        features = self.encoder(x)  # (B, 512)
        return self.classifier(features)

    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False
        print("  → Encoder FROZEN")
    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True
        print("  → Encoder UNFROZEN")


# metrics and train/eval
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
    for x, labels in tqdm(loader, desc='Train', leave=False):
        x, labels = x.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.squeeze(), labels.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item()
        ap.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        al.extend(labels.cpu().numpy().flatten())
    return tot/len(loader), roc_auc_score(al, ap), accuracy_score(al, (np.array(ap)>=0.5).astype(int))


@torch.no_grad()
def evaluate_full(model, loader, criterion):
    model.eval(); tot, ap, al = 0, [], []
    for x, labels in tqdm(loader, desc='Val  ', leave=False):
        x, labels = x.to(DEVICE), labels.to(DEVICE)
        logits = model(x)
        tot += criterion(logits.squeeze(), labels.squeeze()).item()
        ap.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        al.extend(labels.cpu().numpy().flatten())
    return tot/len(loader), compute_metrics(al, ap), al, ap


# Saving plots
def save_roc_plot(fpr, tpr, auc_val, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc_val:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    idx = np.argmin(np.abs(tpr - 0.7))
    ax.plot(fpr[idx], tpr[idx], 'ro', ms=8, label=f'TPR=0.7 → 1/FPR={1/max(fpr[idx],1e-10):.1f}')
    ax.axhline(0.7, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC — Dense ResNet-18 SAE'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_loss_plot(history, path):
    eps = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, [h['tr_loss'] for h in history], 'b-o', ms=3, label='Train')
    ax.plot(eps, [h['va_loss'] for h in history], 'r-o', ms=3, label='Val')
    ax.axvline(FREEZE_EPOCHS+0.5, color='green', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
    ax.set_title('Loss (Dense ResNet-18 SAE)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_auc_plot(history, path):
    eps = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, [h['tr_auc'] for h in history], 'b-o', ms=3, label='Train')
    ax.plot(eps, [h['va_auc'] for h in history], 'r-o', ms=3, label='Val')
    ax.axvline(FREEZE_EPOCHS+0.5, color='green', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_xlabel('Epoch'); ax.set_ylabel('AUC')
    ax.set_title('ROC-AUC (Dense ResNet-18 SAE)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_acc_plot(history, path):
    eps = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, [h['tr_acc'] for h in history], 'b-o', ms=3, label='Train')
    ax.plot(eps, [h['va_acc'] for h in history], 'r-o', ms=3, label='Val')
    ax.axvline(FREEZE_EPOCHS+0.5, color='green', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy (Dense ResNet-18 SAE)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_cm_plot(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Dense ResNet-18 SAE)')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Class 0','Class 1']); ax.set_yticklabels(['Class 0','Class 1'])
    for ii in range(2):
        for jj in range(2):
            c = 'white' if cm[ii][jj] > cm.max()/2 else 'black'
            ax.text(jj, ii, f'{cm[ii][jj]}', ha='center', va='center',
                    fontsize=16, fontweight='bold', color=c)
    plt.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


if __name__ == '__main__':
    t0 = time.time()
    dataset = LabelledDenseJetDataset(LABELED_PATH)
    n = len(dataset); nv = int(VAL_SPLIT * n); nt = n - nv
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [nt, nv], generator=torch.Generator().manual_seed(SEED))
    print(f"Train: {nt}  Val: {nv}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = DenseResNetClassifier(IN_CHANNELS, DROPOUT).to(DEVICE)
    if os.path.exists(ENCODER_PATH):
        ckpt = torch.load(ENCODER_PATH, map_location=DEVICE, weights_only=True)
        m, u = model.encoder.load_state_dict(ckpt, strict=False)
        print(f"Loaded encoder (Dense ResNet-18 SAE) | Missing: {len(m)} | Unexpected: {len(u)}")
        if not m and not u: print("  All keys matched.")
    else:
        print(f"WARNING: No encoder at {ENCODER_PATH}, training from scratch")

    criterion = nn.BCEWithLogitsLoss()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Params: {total_params:,} | Encoder: {enc_params:,}")

    history, best_auc, pat, phase = [], 0, 0, 'head-only'
    print(f"\n{'Ep':<5} {'TrL':<8} {'TrAUC':<8} {'TrAcc':<8} {'VaL':<8} {'VaAUC':<8} {'VaAcc':<8} {'Phase'}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        if epoch == 1:
            model.freeze_encoder()
            opt = torch.optim.AdamW(model.classifier.parameters(),
                                    lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=FREEZE_EPOCHS, eta_min=1e-5)
        if epoch == FREEZE_EPOCHS + 1:
            model.unfreeze_encoder(); phase = 'full'; pat = 0
            opt = torch.optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': LR_ENCODER},
                {'params': model.classifier.parameters(), 'lr': LR_HEAD}],
                weight_decay=WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=EPOCHS - FREEZE_EPOCHS, eta_min=1e-6)

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
                        'val_metrics': {k: v for k, v in vm.items() if k not in ('fpr', 'tpr')}},
                       os.path.join(SAVE_DIR, 'finetuned_classifier.pt'))
            mk = ' ← best'
        elif epoch > FREEZE_EPOCHS:
            pat += 1
        print(f"{epoch:<5} {tl:<8.4f} {ta:<8.4f} {tac:<8.4f} {vl:<8.4f} {va:<8.4f} {vac:<8.4f} {phase}{mk}")
        if pat >= PATIENCE and epoch > FREEZE_EPOCHS:
            print(f"\nEarly stopping at epoch {epoch}"); break

    # ── final eval ────────────────────────────────────────────────────────
    print("\n" + "=" * 50 + "\nFINAL EVALUATION (Dense ResNet-18 SAE)\n" + "=" * 50)
    ckpt = torch.load(os.path.join(SAVE_DIR, 'finetuned_classifier.pt'),
                       map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    _, fm, _, _ = evaluate_full(model, val_loader, criterion)

    print(f"  AUC: {fm['auc']:.4f} | Acc: {fm['accuracy']:.4f} | F1: {fm['f1']:.4f}")
    print(f"  Precision: {fm['precision']:.4f} | Recall: {fm['recall']:.4f}")
    print(f"  1/FPR@0.7: {fm['inv_fpr_at_tpr07']:.1f}")
    print(f"\n  Baseline (Sparse ResNet MAE):   AUC = 0.9602")
    print(f"  Dense ResNet-18 SAE (this):     AUC = {fm['auc']:.4f}")
    print(f"  Δ (this - baseline):            {fm['auc'] - 0.9602:+.4f}")

    save_roc_plot(fm['fpr'], fm['tpr'], fm['auc'], os.path.join(SAVE_DIR, 'roc_curve.jpg'))
    save_loss_plot(history, os.path.join(SAVE_DIR, 'loss_curve.jpg'))
    save_auc_plot(history, os.path.join(SAVE_DIR, 'auc_curve.jpg'))
    save_acc_plot(history, os.path.join(SAVE_DIR, 'accuracy_curve.jpg'))
    save_cm_plot(np.array(fm['confusion_matrix']), os.path.join(SAVE_DIR, 'confusion_matrix.jpg'))

    with open(os.path.join(SAVE_DIR, 'finetune_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    summary = dict(best_epoch=ckpt['epoch'], auc=fm['auc'], accuracy=fm['accuracy'],
                   f1=fm['f1'], inv_fpr_07=fm['inv_fpr_at_tpr07'],
                   confusion_matrix=fm['confusion_matrix'],
                   architecture='dense_resnet18_sparse_autoencoder',
                   comparison={'baseline_mae_auc': 0.9602,
                               'delta': round(fm['auc'] - 0.9602, 4)})
    with open(os.path.join(SAVE_DIR, 'finetune_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min | All saved to {SAVE_DIR}")
    print("Done.")
