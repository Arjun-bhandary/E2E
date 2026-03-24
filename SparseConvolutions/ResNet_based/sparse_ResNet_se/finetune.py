import os
import json
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import spconv.pytorch as spconv


class Configs:
    LABELED_PATH       = '/raid/home/dgx1736/Arush1/Dataset_Specific_labelled.h5'
    ENCODER_PATH       = '/raid/home/dgx1736/Arush1/sparse_resnet_se/sparse_mae_encoder.pt'
    SAVE_DIR           = '/raid/home/dgx1736/Arush1/sparse_resnet_se'
    DEVICE             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED               = 42
    THRESHOLD          = 0.0
    BATCH_SIZE         = 64
    EPOCHS             = 30
    LR_HEAD            = 1e-3
    LR_ENCODER         = 5e-5
    FREEZE_EPOCHS      = 5
    WEIGHT_DECAY       = 1e-3
    DROPOUT            = 0.5
    SPATIAL_SIZE       = [125, 125]
    IN_CHANNELS        = 8
    ENCODER_DIM        = 512
    VAL_SPLIT          = 0.2
    PATIENCE           = 6
    LABELED_JET_OFFSET = 2048
    LABELED_Y_OFFSET   = 5000002048
    LABELED_N_SAMPLES  = 10000


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabelledSparseJetDataset(Dataset):
    def __init__(self, path, threshold=0.0):
        self.path = path
        self.threshold = threshold
        self.n_samples = Configs.LABELED_N_SAMPLES
        self.sample_bytes = 125 * 125 * 8 * 4
        with open(path, 'rb') as f:
            f.seek(Configs.LABELED_Y_OFFSET)
            self.Y = np.frombuffer(f.read(self.n_samples * 4), dtype=np.float32).copy()

    def __len__(self): 
        return self.n_samples

    def __getitem__(self, idx):
        with open(self.path, 'rb') as f:
            f.seek(Configs.LABELED_JET_OFFSET + idx * self.sample_bytes)
            raw = f.read(self.sample_bytes)
        img = np.frombuffer(raw, dtype=np.float32).reshape(125, 125, 8).copy() / 255.0
        y = self.Y[idx]
        coords = np.argwhere(img.sum(axis=-1) > self.threshold)
        
        if len(coords) == 0:
            coords = np.array([[0, 0]], dtype=np.int32)
            feats = np.zeros((1, Configs.IN_CHANNELS), dtype=np.float32)
        else:
            feats = img[coords[:, 0], coords[:, 1], :]
            coords = coords.astype(np.int32)
            
        return {'coords': torch.IntTensor(coords), 'feats': torch.FloatTensor(feats),
                'label': torch.FloatTensor([y])}

def collate_fn(batch):
    cl, fl, ll = [], [], []
    for i, s in enumerate(batch):
        n = s['coords'].shape[0]
        cl.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), s['coords']], 1))
        fl.append(s['feats'])
        ll.append(s['label'])
    return torch.cat(cl), torch.cat(fl), torch.cat(ll)


class SparseSEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feats, bidx = x.features, x.indices[:, 0].long()
        bs, C = x.batch_size, feats.shape[1]
        pooled = torch.zeros(bs, C, device=feats.device)
        counts = torch.zeros(bs, 1, device=feats.device)
        pooled.scatter_add_(0, bidx.unsqueeze(1).expand_as(feats), feats)
        counts.scatter_add_(0, bidx.unsqueeze(1), torch.ones(len(bidx), 1, device=feats.device))
        se = self.sigmoid(self.fc2(self.relu(self.fc1(pooled / counts.clamp(min=1)))))
        return x.replace_feature(feats * se[bidx])


class SparseResBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, indice_key=None, se_reduction=4):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = spconv.SubMConv2d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv2d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key)
        self.se = SparseSEBlock(out_ch, se_reduction)
        self.skip = (spconv.SubMConv2d(in_ch, out_ch, 1, bias=False, indice_key=indice_key+'_skip')
                     if in_ch != out_ch else None)

    def forward(self, x):
        identity = x
        out = x.replace_feature(self.relu1(self.bn1(x.features)))
        out = self.conv1(out)
        out = out.replace_feature(self.relu2(self.bn2(out.features)))
        out = self.conv2(out)
        out = self.se(out)
        if self.skip: 
            identity = self.skip(identity)
        return out.replace_feature(out.features + identity.features)


class SparseDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, indice_key=None):
        super().__init__()
        self.conv = spconv.SparseConv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False, indice_key=indice_key)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.relu(self.bn(x.features)))


class SparseResNetSEEncoder(nn.Module):
    def __init__(self, in_channels=8, base_ch=64):
        super().__init__()
        self.stem = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, base_ch, 3, padding=1, bias=False, indice_key='stem'),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True))
        
        self.stage1 = nn.ModuleList([
            SparseResBlockSE(base_ch, base_ch, 's1'),
            SparseResBlockSE(base_ch, base_ch, 's1'),
            SparseResBlockSE(base_ch, base_ch, 's1')])
        self.down1 = SparseDownsample(base_ch, base_ch, 'd1')
        
        self.stage2 = nn.ModuleList([
            SparseResBlockSE(base_ch, base_ch*2, 's2'),
            SparseResBlockSE(base_ch*2, base_ch*2, 's2b'),
            SparseResBlockSE(base_ch*2, base_ch*2, 's2b')])
        self.down2 = SparseDownsample(base_ch*2, base_ch*2, 'd2')
        
        self.stage3 = nn.ModuleList([
            SparseResBlockSE(base_ch*2, base_ch*4, 's3'),
            SparseResBlockSE(base_ch*4, base_ch*4, 's3b'),
            SparseResBlockSE(base_ch*4, base_ch*4, 's3b')])
        self.down3 = SparseDownsample(base_ch*4, base_ch*4, 'd3')
        
        self.stage4 = nn.ModuleList([
            SparseResBlockSE(base_ch*4, base_ch*8, 's4'),
            SparseResBlockSE(base_ch*8, base_ch*8, 's4b'),
            SparseResBlockSE(base_ch*8, base_ch*8, 's4b')])

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


class SparseMAEClassifier_SE(nn.Module):
    def __init__(self, in_channels=8, enc_dim=512, dropout=0.5):
        super().__init__()
        self.encoder = SparseResNetSEEncoder(in_channels, enc_dim // 8)
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1))

    def forward(self, coords, feats, batch_size):
        x = spconv.SparseConvTensor(feats, coords, Configs.SPATIAL_SIZE, batch_size)
        x = self.encoder(x)
        features, batch_idx = x.features, x.indices[:, 0].long()
        pooled = torch.zeros(batch_size, features.shape[1], device=features.device)
        counts = torch.zeros(batch_size, 1, device=features.device)
        pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(features), features)
        counts.scatter_add_(0, batch_idx.unsqueeze(1), torch.ones(len(batch_idx), 1, device=features.device))
        return self.classifier(pooled / counts.clamp(min=1))

    def freeze_encoder(self):
        for p in self.encoder.parameters(): 
            p.requires_grad = False
            
    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): 
            p.requires_grad = True


def compute_metrics(labels, probs):
    auc = roc_auc_score(labels, probs)
    fpr, tpr, th = roc_curve(labels, probs)
    inv_fpr = None
    for i in range(len(tpr)):
        if tpr[i] >= 0.7: 
            inv_fpr = 1.0 / max(fpr[i], 1e-10)
            break
            
    j = np.argmax(tpr - fpr)
    bt = th[j]
    preds = (np.array(probs) >= bt).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    cm = confusion_matrix(labels, preds)
    
    return dict(auc=auc, accuracy=acc, precision=prec, recall=rec, f1=f1,
                inv_fpr_at_tpr07=inv_fpr, best_threshold=float(bt),
                confusion_matrix=cm.tolist(), fpr=fpr, tpr=tpr)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    tot, ap, al = 0, [], []
    for coords, feats, labels in tqdm(loader, desc='Train', leave=False):
        coords, feats, labels = coords.to(Configs.DEVICE), feats.to(Configs.DEVICE), labels.to(Configs.DEVICE)
        optimizer.zero_grad()
        logits = model(coords, feats, labels.shape[0])
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
    model.eval()
    tot, ap, al = 0, [], []
    for coords, feats, labels in tqdm(loader, desc='Val  ', leave=False):
        coords, feats, labels = coords.to(Configs.DEVICE), feats.to(Configs.DEVICE), labels.to(Configs.DEVICE)
        logits = model(coords, feats, labels.shape[0])
        tot += criterion(logits.squeeze(), labels.squeeze()).item()
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
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC — Sparse ResNet-SE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_loss_plot(history, path):
    eps = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, [h['tr_loss'] for h in history], 'b-o', ms=3, label='Train')
    ax.plot(eps, [h['va_loss'] for h in history], 'r-o', ms=3, label='Val')
    ax.axvline(Configs.FREEZE_EPOCHS+0.5, color='green', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.set_title('Loss — Sparse ResNet-SE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_auc_plot(history, path):
    eps = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, [h['tr_auc'] for h in history], 'b-o', ms=3, label='Train')
    ax.plot(eps, [h['va_auc'] for h in history], 'r-o', ms=3, label='Val')
    ax.axvline(Configs.FREEZE_EPOCHS+0.5, color='green', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('ROC-AUC — Sparse ResNet-SE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_acc_plot(history, path):
    eps = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, [h['tr_acc'] for h in history], 'b-o', ms=3, label='Train')
    ax.plot(eps, [h['va_acc'] for h in history], 'r-o', ms=3, label='Val')
    ax.axvline(Configs.FREEZE_EPOCHS+0.5, color='green', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy — Sparse ResNet-SE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_cm_plot(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Sparse ResNet-SE')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['Class 0','Class 1'])
    ax.set_yticklabels(['Class 0','Class 1'])
    for ii in range(2):
        for jj in range(2):
            c = 'white' if cm[ii][jj] > cm.max()/2 else 'black'
            ax.text(jj, ii, f'{cm[ii][jj]}', ha='center', va='center', fontsize=16, fontweight='bold', color=c)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == '__main__':
    os.makedirs(Configs.SAVE_DIR, exist_ok=True)
    seed_everything(Configs.SEED)
    
    t0 = time.time()
    dataset = LabelledSparseJetDataset(Configs.LABELED_PATH, Configs.THRESHOLD)
    n = len(dataset)
    nv = int(Configs.VAL_SPLIT * n)
    nt = n - nv
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [nt, nv], generator=torch.Generator().manual_seed(Configs.SEED))

    kw = dict(batch_size=Configs.BATCH_SIZE, num_workers=0, pin_memory=False, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = SparseMAEClassifier_SE(Configs.IN_CHANNELS, Configs.ENCODER_DIM, Configs.DROPOUT).to(Configs.DEVICE)
    if os.path.exists(Configs.ENCODER_PATH):
        ckpt = torch.load(Configs.ENCODER_PATH, map_location=Configs.DEVICE, weights_only=True)
        m, u = model.encoder.load_state_dict(ckpt, strict=False)
        
    criterion = nn.BCEWithLogitsLoss()
    
    history, best_auc, pat, phase = [], 0, 0, 'head-only'

    for epoch in range(1, Configs.EPOCHS + 1):
        if epoch == 1:
            model.freeze_encoder()
            opt = torch.optim.AdamW(model.classifier.parameters(), lr=Configs.LR_HEAD, weight_decay=Configs.WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Configs.FREEZE_EPOCHS, eta_min=1e-5)
            
        if epoch == Configs.FREEZE_EPOCHS + 1:
            model.unfreeze_encoder()
            phase = 'full'
            pat = 0
            opt = torch.optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': Configs.LR_ENCODER},
                {'params': model.classifier.parameters(), 'lr': Configs.LR_HEAD}], weight_decay=Configs.WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Configs.EPOCHS-Configs.FREEZE_EPOCHS, eta_min=1e-6)

        tl, ta, tac = train_one_epoch(model, train_loader, opt, criterion)
        vl, vm, _, _ = evaluate_full(model, val_loader, criterion)
        sched.step()
        
        va, vac = vm['auc'], vm['accuracy']
        history.append(dict(epoch=epoch, phase=phase, tr_loss=tl, tr_auc=ta, tr_acc=tac,
                            va_loss=vl, va_auc=va, va_acc=vac, va_f1=vm['f1'],
                            va_inv_fpr_07=vm['inv_fpr_at_tpr07']))
        mk = ''
        
        if va > best_auc:
            best_auc = va
            pat = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_auc': va,
                        'val_metrics': {k:v for k,v in vm.items() if k not in ('fpr','tpr')}},
                       os.path.join(Configs.SAVE_DIR, 'finetuned_classifier.pt'))
            mk = ' ← best'
        elif epoch > Configs.FREEZE_EPOCHS: 
            pat += 1
            
        print(f"{epoch:<5} {tl:<8.4f} {ta:<8.4f} {tac:<8.4f} {vl:<8.4f} {va:<8.4f} {vac:<8.4f} {phase}{mk}")
        if pat >= Configs.PATIENCE and epoch > Configs.FREEZE_EPOCHS:
            break

    ckpt = torch.load(os.path.join(Configs.SAVE_DIR, 'finetuned_classifier.pt'), map_location=Configs.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    _, fm, _, _ = evaluate_full(model, val_loader, criterion)

    save_roc_plot(fm['fpr'], fm['tpr'], fm['auc'], os.path.join(Configs.SAVE_DIR, 'roc_curve.jpg'))
    save_loss_plot(history, os.path.join(Configs.SAVE_DIR, 'loss_curve.jpg'))
    save_auc_plot(history, os.path.join(Configs.SAVE_DIR, 'auc_curve.jpg'))
    save_acc_plot(history, os.path.join(Configs.SAVE_DIR, 'accuracy_curve.jpg'))
    save_cm_plot(np.array(fm['confusion_matrix']), os.path.join(Configs.SAVE_DIR, 'confusion_matrix.jpg'))

    with open(os.path.join(Configs.SAVE_DIR, 'finetune_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
        
    summary = dict(
        architecture='Sparse ResNet-SE (3 blocks/stage, SE attention)',
        best_epoch=ckpt['epoch'], auc=fm['auc'], accuracy=fm['accuracy'],
        precision=fm['precision'], recall=fm['recall'], f1=fm['f1'],
        inv_fpr_07=fm['inv_fpr_at_tpr07'],
        confusion_matrix=fm['confusion_matrix'],
        comparison={'baseline_sparse_resnet_auc': 0.9602,
                    'delta': round(fm['auc'] - 0.9602, 4)},
        config=dict(seed=Configs.SEED, batch_size=Configs.BATCH_SIZE, lr_head=Configs.LR_HEAD,
                    lr_encoder=Configs.LR_ENCODER, freeze_epochs=Configs.FREEZE_EPOCHS,
                    weight_decay=Configs.WEIGHT_DECAY, dropout=Configs.DROPOUT,
                    mask_ratio_pretrain=0.75, blocks_per_stage=3))
                    
    with open(os.path.join(Configs.SAVE_DIR, 'finetune_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
