import os, json, time, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import spconv.pytorch as spconv

# Config
class Config:
  def __init__(self):
    self.LABELED_PATH = '/raid/home/dgx1736/Arush1/Dataset_Specific_labelled.h5'
    self.ENCODER_PATH = '/raid/home/dgx1736/Arush1/sparse_resnet/sparse_mae_encoder.pt'
    self.SAVE_DIR = '/raid/home/dgx1736/Arush1/sparse_resnet'

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.seed = 42
    self.threshold = 0.0

    self.batch_size = 64
    self.epochs = 40
    self.lr_head = 1e-3
    self.lr_encoder = 5e-5
    self.freeze_epochs = 5
    self.weight_decay = 1e-3
    self.dropout = 0.5

    self.spatial_size = [125, 125]
    self.in_channels = 8
    self.encoder_dim = 512

    self.val_split = 0.2
    self.patience = 6

    self.labeled_jet_offset = 2048
    self.labeled_y_offset = 5000002048
    self.labeled_n_samples = 10000

config = Config()
os.makedirs(config.SAVE_DIR, exist_ok=True)

def seed_everything(s):
  random.seed(s); np.random.seed(s); torch.manual_seed(s)
  torch.cuda.manual_seed_all(s)
  torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(config.seed)
print(f"Device: {config.device} | Seed: {config.seed}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

# Dataset+ Lazy Loading
class LabelledSparseJetDataset(Dataset):
  def __init__(self, path, threshold=0.0):
    self.path, self.threshold = path, threshold
    self.n_samples = config.labeled_n_samples
    self.sample_bytes = 125 * 125 * 8 * 4
    with open(path, 'rb') as f:
      f.seek(config.labeled_y_offset)
      self.Y = np.frombuffer(f.read(self.n_samples * 4), dtype=np.float32).copy()

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
    with open(self.path, 'rb') as f:
      f.seek(config.labeled_jet_offset + idx * self.sample_bytes)
      raw = f.read(self.sample_bytes)

    img = np.frombuffer(raw, dtype=np.float32).reshape(125, 125, 8).copy() / 255.0
    y = self.Y[idx]

    coords = np.argwhere(img.sum(axis=-1) > config.threshold)
    if len(coords) == 0:
      coords = np.array([[0, 0]], dtype=np.int32)
      feats = np.zeros((1, config.in_channels), dtype=np.float32)
    else:
      feats = img[coords[:, 0], coords[:, 1], :]
      coords = coords.astype(np.int32)

    return {
      'coords': torch.IntTensor(coords),
      'feats': torch.FloatTensor(feats),
      'label': torch.FloatTensor([y])
    }

def collate_fn(batch):
  cl, fl, ll = [], [], []
  for i, s in enumerate(batch):
    n = s['coords'].shape[0]
    cl.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), s['coords']], 1))
    fl.append(s['feats'])
    ll.append(s['label'])
  return torch.cat(cl), torch.cat(fl), torch.cat(ll)

# Model used
class SparseResBlock(nn.Module):
  def __init__(self, in_ch, out_ch, indice_key=None):
    super().__init__()
    self.bn1 = nn.BatchNorm1d(in_ch)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = spconv.SubMConv2d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key)
    self.bn2 = nn.BatchNorm1d(out_ch)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = spconv.SubMConv2d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key)
    self.skip = spconv.SubMConv2d(in_ch, out_ch, 1, bias=False, indice_key=indice_key+'_skip') if in_ch != out_ch else None

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
    self.bn = nn.BatchNorm1d(out_ch)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv(x)
    return x.replace_feature(self.relu(self.bn(x.features)))

class SparseResNetEncoder(nn.Module):
  def __init__(self, in_channels=8, base_ch=64):
    super().__init__()
    self.stem = spconv.SparseSequential(
      spconv.SubMConv2d(in_channels, base_ch, 3, padding=1, bias=False, indice_key='stem'),
      nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True)
    )
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

class SparseMAEClassifier(nn.Module):
  def __init__(self, in_channels=8, enc_dim=512, dropout=0.5):
    super().__init__()
    self.encoder = SparseResNetEncoder(in_channels, enc_dim // 8)
    self.classifier = nn.Sequential(
      nn.Linear(enc_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
      nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
      nn.Linear(64, 1)
    )
# First 5 epochs backbone is frozen
  def forward(self, coords, feats, batch_size):
    x = spconv.SparseConvTensor(feats, coords, config.spatial_size, batch_size)
    x = self.encoder(x)
    features, batch_idx = x.features, x.indices[:, 0].long()

    pooled = torch.zeros(batch_size, features.shape[1], device=features.device)
    counts = torch.zeros(batch_size, 1, device=features.device)
    pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(features), features)
    counts.scatter_add_(0, batch_idx.unsqueeze(1), torch.ones(len(batch_idx), 1, device=features.device))

    return self.classifier(pooled / counts.clamp(min=1))

  def freeze_encoder(self):
    for p in self.encoder.parameters(): p.requires_grad = False

  def unfreeze_encoder(self):
    for p in self.encoder.parameters(): p.requires_grad = True


if __name__ == '__main__':
  dataset = LabelledSparseJetDataset(config.LABELED_PATH, config.threshold)
  n = len(dataset)
  nv = int(config.val_split * n)
  nt = n - nv

  train_ds, val_ds = torch.utils.data.random_split(
    dataset, [nt, nv], generator=torch.Generator().manual_seed(config.seed)
  )

  kw = dict(batch_size=config.batch_size, num_workers=0, pin_memory=False, collate_fn=collate_fn)
  train_loader = DataLoader(train_ds, shuffle=True, **kw)
  val_loader = DataLoader(val_ds, shuffle=False, **kw)

  model = SparseMAEClassifier(config.in_channels, config.encoder_dim, config.dropout).to(config.device)
  print("Setup complete.")
