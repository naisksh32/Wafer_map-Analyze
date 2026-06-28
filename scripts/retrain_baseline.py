"""WaferCNN 재학습 스크립트 — 이중 불균형 보정 버그 수정 버전

수정 사항:
  - CrossEntropyLoss에서 class_weights 제거 (WeightedRandomSampler와 이중 보정 방지)
  - lr: 1e-3 → 3e-4 (안정적 수렴)
  - patience: 7 → 10
  - epochs: 30 → 40
"""
import os, sys, random, pickle, warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
import json

warnings.filterwarnings('ignore')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from model import WaferCNN

# ── 설정
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_ORDER = ['none','Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']
NUM_CLASSES = 9

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / 'data/processed'
CHECKPOINT_DIR = ROOT / 'checkpoints'
ANALYSIS_DIR   = ROOT / 'analysis'

BATCH_SIZE = 64; EPOCHS = 40; LR = 3e-4; WEIGHT_DECAY = 1e-4; DROPOUT = 0.3; PATIENCE = 10


class WaferMapDataset(Dataset):
    def __init__(self, maps, labels, transform=None):
        self.maps = maps; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        wmap = self.maps[idx].astype(np.float32) / 2.0
        if self.transform:
            wmap = self.transform(image=wmap[:,:,np.newaxis])['image'][:,:,0]
        return torch.from_numpy(np.clip(wmap,0,1).copy()).unsqueeze(0), \
               torch.tensor(self.labels[idx], dtype=torch.long)


def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot_loss = tot_correct = tot_n = 0
    preds_all = []; labels_all = []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x); loss = criterion(out, y)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item() * x.size(0)
            tot_correct += (out.argmax(1)==y).sum().item(); tot_n += x.size(0)
            preds_all.extend(out.argmax(1).cpu().tolist()); labels_all.extend(y.cpu().tolist())
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    return tot_loss/tot_n, tot_correct/tot_n, f1, preds_all, labels_all


def main():
    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # ── 데이터
    all_maps = np.load(PROCESSED_DIR / 'all_maps_resized.npy')
    with open(PROCESSED_DIR / 'split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
    encoded_labels = split['encoded_labels']
    train_idx = split['train_idx']; val_idx = split['val_idx']; test_idx = split['test_idx']
    class_weights = split['class_weights']
    train_labels = encoded_labels[train_idx].astype(int)
    val_labels   = encoded_labels[val_idx].astype(int)
    test_labels  = encoded_labels[test_idx].astype(int)

    train_transform = A.Compose([
        A.Rotate(limit=20, border_mode=0, value=0.0, p=0.5),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.GaussNoise(var_limit=(0.001,0.005), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
        A.CoarseDropout(max_holes=4, max_height=8, max_width=8,
                        min_holes=1, min_height=4, min_width=4, fill_value=0.0, p=0.3),
    ])

    sampler = WeightedRandomSampler(
        torch.FloatTensor(class_weights[train_labels]), len(train_labels), replacement=True)
    train_loader = DataLoader(WaferMapDataset(all_maps[train_idx], train_labels, train_transform),
                              BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(WaferMapDataset(all_maps[val_idx],   val_labels),
                              BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(WaferMapDataset(all_maps[test_idx],  test_labels),
                              BATCH_SIZE, shuffle=False, num_workers=0)

    model = WaferCNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
    print(f'파라미터: {model.count_params()/1e6:.2f}M')

    # 단일 보정: WeightedRandomSampler만, CrossEntropyLoss는 weight 없음
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_f1 = 0.0; best_ckpt = None; patience_cnt = 0
    print(f'\n{"Ep":>3} | {"LR":>8} | {"TrLoss":>7} | {"TrAcc":>6} | {"VlF1":>6} | Best')
    print('-'*55)
    for ep in range(1, EPOCHS+1):
        lr_now = optimizer.param_groups[0]['lr']
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        _, _, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        scheduler.step()
        is_best = vl_f1 > best_val_f1
        if is_best:
            best_val_f1 = vl_f1; patience_cnt = 0
            best_ckpt = CHECKPOINT_DIR / f'WaferCNN_{ep:02d}_{vl_f1:.4f}.pth'
            torch.save({'epoch':ep,'model_state':model.state_dict(),'val_f1':vl_f1,
                        'config':{'num_classes':NUM_CLASSES,'dropout':DROPOUT}}, best_ckpt)
        else:
            patience_cnt += 1
        print(f'{ep:>3} | {lr_now:.2e} | {tr_loss:>7.4f} | {tr_acc:>6.4f} | {vl_f1:>6.4f} | {"★" if is_best else ""}',
              flush=True)
        if patience_cnt >= PATIENCE:
            print(f'  Early stopping at epoch {ep}'); break

    # ── Test
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    _, test_acc, test_f1, test_preds, test_true = run_epoch(model, test_loader, criterion)
    print(f'\n=== WaferCNN (수정) Test ===')
    print(f'Accuracy : {test_acc*100:.2f}%')
    print(f'F1-macro : {test_f1:.4f}  (목표 ≥ 0.80)  {"[OK]" if test_f1>=0.80 else "[MISS]"}')
    print(classification_report(test_true, test_preds, target_names=CLASS_ORDER, zero_division=0))

    res = {'model':'WaferCNN','best_epoch':int(ckpt['epoch']),
           'best_val_f1':round(float(best_val_f1),4),
           'test_accuracy':round(float(test_acc),4),'test_f1_macro':round(float(test_f1),4),
           'target_met':bool(test_f1>=0.80),
           'fix_note':'단일 불균형 보정: WeightedRandomSampler만 사용, lr=3e-4, patience=10',
           'hyperparams':{'epochs':EPOCHS,'batch_size':BATCH_SIZE,'lr':LR,
                          'weight_decay':WEIGHT_DECAY,'dropout':DROPOUT},
           'best_checkpoint':str(best_ckpt)}
    with open(ANALYSIS_DIR / 'baseline_results.json','w',encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'결과 저장: analysis/baseline_results.json')


if __name__ == '__main__':
    main()
