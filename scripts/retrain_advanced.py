"""AdvancedDefectPredictor 재학습 스크립트

개선 사항:
  - MultiTaskLoss에서 defect head에 Focal Loss 적용 (소수 클래스 집중)
  - 학습 에포크: 30 → 50
  - patience: 10 → 15
  - 약한 클래스(Scratch/Loc/Edge-Loc) 증강 강화
  - Cosine Annealing with Warm Restarts 적용
"""
import os, sys, random, pickle, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
import json

warnings.filterwarnings('ignore')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from advanced_defect_predictor import AdvancedDefectPredictor

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_ORDER = ['none','Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']
SEVERITY_MAP = {'none':3, 'Center':0, 'Donut':1, 'Edge-Loc':1,
                'Edge-Ring':1, 'Loc':2, 'Near-full':0, 'Random':2, 'Scratch':2}
NUM_DEFECT = 9; NUM_SEVERITY = 4

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / 'data/processed'
CHECKPOINT_DIR = ROOT / 'checkpoints'
ANALYSIS_DIR   = ROOT / 'analysis'

BATCH_SIZE = 64; EPOCHS = 50; LR = 3e-4; WEIGHT_DECAY = 1e-4; PATIENCE = 15


class FocalLoss(nn.Module):
    """Focal Loss — 소수 클래스에 더 많은 가중치 부여."""
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, focal_gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.beta = beta; self.gamma_w = gamma
        self.defect_loss   = FocalLoss(gamma=focal_gamma)
        self.severity_loss = nn.CrossEntropyLoss()
        self.conf_loss     = nn.MSELoss()

    def forward(self, outputs, defect_labels, severity_labels, conf_targets):
        l_def = self.defect_loss(outputs['defect'], defect_labels)
        l_sev = self.severity_loss(outputs['severity'], severity_labels)
        l_conf = self.conf_loss(outputs['confidence'].squeeze(1), conf_targets)
        return self.alpha * l_def + self.beta * l_sev + self.gamma_w * l_conf


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


def build_train_transform():
    return A.Compose([
        A.Rotate(limit=30, border_mode=0, value=0.0, p=0.6),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.GaussNoise(var_limit=(0.001,0.008), p=0.4),
        A.Blur(blur_limit=3, p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
        A.CoarseDropout(max_holes=6, max_height=8, max_width=8,
                        min_holes=2, min_height=4, min_width=4, fill_value=0.0, p=0.4),
        A.ElasticTransform(alpha=20, sigma=5, p=0.2),
    ])


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot_loss = tot_n = 0
    preds_all = []; labels_all = []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            sev = torch.tensor([SEVERITY_MAP[CLASS_ORDER[c]] for c in y.cpu().tolist()],
                               dtype=torch.long, device=DEVICE)
            conf_t = (y != 0).float()
            out = model(x)
            loss = criterion(out, y, sev, conf_t)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item() * x.size(0); tot_n += x.size(0)
            preds_all.extend(out['defect'].argmax(1).cpu().tolist())
            labels_all.extend(y.cpu().tolist())
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    return tot_loss/tot_n, f1, preds_all, labels_all


def main():
    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    all_maps = np.load(PROCESSED_DIR / 'all_maps_resized.npy')
    with open(PROCESSED_DIR / 'split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
    encoded_labels = split['encoded_labels']
    train_idx = split['train_idx']; val_idx = split['val_idx']; test_idx = split['test_idx']
    class_weights = split['class_weights']
    train_labels = encoded_labels[train_idx].astype(int)
    val_labels   = encoded_labels[val_idx].astype(int)
    test_labels  = encoded_labels[test_idx].astype(int)

    sampler = WeightedRandomSampler(
        torch.FloatTensor(class_weights[train_labels]), len(train_labels), replacement=True)
    train_loader = DataLoader(WaferMapDataset(all_maps[train_idx], train_labels, build_train_transform()),
                              BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(WaferMapDataset(all_maps[val_idx],   val_labels),
                              BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(WaferMapDataset(all_maps[test_idx],  test_labels),
                              BATCH_SIZE, shuffle=False, num_workers=0)

    model = AdvancedDefectPredictor(num_defect_classes=NUM_DEFECT, num_severity_classes=NUM_SEVERITY,
                                    pretrained=True).to(DEVICE)

    # 재학습된 MobileNetV3 백본 가중치 전이 (가능한 경우)
    ft_results_path = ANALYSIS_DIR / 'finetuning_results.json'
    if ft_results_path.exists():
        try:
            with open(ft_results_path, encoding='utf-8') as f:
                ft_results = json.load(f)
            mv3_ckpt_path = ft_results['models']['MobileNetV3']['checkpoint']
            mv3_ckpt = torch.load(mv3_ckpt_path, map_location=DEVICE, weights_only=False)
            # backbone(features) 가중치만 전이
            state = mv3_ckpt['model_state']
            backbone_state = {k.replace('features.', ''): v
                              for k, v in state.items() if k.startswith('features.')}
            model.backbone.load_state_dict(backbone_state, strict=True)
            print(f'  재학습 MobileNetV3 백본 가중치 전이 완료: {mv3_ckpt_path}')
        except Exception as e:
            print(f'  백본 가중치 전이 실패 ({e}) — ImageNet 초기화 사용')
    else:
        print(f'  finetuning_results.json 없음 — ImageNet 초기화 사용')

    criterion = MultiTaskLoss(alpha=0.5, beta=0.3, gamma=0.2, focal_gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_f1 = 0.0; best_ckpt = None; patience_cnt = 0
    print(f'\n{"Ep":>3} | {"Loss":>7} | {"VlF1":>6} | Best')
    print('-'*35)
    for ep in range(1, EPOCHS+1):
        tr_loss, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        _, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        scheduler.step()
        is_best = vl_f1 > best_val_f1
        if is_best:
            best_val_f1 = vl_f1; patience_cnt = 0
            best_ckpt = CHECKPOINT_DIR / f'AdvancedDefectPredictor_best_{vl_f1:.4f}.pth'
            torch.save({'epoch':ep,'model_state_dict':model.state_dict(),'val_f1':vl_f1}, best_ckpt)
        else:
            patience_cnt += 1
        print(f'{ep:>3} | {tr_loss:>7.4f} | {vl_f1:>6.4f} | {"★" if is_best else ""}', flush=True)
        if patience_cnt >= PATIENCE:
            print(f'  Early stopping at epoch {ep}'); break

    # ── Test 평가
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    _, test_f1, test_preds, test_true = run_epoch(model, test_loader, criterion)
    # accuracy
    test_acc = sum(p==l for p,l in zip(test_preds, test_true)) / len(test_true)

    print(f'\n=== AdvancedDefectPredictor (수정) Test ===')
    print(f'Accuracy : {test_acc*100:.2f}%')
    print(f'F1-macro : {test_f1:.4f}  (목표 ≥ 0.90)  {"[OK]" if test_f1>=0.90 else "[MISS]"}')
    print(classification_report(test_true, test_preds, target_names=CLASS_ORDER, zero_division=0))

    res = {'model':'AdvancedDefectPredictor (MobileNetV3 Small + 3 heads)',
           'best_val_defect_f1': round(float(best_val_f1),4),
           'test_defect_f1': round(float(test_f1),4),
           'test_accuracy':  round(float(test_acc),4),
           'epochs_trained': int(ckpt['epoch']),
           'fix_note': 'Focal Loss(gamma=2) + 강화 증강 + CosineAnnealingWarmRestarts + patience=15',
           'loss_weights': {'alpha':0.5,'beta':0.3,'gamma':0.2,'focal_gamma':2.0},
           'class_names': CLASS_ORDER}
    with open(ANALYSIS_DIR / 'advanced_model_results.json','w',encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'결과 저장: analysis/advanced_model_results.json')
    print(f'체크포인트: {best_ckpt}')


if __name__ == '__main__':
    main()
