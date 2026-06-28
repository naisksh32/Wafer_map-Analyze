"""MobileNetV3 / EfficientNet-B0 / ViT-Tiny 재학습 스크립트

수정 사항:
  - CrossEntropyLoss에서 class_weights 제거 (이중 불균형 보정 방지)
  - WeightedRandomSampler는 유지
  - Phase2 epochs: 25 → 35 (충분한 수렴)
  - patience: 7 → 10
"""
import os, sys, random, pickle, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision.models import (mobilenet_v3_small, MobileNet_V3_Small_Weights,
                                 efficientnet_b0, EfficientNet_B0_Weights)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
import json

warnings.filterwarnings('ignore')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

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

BATCH_SIZE = 64; WEIGHT_DECAY = 1e-4; PATIENCE = 10
PHASE1_EPOCHS = 5;  PHASE1_LR = 1e-3
PHASE2_EPOCHS = 35; PHASE2_LR = 1e-4


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


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot_loss = tot_correct = tot_n = 0
    preds_all = []; labels_all = []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); loss = criterion(out, y)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item() * x.size(0)
            tot_correct += (out.argmax(1)==y).sum().item(); tot_n += x.size(0)
            preds_all.extend(out.argmax(1).cpu().tolist()); labels_all.extend(y.cpu().tolist())
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    return tot_loss/tot_n, tot_correct/tot_n, f1, preds_all, labels_all


def build_mobilenet_v3_small(pretrained=True):
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = mobilenet_v3_small(weights=weights)
    old = m.features[0][0]
    new = nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False)
    if pretrained:
        new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
    m.features[0][0] = new
    m.classifier[-1] = nn.Linear(1024, NUM_CLASSES)
    return m


def build_efficientnet_b0(pretrained=True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = efficientnet_b0(weights=weights)
    old = m.features[0][0]
    new = nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
    if pretrained:
        new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
    m.features[0][0] = new
    m.classifier[-1] = nn.Linear(1280, NUM_CLASSES)
    return m


def build_vit_tiny(pretrained=True):
    m = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained,
                           num_classes=NUM_CLASSES, in_chans=1, img_size=64)
    return m


def freeze_backbone(model, name):
    if name == 'MobileNetV3':
        for p in model.features.parameters(): p.requires_grad = False
    elif name == 'EfficientNet-B0':
        for n, p in model.named_parameters():
            if 'classifier' not in n: p.requires_grad = False
    elif name == 'ViT-Tiny':
        for n, p in model.named_parameters():
            if 'head' not in n: p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters(): p.requires_grad = True


def finetune(name, model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()  # 단일 보정: Sampler만 사용
    best_val_f1 = 0.0; best_ckpt = None; patience_cnt = 0

    # Phase 1
    freeze_backbone(model, name)
    opt1 = optim.Adam([p for p in model.parameters() if p.requires_grad],
                      lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)
    print(f'\n[{name}] Phase 1 (백본 동결, {PHASE1_EPOCHS} epochs)')
    for ep in range(1, PHASE1_EPOCHS+1):
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, opt1)
        _, _, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        is_best = vl_f1 > best_val_f1
        if is_best:
            best_val_f1 = vl_f1; patience_cnt = 0
            best_ckpt = CHECKPOINT_DIR / f'{name}_p1_{ep:02d}_{vl_f1:.4f}.pth'
            torch.save({'epoch':ep,'model_state':model.state_dict(),'val_f1':vl_f1,'model_name':name}, best_ckpt)
        else:
            patience_cnt += 1
        print(f'  P1 Ep{ep:02d} | TrLoss={tr_loss:.4f} TrAcc={tr_acc:.4f} VlF1={vl_f1:.4f} {"★" if is_best else ""}', flush=True)

    # Phase 2
    unfreeze_all(model); patience_cnt = 0
    opt2 = optim.Adam([
        {'params': [p for n,p in model.named_parameters() if 'classifier' not in n and 'head' not in n],
         'lr': PHASE2_LR * 0.1},
        {'params': [p for n,p in model.named_parameters() if 'classifier' in n or 'head' in n],
         'lr': PHASE2_LR},
    ], weight_decay=WEIGHT_DECAY)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EPOCHS, eta_min=1e-6)
    print(f'[{name}] Phase 2 (전체 Unfreeze, {PHASE2_EPOCHS} epochs, backbone_lr={PHASE2_LR*0.1:.1e})')
    for ep in range(1, PHASE2_EPOCHS+1):
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, opt2)
        _, _, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        sch2.step()
        is_best = vl_f1 > best_val_f1
        if is_best:
            best_val_f1 = vl_f1; patience_cnt = 0
            best_ckpt = CHECKPOINT_DIR / f'{name}_{ep:02d}_{vl_f1:.4f}.pth'
            torch.save({'epoch':ep,'model_state':model.state_dict(),'val_f1':vl_f1,
                        'val_acc':tr_acc,'model_name':name}, best_ckpt)
        else:
            patience_cnt += 1
        print(f'  P2 Ep{ep:02d} | TrLoss={tr_loss:.4f} VlF1={vl_f1:.4f} {"★" if is_best else ""}', flush=True)
        if patience_cnt >= PATIENCE:
            print(f'  Early stopping at P2 epoch {ep}'); break

    return best_val_f1, best_ckpt, criterion


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

    BUILD_FNS = {
        'MobileNetV3':    build_mobilenet_v3_small,
        'EfficientNet-B0': build_efficientnet_b0,
        'ViT-Tiny':        build_vit_tiny,
    }

    all_results = {}
    for model_name, build_fn in BUILD_FNS.items():
        print(f'\n{"="*60}')
        print(f'모델: {model_name}')
        model = build_fn(pretrained=True).to(DEVICE)
        best_val_f1, best_ckpt, criterion = finetune(model_name, model, train_loader, val_loader)

        # Test 평가
        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        model_eval = build_fn(pretrained=False).to(DEVICE)
        model_eval.load_state_dict(ckpt['model_state'])
        _, test_acc, test_f1, test_preds, test_true = run_epoch(model_eval, test_loader, criterion)
        print(f'\n[{model_name}] Test Acc={test_acc*100:.2f}%  F1={test_f1:.4f}  '
              f'(목표≥0.88) {"[OK]" if test_f1>=0.88 else "[MISS]"}')
        print(classification_report(test_true, test_preds, target_names=CLASS_ORDER, zero_division=0))
        all_results[model_name] = {
            'best_val_f1': round(float(best_val_f1),4),
            'test_accuracy': round(float(test_acc),4),
            'test_f1_macro': round(float(test_f1),4),
            'target_met_acc': bool(test_acc >= 0.90),
            'target_met_f1':  bool(test_f1 >= 0.88),
            'checkpoint': str(best_ckpt),
        }

    best_model = max(all_results, key=lambda k: all_results[k]['test_f1_macro'])
    result_json = {'best_model': best_model, 'models': all_results,
                   'fix_note': '단일 불균형 보정: WeightedRandomSampler만, CrossEntropyLoss weight 제거',
                   'hyperparams': {'phase1_epochs':PHASE1_EPOCHS,'phase1_lr':PHASE1_LR,
                                   'phase2_epochs':PHASE2_EPOCHS,'phase2_lr':PHASE2_LR,
                                   'batch_size':BATCH_SIZE,'weight_decay':WEIGHT_DECAY,'patience':PATIENCE}}
    with open(ANALYSIS_DIR / 'finetuning_results.json','w',encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f'\n결과 저장: analysis/finetuning_results.json')
    print(f'최고 성능 모델: {best_model}  F1={all_results[best_model]["test_f1_macro"]:.4f}')


if __name__ == '__main__':
    main()
