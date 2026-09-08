"""전 모델 통합 재평가 — 동일 체크포인트 · 동일 테스트셋 · 동일 지표 계산

README 성능표의 근거 파일을 생성한다.
대응 이슈: "F1과 Accuracy가 서로 다른 시점/다른 로더에서 측정되어 수치가 성립 불가"
           (예: macro F1 0.501 인데 Accuracy 10.65% 는 동시에 나올 수 없음)

핵심 원칙:
    1. 하나의 체크포인트에서 두 지표를 동시에 산출
    2. 테스트셋은 증강·WeightedRandomSampler 미적용 → 원본 불균형 분포 그대로
    3. 클래스별 Precision/Recall/F1/Support 와 Confusion Matrix 를 함께 저장

산출물:
    analysis/final_evaluation.json          — 모델별 Accuracy / macro F1 / weighted F1
    analysis/per_class_metrics.csv          — 모델 × 클래스별 지표
    analysis/final_confusion_matrices.png   — 모델별 혼동행렬
    analysis/baseline_classification_report.txt
"""
import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import WaferMapDataset, CLASS_ORDER          # noqa: E402
from src.model import WaferCNN                                     # noqa: E402
from src.advanced_defect_predictor import AdvancedDefectPredictor  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
NUM_CLASSES = len(CLASS_ORDER)


# ── 모델 빌더 (05_finetuning.ipynb 과 동일 구조) ─────────────────────
def build_mobilenet_v3_small(num_classes=NUM_CLASSES):
    model = torchvision.models.mobilenet_v3_small(weights=None)
    old = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, old.out_channels, old.kernel_size,
                                     old.stride, old.padding, bias=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes=NUM_CLASSES):
    # 주의: 재학습(scripts/retrain_finetune.py)은 timm 이 아닌 torchvision 구조를 사용했다.
    #       체크포인트 키가 features.* / classifier.1 이므로 동일하게 맞춘다.
    model = torchvision.models.efficientnet_b0(weights=None)
    old = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, old.out_channels, old.kernel_size,
                                     old.stride, old.padding, bias=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def build_vit_tiny(num_classes=NUM_CLASSES):
    import timm
    return timm.create_model('vit_tiny_patch16_224', pretrained=False,
                             num_classes=num_classes, in_chans=1, img_size=64)


def build_wafer_cnn(num_classes=NUM_CLASSES, dropout=0.3):
    return WaferCNN(num_classes=num_classes, dropout=dropout)


# 평가 대상: (표시명, 빌더, 체크포인트, 비고)
MODELS = [
    ('WaferCNN', build_wafer_cnn,
     'checkpoints/WaferCNN_23_0.8485.pth', '커스텀 4-Conv CNN 베이스라인'),
    ('WaferCNN + Optuna HPO', build_wafer_cnn,
     'checkpoints/WaferCNN_best_hpo.pth', 'Optuna 20 trials 최적 파라미터 재학습'),
    ('MobileNetV3-Small', build_mobilenet_v3_small,
     'checkpoints/MobileNetV3_34_0.7417.pth', '2-Phase 파인튜닝 · 엣지 배포 채택'),
    ('EfficientNet-B0', build_efficientnet_b0,
     'checkpoints/EfficientNet-B0_19_0.7999.pth', '2-Phase 파인튜닝'),
    ('ViT-Tiny', build_vit_tiny,
     'checkpoints/ViT-Tiny_29_0.8351.pth', '2-Phase 파인튜닝'),
    ('AdvancedDefectPredictor', None,
     'checkpoints/AdvancedDefectPredictor_best_0.8279.pth', 'Multi-output(분류+심각도+신뢰도)'),
]


def load_state(ckpt):
    """체크포인트 dict 에서 state_dict 추출 (키 이름이 파일마다 다름)."""
    if not isinstance(ckpt, dict):
        return ckpt, {}
    for key in ('model_state', 'model_state_dict', 'state_dict'):
        if key in ckpt:
            return ckpt[key], ckpt
    return ckpt, {}


def get_test_loader(batch_size=256):
    maps = np.load(ROOT / 'data/processed/all_maps_resized.npy', mmap_mode='r')
    with open(ROOT / 'data/processed/split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
    test_idx = split['test_idx']
    labels = split['encoded_labels'][test_idx].astype(int)
    # 증강 없음 · 샘플러 없음 → 원본 불균형 분포 그대로의 테스트셋
    ds = WaferMapDataset(np.asarray(maps[test_idx]), labels, transform=None)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), labels


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        out = model(x.to(DEVICE))
        if isinstance(out, dict):          # Multi-output 모델은 분류 헤드만 사용
            out = out['defect']
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def main():
    print(f'Device: {DEVICE}')
    loader, test_labels = get_test_loader()
    dist = pd.Series(test_labels).value_counts().sort_index()
    print(f'Test set: {len(test_labels):,}개 (증강·샘플러 미적용, 원본 불균형 유지)')
    print('클래스 분포:', {CLASS_ORDER[i]: int(n) for i, n in dist.items()})
    print()

    results, per_class_rows, cms = {}, [], {}

    for name, builder, ckpt_rel, note in MODELS:
        path = ROOT / ckpt_rel
        if not path.exists():
            print(f'[SKIP] {name}: 체크포인트 없음 ({ckpt_rel})')
            continue
        try:
            raw = torch.load(path, map_location=DEVICE, weights_only=False)
            state, meta = load_state(raw)

            if name == 'AdvancedDefectPredictor':
                model = AdvancedDefectPredictor(pretrained=False)
            elif builder is build_wafer_cnn:
                cfg = meta.get('config') or {}
                model = WaferCNN(num_classes=cfg.get('num_classes', NUM_CLASSES),
                                 dropout=cfg.get('dropout', 0.3))
            else:
                model = builder()

            model.load_state_dict(state)
            model.to(DEVICE)
        except Exception as e:
            print(f'[FAIL] {name}: {type(e).__name__}: {e}')
            continue

        preds, trues = predict(model, loader)
        acc = accuracy_score(trues, preds)
        f1_mac = f1_score(trues, preds, average='macro', zero_division=0)
        f1_wtd = f1_score(trues, preds, average='weighted', zero_division=0)
        n_params = sum(p.numel() for p in model.parameters())

        results[name] = {
            'checkpoint': ckpt_rel,
            'note': note,
            'params_M': round(n_params / 1e6, 2),
            'test_accuracy': round(float(acc), 4),
            'test_f1_macro': round(float(f1_mac), 4),
            'test_f1_weighted': round(float(f1_wtd), 4),
        }
        print(f'{name:<26} Acc {acc * 100:6.2f}%  macroF1 {f1_mac:.4f}  '
              f'wtdF1 {f1_wtd:.4f}  ({n_params / 1e6:.2f}M params)')

        p, r, f, s = precision_recall_fscore_support(
            trues, preds, labels=range(NUM_CLASSES), zero_division=0)
        for i, cls in enumerate(CLASS_ORDER):
            per_class_rows.append({'model': name, 'class': cls, 'support': int(s[i]),
                                   'precision': round(float(p[i]), 4),
                                   'recall': round(float(r[i]), 4),
                                   'f1': round(float(f[i]), 4)})
        cms[name] = confusion_matrix(trues, preds, labels=range(NUM_CLASSES))

        if name == 'WaferCNN':
            (ROOT / 'analysis').mkdir(exist_ok=True)
            (ROOT / 'analysis/baseline_classification_report.txt').write_text(
                classification_report(trues, preds, target_names=CLASS_ORDER,
                                      digits=4, zero_division=0), encoding='utf-8')

        del model
        torch.cuda.empty_cache()

    out = {
        'description': '전 모델 통합 재평가 — 동일 테스트셋(증강/샘플러 미적용) · 동일 지표 계산',
        'test_set_size': int(len(test_labels)),
        'test_class_distribution': {CLASS_ORDER[i]: int(n) for i, n in dist.items()},
        'device': str(DEVICE),
        'seed': SEED,
        'models': results,
    }
    (ROOT / 'analysis/final_evaluation.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(per_class_rows).to_csv(ROOT / 'analysis/per_class_metrics.csv',
                                        index=False, encoding='utf-8-sig')

    if cms:
        ncol = 3
        nrow = (len(cms) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, (name, cm) in zip(axes, cms.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax,
                        annot_kws={'size': 7})
            ax.set_title(f'{name}\nAcc {results[name]["test_accuracy"] * 100:.2f}% / '
                         f'macro F1 {results[name]["test_f1_macro"]:.4f}', fontsize=10)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.tick_params(labelsize=7)
        for ax in axes[len(cms):]:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(ROOT / 'analysis/final_confusion_matrices.png', dpi=130,
                    bbox_inches='tight')

    print('\n저장 완료:')
    print('  analysis/final_evaluation.json')
    print('  analysis/per_class_metrics.csv')
    print('  analysis/final_confusion_matrices.png')
    print('  analysis/baseline_classification_report.txt')


if __name__ == '__main__':
    main()
