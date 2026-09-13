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
# 체크포인트 경로는 재학습 스크립트가 남긴 analysis/*_results.json 에서 동적으로 읽는다.
# (과거 버전은 특정 파일명이 하드코딩되어 있어 재학습 시마다 수정이 필요했음)
def _read_json(rel):
    path = ROOT / rel
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _latest(pattern):
    cands = sorted((ROOT / 'checkpoints').glob(pattern), key=lambda p: p.stat().st_mtime)
    return cands[-1].relative_to(ROOT).as_posix() if cands else None


def resolve_checkpoints():
    base = _read_json('analysis/baseline_results.json')
    ft   = _read_json('analysis/finetuning_results.json').get('models', {})
    adv  = _read_json('analysis/advanced_model_results.json')

    def _rel(p):
        if not p:
            return None
        p = Path(p)
        if p.is_absolute():
            # 다른 머신의 절대경로가 남아 있을 수 있음 → 파일명만 취해 checkpoints/ 에서 찾는다
            cand = ROOT / 'checkpoints' / p.name
            return cand.relative_to(ROOT).as_posix() if cand.exists() else None
        return p.as_posix() if (ROOT / p).exists() else None

    return {
        'WaferCNN':               _rel(base.get('best_checkpoint')) or _latest('WaferCNN_[0-9]*_*.pth'),
        'WaferCNN + Optuna HPO':  _rel('checkpoints/WaferCNN_best_hpo.pth'),
        'MobileNetV3-Small':      _rel(ft.get('MobileNetV3', {}).get('checkpoint')) or _latest('MobileNetV3_[0-9]*_*.pth'),
        'EfficientNet-B0':        _rel(ft.get('EfficientNet-B0', {}).get('checkpoint')) or _latest('EfficientNet-B0_[0-9]*_*.pth'),
        'ViT-Tiny':               _rel(ft.get('ViT-Tiny', {}).get('checkpoint')) or _latest('ViT-Tiny_[0-9]*_*.pth'),
        'AdvancedDefectPredictor': _rel(adv.get('checkpoint')) or _latest('AdvancedDefectPredictor_best_*.pth'),
    }


_CK = resolve_checkpoints()
MODELS = [
    ('WaferCNN', build_wafer_cnn, _CK['WaferCNN'], '커스텀 4-Conv CNN 베이스라인'),
    ('WaferCNN + Optuna HPO', build_wafer_cnn, _CK['WaferCNN + Optuna HPO'], 'Optuna 20 trials 최적 파라미터 재학습'),
    ('MobileNetV3-Small', build_mobilenet_v3_small, _CK['MobileNetV3-Small'], '2-Phase 파인튜닝 · 엣지 배포 채택'),
    ('EfficientNet-B0', build_efficientnet_b0, _CK['EfficientNet-B0'], '2-Phase 파인튜닝'),
    ('ViT-Tiny', build_vit_tiny, _CK['ViT-Tiny'], '2-Phase 파인튜닝'),
    ('AdvancedDefectPredictor', None, _CK['AdvancedDefectPredictor'], 'Multi-output(분류+심각도+신뢰도)'),
]


def bootstrap_ci(trues, preds, n_boot=1000, seed=SEED):
    """테스트셋 재표집으로 macro F1 의 95% 신뢰구간 산출 (모델 간 차이가 유의한지 판단용)."""
    rng = np.random.default_rng(seed)
    n = len(trues); vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = f1_score(trues[idx], preds[idx], average='macro', zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


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
        if not ckpt_rel or not (ROOT / ckpt_rel).exists():
            print(f'[SKIP] {name}: 체크포인트 없음 ({ckpt_rel})')
            continue
        path = ROOT / ckpt_rel
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
        p_mac, r_mac, _, _ = precision_recall_fscore_support(
            trues, preds, average='macro', zero_division=0)
        ci_lo, ci_hi = bootstrap_ci(trues, preds)
        n_params = sum(p.numel() for p in model.parameters())

        results[name] = {
            'checkpoint': ckpt_rel,
            'note': note,
            'params_M': round(n_params / 1e6, 2),
            'test_accuracy': round(float(acc), 4),
            'test_f1_macro': round(float(f1_mac), 4),
            'test_f1_weighted': round(float(f1_wtd), 4),
            'test_precision_macro': round(float(p_mac), 4),
            'test_recall_macro': round(float(r_mac), 4),
            'test_f1_macro_ci95': [round(ci_lo, 4), round(ci_hi, 4)],
        }
        print(f'{name:<26} Acc {acc * 100:6.2f}%  macroF1 {f1_mac:.4f} '
              f'[{ci_lo:.4f}, {ci_hi:.4f}]  wtdF1 {f1_wtd:.4f}  '
              f'macroP {p_mac:.4f} macroR {r_mac:.4f}  ({n_params / 1e6:.2f}M params)')

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
        'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None,
        'seed': SEED,
        'ci_note': 'test_f1_macro_ci95 = 테스트셋 bootstrap 1000회 재표집 95% 구간',
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
