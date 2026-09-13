"""동일 프로토콜 공정 비교 실험 (Fair Comparison) — "커스텀 CNN > MobileNet" 검증용

[문제 정의]
기존 파이프라인은 모델마다 학습 레시피가 다르다.
  - WaferCNN          : from-scratch, Adam lr 3e-4 (전체 파라미터), 최대 40 epochs, patience 10
  - 사전학습 3종        : 2-Phase (P1 head-only 5ep @1e-3 → P2 backbone lr 1e-5 / head 1e-4, 35ep)
따라서 "WaferCNN(0.8458) > MobileNetV3(0.7369)" 가 아키텍처 차이인지 레시피 차이인지 분리할 수 없다.

[설계]
모든 모델을 같은 데이터·증강·WeightedRandomSampler·CrossEntropy·AdamW·epoch 예산·
체크포인트 선택 기준(val macro F1)으로 학습하고, 같은 테스트셋(25,943개)에서 평가한다.
seed 를 여러 개 돌려 평균±표준편차와 bootstrap 95% CI 를 함께 기록한다.

[MobileNetV3 진단 변인]
  mv3_pre64      ImageNet 사전학습 · 64×64 그대로 (기존 입력)   → stride 32 → 최종 feature map 2×2
  mv3_pre128     ImageNet 사전학습 · 128×128 nearest 업샘플     → 최종 4×4 (WaferCNN 과 동일 해상도)
  mv3_scratch64  사전학습 없음 · 64×64                          → ImageNet 전이 효과 분리
  mv3_scratch128 사전학습 없음 · 128×128
  wafercnn       기준 모델
  effb0_pre128 / vit_pre64  참고 (기존 파인튜닝 대상)

[사용]
  python scripts/fair_compare.py --models wafercnn mv3_pre64 mv3_pre128 mv3_scratch64 --seeds 42 43 44
  python scripts/fair_compare.py --summarize          # analysis/fair_compare/summary.{json,csv,md}
"""
import os, sys, json, time, random, pickle, argparse, warnings, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix)

warnings.filterwarnings('ignore')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import WaferMapDataset, build_train_transform, CLASS_ORDER  # noqa: E402
from src.model import WaferCNN  # noqa: E402

NUM_CLASSES = len(CLASS_ORDER)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = ROOT / 'analysis/fair_compare'
RUN_DIR = OUT_DIR / 'runs'
CKPT_DIR = ROOT / 'checkpoints/fair'

# ── 공통 학습 프로토콜 (모든 모델 동일) ─────────────────────────────
# patience=0 → 조기종료 없음. val macro F1 은 epoch 간 ±0.03 씩 요동하므로(소수 클래스 22~83개),
# 조기종료를 쓰면 "LR 이 아직 높은 시점의 운 좋은 epoch" 이 선택되어 비교가 seed 운에 좌우된다.
# 대신 cosine 스케줄을 끝까지 돌리고 (a) best-val 체크포인트 (b) 마지막 epoch 두 가지를 모두 평가한다.
PROTOCOL = dict(batch_size=64, lr=3e-4, weight_decay=1e-4, epochs=25, patience=0,
                optimizer='AdamW', scheduler='CosineAnnealingLR(eta_min=1e-6)',
                loss='CrossEntropyLoss (weight 없음)', sampler='WeightedRandomSampler(balanced)',
                augmentation='src.data_loader.build_train_transform',
                select_by='val macro F1 (best) + last epoch 병기', amp='bf16 autocast')


# ── 모델 정의 ───────────────────────────────────────────────────────
class Upsampled(nn.Module):
    """입력을 nearest 로 확대한 뒤 백본에 전달 (픽셀값 0/0.5/1 유지)."""
    def __init__(self, backbone, size):
        super().__init__()
        self.backbone = backbone
        self.size = size

    def forward(self, x):
        if self.size and x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='nearest')
        return self.backbone(x)


def _mv3(pretrained):
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    old = m.features[0][0]
    new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    if pretrained:
        new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
    m.features[0][0] = new
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, NUM_CLASSES)
    return m


def _effb0(pretrained):
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    old = m.features[0][0]
    new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    if pretrained:
        new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
    m.features[0][0] = new
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, NUM_CLASSES)
    return m


def _vit(pretrained, img_size=64):
    import timm
    return timm.create_model('vit_tiny_patch16_224', pretrained=pretrained,
                             num_classes=NUM_CLASSES, in_chans=1, img_size=img_size)


MODEL_ZOO = {
    # key            : (설명, 빌더)
    'wafercnn':       ('WaferCNN (custom, scratch, 64px)',            lambda: WaferCNN(NUM_CLASSES, 0.3)),
    'mv3_pre64':      ('MobileNetV3-S (ImageNet, 64px)',              lambda: Upsampled(_mv3(True), None)),
    'mv3_pre128':     ('MobileNetV3-S (ImageNet, 128px upsample)',    lambda: Upsampled(_mv3(True), 128)),
    'mv3_scratch64':  ('MobileNetV3-S (scratch, 64px)',               lambda: Upsampled(_mv3(False), None)),
    'mv3_scratch128': ('MobileNetV3-S (scratch, 128px upsample)',     lambda: Upsampled(_mv3(False), 128)),
    'effb0_pre128':   ('EfficientNet-B0 (ImageNet, 128px upsample)',  lambda: Upsampled(_effb0(True), 128)),
    'effb0_pre64':    ('EfficientNet-B0 (ImageNet, 64px)',            lambda: Upsampled(_effb0(True), None)),
    'vit_pre64':      ('ViT-Tiny/16 (ImageNet, 64px → 16 tokens)',    lambda: _vit(True, 64)),
}


# ── 유틸 ────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True   # 속도 우선 (완전 결정성은 보장하지 않음 → seed 반복으로 보정)


def load_data():
    maps = np.load(ROOT / 'data/processed/all_maps_resized.npy')
    with open(ROOT / 'data/processed/split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
    y = split['encoded_labels'].astype(int)
    return maps, split, y


def make_loaders(maps, split, y, batch_size, num_workers, seed):
    tr, va, te = split['train_idx'], split['val_idx'], split['test_idx']
    cw = split['class_weights']
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = WeightedRandomSampler(torch.DoubleTensor(cw[y[tr]]), len(tr), replacement=True, generator=g)
    kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)
    train = DataLoader(WaferMapDataset(maps[tr], y[tr], build_train_transform()),
                       batch_size, sampler=sampler, **kw)
    val = DataLoader(WaferMapDataset(maps[va], y[va]), 256, shuffle=False, num_workers=0)
    test = DataLoader(WaferMapDataset(maps[te], y[te]), 256, shuffle=False, num_workers=0)
    return train, val, test


@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, T = [], []
    for x, yb in loader:
        out = model(x.to(DEVICE, non_blocking=True))
        P.append(out.argmax(1).cpu().numpy())
        T.append(yb.numpy())
    return np.concatenate(P), np.concatenate(T)


def bootstrap_ci(trues, preds, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(trues)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = f1_score(trues[idx], preds[idx], average='macro', zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def full_metrics(trues, preds):
    p, r, f, s = precision_recall_fscore_support(trues, preds, labels=range(NUM_CLASSES), zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(trues, preds, average='macro', zero_division=0)
    lo, hi = bootstrap_ci(trues, preds)
    return {
        'accuracy': float(accuracy_score(trues, preds)),
        'f1_macro': float(fm), 'f1_macro_ci95': [lo, hi],
        'f1_weighted': float(f1_score(trues, preds, average='weighted', zero_division=0)),
        'precision_macro': float(pm), 'recall_macro': float(rm),
        'per_class': {c: {'support': int(s[i]), 'precision': float(p[i]),
                          'recall': float(r[i]), 'f1': float(f[i])} for i, c in enumerate(CLASS_ORDER)},
        'confusion_matrix': confusion_matrix(trues, preds, labels=range(NUM_CLASSES)).tolist(),
    }


# ── 한 run ──────────────────────────────────────────────────────────
def run_one(key, seed, args, data):
    desc, builder = MODEL_ZOO[key]
    tag = getattr(args, 'tag', '') or ''
    run_key = f'{key}@{tag}' if tag else key          # 예: wafercnn@ep40 (프로토콜 변형 실험 구분)
    if tag:
        desc = f'{desc} [{tag}]'
    out_json = RUN_DIR / f'{run_key.replace("@", "_")}_seed{seed}.json'
    if out_json.exists() and not args.force:
        print(f'[SKIP] {key} seed={seed} (이미 존재: {out_json.name})')
        return
    set_seed(seed)
    maps, split, y = data
    train_loader, val_loader, test_loader = make_loaders(maps, split, y, args.batch_size,
                                                         args.num_workers, seed)
    model = builder().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    print(f'\n=== {key} | {desc} | seed={seed} | {n_params/1e6:.2f}M params | {DEVICE} ===', flush=True)
    best_f1, best_state, best_ep, bad = -1.0, None, 0, 0
    history = []
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss = tot_n = 0
        for x, yb in train_loader:
            x, yb = x.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE.type == 'cuda' and not args.no_amp)):
                loss = criterion(model(x), yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * x.size(0)
            tot_n += x.size(0)
        sch.step()
        vp, vt = predict(model, val_loader)
        vf1 = f1_score(vt, vp, average='macro', zero_division=0)
        vacc = accuracy_score(vt, vp)
        is_best = vf1 > best_f1
        if is_best:
            best_f1, best_ep, bad = vf1, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        dt = time.time() - t0
        history.append({'epoch': ep, 'train_loss': tot_loss / tot_n, 'val_f1_macro': float(vf1),
                        'val_acc': float(vacc), 'sec': dt})
        print(f'  ep {ep:>2}/{args.epochs} loss {tot_loss/tot_n:.4f}  val_acc {vacc:.4f}  '
              f'val_f1 {vf1:.4f} {"*" if is_best else " "}  {dt:5.0f}s', flush=True)
        if args.patience > 0 and bad >= args.patience:
            print(f'  early stop @ {ep} (best ep {best_ep})')
            break
    train_sec = time.time() - t_start

    # (b) 마지막 epoch 모델 — 선택 편향 없음
    tp, tt = predict(model, test_loader)
    test_last = full_metrics(tt, tp)
    # (a) best-val 체크포인트
    model.load_state_dict(best_state)
    tp, tt = predict(model, test_loader)
    test = full_metrics(tt, tp)
    vp, vt = predict(model, val_loader)
    val = {'f1_macro': float(f1_score(vt, vp, average='macro', zero_division=0)),
           'accuracy': float(accuracy_score(vt, vp))}

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f'{run_key.replace("@", "_")}_seed{seed}.pth'
    torch.save({'model_key': run_key, 'seed': seed, 'epoch': best_ep, 'model_state': best_state,
                'val_f1': best_f1, 'protocol': {**PROTOCOL, 'epochs': args.epochs,
                                               'patience': args.patience}}, ckpt_path)
    rec = {'model_key': run_key, 'description': desc, 'seed': seed, 'params_M': round(n_params / 1e6, 3),
           'device': str(DEVICE), 'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None,
           'protocol': {**PROTOCOL, 'epochs': args.epochs, 'patience': args.patience,
                        'lr': args.lr, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size},
           'best_epoch': best_ep, 'epochs_run': len(history), 'train_sec': round(train_sec, 1),
           'val': val, 'test': test, 'test_last_epoch': test_last, 'history': history,
           'checkpoint': ckpt_path.relative_to(ROOT).as_posix()}
    out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    lo, hi = test['f1_macro_ci95']
    print(f'  -> TEST(best-val ep{best_ep}) acc {test["accuracy"]:.4f}  macroF1 {test["f1_macro"]:.4f} [{lo:.4f},{hi:.4f}]  '
          f'wF1 {test["f1_weighted"]:.4f}  P {test["precision_macro"]:.4f} R {test["recall_macro"]:.4f}', flush=True)
    print(f'  -> TEST(last ep{len(history)})    acc {test_last["accuracy"]:.4f}  macroF1 {test_last["f1_macro"]:.4f}  '
          f'wF1 {test_last["f1_weighted"]:.4f}  P {test_last["precision_macro"]:.4f} R {test_last["recall_macro"]:.4f}  '
          f'({train_sec/60:.1f} min)  saved {out_json.name}', flush=True)
    del model
    torch.cuda.empty_cache()


# ── 집계 ────────────────────────────────────────────────────────────
def summarize():
    import pandas as pd
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(RUN_DIR.glob('*.json'))]
    if not runs:
        print('run 결과 없음')
        return
    rows = []
    for r in runs:
        t = r['test']
        row = {'model_key': r['model_key'], 'description': r['description'], 'seed': r['seed'],
               'params_M': r['params_M'], 'best_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'],
               'val_f1_macro': r['val']['f1_macro'], 'test_accuracy': t['accuracy'],
               'test_f1_macro': t['f1_macro'], 'test_f1_weighted': t['f1_weighted'],
               'test_precision_macro': t['precision_macro'], 'test_recall_macro': t['recall_macro'],
               'ci_lo': t['f1_macro_ci95'][0], 'ci_hi': t['f1_macro_ci95'][1],
               'last_test_accuracy': r.get('test_last_epoch', {}).get('accuracy', float('nan')),
               'last_test_f1_macro': r.get('test_last_epoch', {}).get('f1_macro', float('nan')),
               'train_min': r['train_sec'] / 60}
        for c in CLASS_ORDER:
            row[f'f1_{c}'] = t['per_class'][c]['f1']
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'runs.csv', index=False, encoding='utf-8-sig')

    zoo = list(MODEL_ZOO)
    order = sorted(set(df.model_key), key=lambda k: (zoo.index(k.split('@')[0]), k))
    agg_spec = dict(description=('description', 'first'), params_M=('params_M', 'first'),
                    n_seeds=('seed', 'count'),
                    acc_mean=('test_accuracy', 'mean'), acc_std=('test_accuracy', 'std'),
                    f1_mean=('test_f1_macro', 'mean'), f1_std=('test_f1_macro', 'std'),
                    f1_min=('test_f1_macro', 'min'), f1_max=('test_f1_macro', 'max'),
                    last_f1_mean=('last_test_f1_macro', 'mean'), last_f1_std=('last_test_f1_macro', 'std'),
                    last_acc_mean=('last_test_accuracy', 'mean'),
                    wf1_mean=('test_f1_weighted', 'mean'),
                    prec_mean=('test_precision_macro', 'mean'), rec_mean=('test_recall_macro', 'mean'),
                    val_f1_mean=('val_f1_macro', 'mean'), best_epoch_mean=('best_epoch', 'mean'),
                    train_min_mean=('train_min', 'mean'))
    for c in CLASS_ORDER:
        agg_spec[f'f1_{c}_mean'] = (f'f1_{c}', 'mean')
    agg = df.groupby('model_key').agg(**agg_spec).reindex(order).reset_index().fillna(0.0)
    agg.to_csv(OUT_DIR / 'summary.csv', index=False, encoding='utf-8-sig')

    summary = {'protocol': PROTOCOL, 'n_runs': len(runs), 'models': {}}
    for _, a in agg.iterrows():
        d = {}
        for k, v in a.items():
            if k == 'model_key':
                continue
            if isinstance(v, (float, np.floating)):
                d[k] = round(float(v), 4)
            elif isinstance(v, np.integer):
                d[k] = int(v)
            else:
                d[k] = v
        summary['models'][a.model_key] = d
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    ep0 = runs[0]['protocol']
    md = ['# 동일 프로토콜 공정 비교 결과 (Test 25,943개)', '',
          f'프로토콜: {PROTOCOL["optimizer"]} lr {PROTOCOL["lr"]} · wd {PROTOCOL["weight_decay"]} · '
          f'batch {PROTOCOL["batch_size"]} · {PROTOCOL["scheduler"]} · {PROTOCOL["loss"]} · '
          f'{PROTOCOL["sampler"]} · 최대 {ep0["epochs"]} epochs / patience {ep0["patience"]} · '
          f'선택 기준 {PROTOCOL["select_by"]}', '',
          '| 모델 | 파라미터 | seeds | Accuracy | **macro F1 (best-val ckpt, mean±std)** | min–max | macro F1 (last epoch, mean±std) | weighted F1 | macro P | macro R |',
          '|---|---:|:-:|---:|---:|---:|---:|---:|---:|---:|']
    for _, a in agg.iterrows():
        md.append(f'| {a.description} | {a.params_M:.2f}M | {int(a.n_seeds)} | {a.acc_mean*100:.2f}% | '
                  f'**{a.f1_mean:.4f} ± {a.f1_std:.4f}** | {a.f1_min:.4f}–{a.f1_max:.4f} | '
                  f'{a.last_f1_mean:.4f} ± {a.last_f1_std:.4f} | '
                  f'{a.wf1_mean:.4f} | {a.prec_mean:.4f} | {a.rec_mean:.4f} |')
    md += ['', '## 클래스별 F1 (seed 평균)', '',
           '| 모델 | ' + ' | '.join(CLASS_ORDER) + ' |', '|---|' + '---:|' * NUM_CLASSES]
    for _, a in agg.iterrows():
        cells = ' | '.join(f'{a[f"f1_{c}_mean"]:.3f}' for c in CLASS_ORDER)
        md.append(f'| {a.description} | {cells} |')
    md += ['', '## run 별 원자료', '',
           '| model | seed | best ep | val F1 | test Acc | test macro F1 | 95% CI | last-ep test F1 | 학습(min) |',
           '|---|:-:|:-:|---:|---:|---:|---|---:|---:|']
    for _, r in df.sort_values(['model_key', 'seed']).iterrows():
        md.append(f'| {r.model_key} | {r.seed} | {r.best_epoch} | {r.val_f1_macro:.4f} | {r.test_accuracy:.4f} | '
                  f'{r.test_f1_macro:.4f} | [{r.ci_lo:.4f}, {r.ci_hi:.4f}] | {r.last_test_f1_macro:.4f} | {r.train_min:.1f} |')
    (OUT_DIR / 'summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md[:6 + len(agg)]))
    print(f'\n저장: {OUT_DIR / "summary.md"}, summary.json, summary.csv, runs.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['wafercnn', 'mv3_pre64', 'mv3_pre128', 'mv3_scratch64'],
                    choices=list(MODEL_ZOO))
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=PROTOCOL['epochs'])
    ap.add_argument('--patience', type=int, default=PROTOCOL['patience'])
    ap.add_argument('--lr', type=float, default=PROTOCOL['lr'])
    ap.add_argument('--weight_decay', type=float, default=PROTOCOL['weight_decay'])
    ap.add_argument('--batch_size', type=int, default=PROTOCOL['batch_size'])
    ap.add_argument('--num_workers', type=int, default=int(os.environ.get('NUM_WORKERS', '4')))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--no_amp', action='store_true', help='bf16 autocast 비활성화')
    ap.add_argument('--tag', default='', help='프로토콜 변형 실험 구분용 접미사 (예: ep40)')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    if args.summarize:
        summarize()
        return
    print(f'Device: {DEVICE}' + (f' ({torch.cuda.get_device_name(0)})' if DEVICE.type == 'cuda' else ''))
    data = load_data()
    for seed in args.seeds:          # seed 바깥 루프: 모델 간 비교가 같은 seed 로 먼저 채워지도록
        for key in args.models:
            run_one(key, seed, args, data)
    summarize()


if __name__ == '__main__':
    main()
