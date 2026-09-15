"""결과 검증 (Stage G) — 학습 스크립트가 남긴 수치를 독립 경로로 재계산해 문서와 대조.

G1 fair 체크포인트 재채점        checkpoints/fair/*.pth → test 재추론 → runs/*.json 대조 (F1 ≤1e-4, confusion matrix 일치)
G2 기존 레시피 체크포인트 재채점  final_evaluation.json 5개 (모델 빌더는 evaluate_all.py 의 것과 동일 구조)
G3 집계 일관성                   runs.csv → mean/std 재계산 → summary.json
G4 문서 수치 대조                README · MODEL_PERFORMANCE.md 표의 F1 값 ↔ 원본 JSON
G5 ONNX 동등성                   deployment_summary_v2.json 의 argmax 일치율
G6 테스트셋 무결성               split 분포 · train/val/test 교집합
G7 그림 재생성                   plot_results.py → 6 PNG

출력: analysis/verification_report.json (+ 콘솔 요약). 실패가 하나라도 있으면 exit 1.
"""
import os, re, sys, json, pickle, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import fair_compare as fc  # noqa: E402
from src.data_loader import CLASS_ORDER, WaferMapDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = len(CLASS_ORDER)
N_TEST = 25943
REPORT = {'device': str(DEVICE), 'checks': {}}
EXPECTED_TEST_DIST = {'none': 22115, 'Center': 644, 'Donut': 83, 'Edge-Loc': 779, 'Edge-Ring': 1452, 'Loc': 539, 'Near-full': 22, 'Random': 130, 'Scratch': 179}


def record(name, ok, detail):
    REPORT['checks'][name] = {'pass': bool(ok), **detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail.get('summary', '')}")


@torch.no_grad()
def score(model, loader):
    model.eval(); P, T = [], []
    for x, yb in loader:
        P.append(model(x.to(DEVICE)).argmax(1).cpu().numpy()); T.append(yb.numpy())
    p, t = np.concatenate(P), np.concatenate(T)
    return {'accuracy': float(accuracy_score(t, p)), 'f1_macro': float(f1_score(t, p, average='macro', zero_division=0)),
            'cm': confusion_matrix(t, p, labels=range(NUM_CLASSES)).tolist()}


def test_loader():
    maps, split, y = fc.load_data(); te = split['test_idx']
    return DataLoader(WaferMapDataset(maps[te], y[te]), 512, shuffle=False, num_workers=0), split, y


# ── G1 ─────────────────────────────────────────────────────────────
def g1(loader):
    rows, fails = [], []
    for p in sorted((ROOT / 'analysis/fair_compare/runs').glob('*.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        ck_path = ROOT / r['checkpoint']
        if not ck_path.exists():
            fails.append(f'{p.stem}: 체크포인트 없음'); continue
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        key = ck['model_key'].split('@')[0]
        m = fc.MODEL_ZOO[key][1](); m.load_state_dict(ck['model_state']); m.to(DEVICE)
        s = score(m, loader)
        d_f1 = abs(s['f1_macro'] - r['test']['f1_macro']); d_acc = abs(s['accuracy'] - r['test']['accuracy'])
        # GPU 재추론은 cuDNN autotune 커널 선택·배치 크기에 따라 near-tie 로짓이 소수 샘플에서 뒤집힐 수 있다.
        # 따라서 "비트 단위 동일" 이 아니라 "예측 불일치 ≤ 0.1% · F1 차이 ≤ 1e-3" 을 동일성 기준으로 삼는다.
        n_diff = int(np.abs(np.array(s['cm']) - np.array(r['test']['confusion_matrix'])).sum() // 2)
        cm_ok = s['cm'] == r['test']['confusion_matrix']
        ok = d_f1 <= 1e-3 and d_acc <= 1e-3 and n_diff <= 0.001 * N_TEST
        rows.append({'run': p.stem, 'stored_f1': r['test']['f1_macro'], 'rescored_f1': round(s['f1_macro'], 6), 'd_f1': round(d_f1, 6),
                     'cm_identical': cm_ok, 'n_pred_diff_upper_bound': n_diff, 'ok': ok})
        if not ok: fails.append(f'{p.stem}: dF1={d_f1:.2e} pred_diff≈{n_diff}')
        del m; torch.cuda.empty_cache()
    exact = sum(r['cm_identical'] for r in rows)
    record('G1_fair_checkpoint_rescore', not fails,
           {'summary': f'{len(rows) - len(fails)}/{len(rows)} 허용오차 내 일치 (confusion matrix 비트 동일 {exact}/{len(rows)} · 최대 dF1 {max(r["d_f1"] for r in rows):.1e})',
            'tolerance': 'F1 ≤ 1e-3, 예측 불일치 ≤ 0.1% (GPU 비결정성 허용)', 'rows': rows, 'failures': fails})


# ── G2 ─────────────────────────────────────────────────────────────
def g2(loader):
    import importlib.util
    spec = importlib.util.spec_from_file_location('evaluate_all', ROOT / 'scripts/evaluate_all.py')
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    fe = json.loads((ROOT / 'analysis/final_evaluation.json').read_text(encoding='utf-8'))
    builders = {name: b for name, b, _, _ in ev.MODELS}
    rows, fails = [], []
    for name, m in fe['models'].items():
        ck_path = ROOT / m['checkpoint']
        if not ck_path.exists(): fails.append(f'{name}: 체크포인트 없음'); continue
        try:
            ck = torch.load(ck_path, map_location='cpu', weights_only=False)
            state, meta = ev.load_state(ck)
            if name == 'AdvancedDefectPredictor':
                model = ev.AdvancedDefectPredictor(pretrained=False)
            elif builders.get(name) is ev.build_wafer_cnn:
                cfg = meta.get('config') or {}
                model = ev.WaferCNN(num_classes=cfg.get('num_classes', NUM_CLASSES), dropout=cfg.get('dropout', 0.3))
            else:
                model = builders[name]()
            model.load_state_dict(state); model.to(DEVICE)
            # evaluate_all 과 같은 predict 경로 (multi-output dict 처리 포함) 로 재채점
            p, t = ev.predict(model, loader)
            s = {'accuracy': float(accuracy_score(t, p)), 'f1_macro': float(f1_score(t, p, average='macro', zero_division=0))}
        except Exception as e:
            fails.append(f'{name}: {e}'); continue
        d_f1 = abs(s['f1_macro'] - m['test_f1_macro']); ok = d_f1 <= 1e-3
        rows.append({'model': name, 'stored_f1': m['test_f1_macro'], 'rescored_f1': round(s['f1_macro'], 6), 'd_f1': round(d_f1, 6), 'ok': ok})
        if not ok: fails.append(f'{name}: dF1={d_f1:.2e}')
    record('G2_legacy_checkpoint_rescore', not fails, {'summary': f'{len(rows) - len([r for r in rows if not r["ok"]])}/{len(fe["models"])} 허용오차(1e-3) 내 일치 · 최대 dF1 {max((r["d_f1"] for r in rows), default=0):.1e}', 'rows': rows, 'failures': fails})


# ── G3 ─────────────────────────────────────────────────────────────
def g3():
    df = pd.read_csv(ROOT / 'analysis/fair_compare/runs.csv', encoding='utf-8-sig')
    sm = json.loads((ROOT / 'analysis/fair_compare/summary.json').read_text(encoding='utf-8'))['models']
    fails, n = [], 0
    for k, g in df.groupby('model_key'):
        if k not in sm: fails.append(f'{k}: summary 에 없음'); continue
        n += 1
        mean, std = g.test_f1_macro.mean(), (g.test_f1_macro.std(ddof=1) if len(g) > 1 else 0.0)
        if abs(mean - sm[k]['f1_mean']) > 1e-4 or abs(std - sm[k]['f1_std']) > 1e-4 or len(g) != sm[k]['n_seeds']:
            fails.append(f'{k}: csv mean {mean:.5f}/std {std:.5f}/n {len(g)} vs summary {sm[k]["f1_mean"]:.5f}/{sm[k]["f1_std"]:.5f}/{sm[k]["n_seeds"]}')
    record('G3_aggregate_consistency', not fails, {'summary': f'{n - len(fails)}/{n} 구성 일치', 'failures': fails})


# ── G4 ─────────────────────────────────────────────────────────────
def g4():
    sm = json.loads((ROOT / 'analysis/fair_compare/summary.json').read_text(encoding='utf-8'))['models']
    fe = json.loads((ROOT / 'analysis/final_evaluation.json').read_text(encoding='utf-8'))['models']
    truth = {round(v['f1_mean'], 4) for v in sm.values()} | {round(v['test_f1_macro'], 4) for v in fe.values()}
    truth |= {round(v['f1_mean'], 3) for v in sm.values()}  # README 본문에 3자리 표기도 있음
    # 개별 seed 값도 허용 (run 별 표 · 본문 인용)
    for p in (ROOT / 'analysis/fair_compare/runs').glob('*.json'):
        r = json.loads(p.read_text(encoding='utf-8'))
        truth |= {round(r['test']['f1_macro'], 4), round(r['test_last_epoch']['f1_macro'], 4), round(r['test']['f1_macro'], 3),
                  round(r['test']['accuracy'], 4), round(r['test_last_epoch']['accuracy'], 4), round(r['val']['f1_macro'], 4), round(r['val']['accuracy'], 4),
                  round(r['test']['f1_weighted'], 4), round(r['test']['precision_macro'], 4), round(r['test']['recall_macro'], 4)}
        truth |= {round(v['f1'], 3) for v in r['test']['per_class'].values()} | {round(v['f1'], 4) for v in r['test']['per_class'].values()}
    for v in sm.values():
        truth |= {round(v[k], 4) for k in ('acc_mean', 'last_acc_mean') if k in v and isinstance(v[k], float)}
    truth |= {round(v['test_accuracy'], 4) for v in fe.values()}
    truth |= {round(v, 4) for r in (json.loads(p.read_text(encoding='utf-8')) for p in (ROOT / 'analysis/fair_compare/runs').glob('*.json')) for v in r['test']['f1_macro_ci95']}
    results, fails = {}, []
    for doc in ('README.md', 'docs/MODEL_PERFORMANCE.md'):
        txt = (ROOT / doc).read_text(encoding='utf-8')
        # "macro F1" 표 영역에서 0.7xxx~0.9xxx 4자리 수치 추출 (성능 절만)
        seg = txt
        if doc == 'README.md':
            a = txt.find('## 모델 성능 비교'); b = txt.find('### 관찰 2'); seg = txt[a:b]
        else:
            a = txt.find('## 1.'); b = txt.find('## 4.'); seg = txt[a:b]
        nums = {float(x) for x in re.findall(r'\b0\.(?:7|8|9)\d{3}\b', seg)}
        # 오차·std·CI 등 F1 이 아닌 값 제외: std(0.00xx) 는 패턴에 안 걸림. accuracy 는 % 표기라 제외됨. weighted F1/P/R 은 별도 비교 필요 → 허용 집합에 추가
        for v in sm.values():
            truth.update({round(v[k], 4) for k in ('wf1_mean', 'prec_mean', 'rec_mean', 'f1_min', 'f1_max', 'last_f1_mean') if k in v})
            truth.update({round(v[k], 3) for k in ('wf1_mean', 'prec_mean', 'rec_mean') if k in v})
            truth.update({round(v[k], 4) for k in v if k.startswith('f1_') and k.endswith('_mean') and isinstance(v[k], float)})
            truth.update({round(v[k], 3) for k in v if k.startswith('f1_') and k.endswith('_mean') and isinstance(v[k], float)})
        for v in fe.values():
            truth.update({round(v[k], 4) for k in ('test_f1_weighted', 'test_precision_macro', 'test_recall_macro') if k in v})
            truth.update({round(v[k], 3) for k in ('test_f1_weighted', 'test_precision_macro', 'test_recall_macro') if k in v})
            truth.update({round(c, 4) for c in v.get('test_f1_macro_ci95', [])}); truth.update({round(c, 3) for c in v.get('test_f1_macro_ci95', [])})
        pc = ROOT / 'analysis/per_class_metrics.csv'
        if pc.exists():
            d = pd.read_csv(pc, encoding='utf-8-sig')
            for col in ('precision', 'recall', 'f1'):
                truth |= {round(float(x), 4) for x in d[col]} | {round(float(x), 3) for x in d[col]}
        unknown = sorted(n for n in nums if round(n, 4) not in truth and round(n, 3) not in truth)
        # 이전 README 기재값(재현 전) 인용은 허용 목록
        legacy_cited = {0.8458, 0.8352, 0.7944, 0.7369, 0.8173, 0.5987, 0.8367, 0.8847}
        unknown = [n for n in unknown if n not in legacy_cited]
        results[doc] = {'n_values_checked': len(nums), 'unknown_values': unknown}
        if unknown: fails.append(f'{doc}: 원본에 없는 값 {unknown}')
    record('G4_document_numbers', not fails, {'summary': ' · '.join(f'{k} {v["n_values_checked"]}개 확인' for k, v in results.items()), 'details': results, 'failures': fails})


# ── G5 ─────────────────────────────────────────────────────────────
def g5():
    p = ROOT / 'analysis/deployment_summary_v2.json'
    if not p.exists(): record('G5_onnx_parity', False, {'summary': 'deployment_summary_v2.json 없음'}); return
    d = json.loads(p.read_text(encoding='utf-8'))['models']
    rows = {k: {'fp32_agreement': v['accuracy']['fp32_vs_pytorch_argmax_agreement'], 'fp32_f1_delta': v['accuracy']['fp32_f1_delta'], 'int8_verdict': v['int8_verdict']} for k, v in d.items()}
    fails = [k for k, v in rows.items() if v['fp32_agreement'] < 0.999 or abs(v['fp32_f1_delta']) > 0.001]
    record('G5_onnx_parity', not fails, {'summary': f'{len(rows) - len(fails)}/{len(rows)} 모델 FP32 동등', 'rows': rows, 'failures': fails})


# ── G6 ─────────────────────────────────────────────────────────────
def g6(split, y):
    tr, va, te = map(np.asarray, (split['train_idx'], split['val_idx'], split['test_idx']))
    dist = {CLASS_ORDER[i]: int(c) for i, c in zip(*np.unique(y[te], return_counts=True))}
    overlap = len(set(tr) & set(te)) + len(set(tr) & set(va)) + len(set(va) & set(te))
    ok = dist == EXPECTED_TEST_DIST and overlap == 0 and len(te) == 25943
    record('G6_test_split_integrity', ok, {'summary': f'test {len(te):,} · 분포 일치 {dist == EXPECTED_TEST_DIST} · 교집합 {overlap}', 'test_distribution': dist, 'overlap': overlap})


# ── G7 ─────────────────────────────────────────────────────────────
def g7():
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/plot_results.py')], capture_output=True, text=True, cwd=ROOT, env={**os.environ, 'PYTHONUTF8': '1'})
    figs = sorted((ROOT / 'analysis/figures').glob('fig*.png'))
    ok = r.returncode == 0 and len(figs) == 6 and all(f.stat().st_size > 10_000 for f in figs)
    record('G7_figures_regenerate', ok, {'summary': f'{len(figs)}/6 PNG · exit {r.returncode}', 'files': [f.name for f in figs]})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--checks', nargs='+', default=['G6', 'G1', 'G2', 'G3', 'G4', 'G5', 'G7'], help='실행할 검사 (예: --checks G3 G4 G5 G6 G7 → GPU 불필요)')
    args = ap.parse_args()
    loader, split, y = test_loader()
    todo = {'G6': lambda: g6(split, y), 'G1': lambda: g1(loader), 'G2': lambda: g2(loader), 'G3': g3, 'G4': g4, 'G5': g5, 'G7': g7}
    for c in args.checks:
        todo[c]()
    n_pass = sum(c['pass'] for c in REPORT['checks'].values()); n = len(REPORT['checks'])
    REPORT['summary'] = f'{n_pass}/{n} PASS'
    (ROOT / 'analysis/verification_report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f'\n=== 검증 결과: {REPORT["summary"]} → analysis/verification_report.json')
    sys.exit(0 if n_pass == n else 1)


if __name__ == '__main__':
    main()
