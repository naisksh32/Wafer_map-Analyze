"""재학습·공정 비교 결과 시각화.

출력 (analysis/figures/):
  fig1_model_f1_bar.png          모델별 macro F1 막대 — 기존 레시피 vs 동일 레시피 (95% CI 오차막대)
  fig2_preprocess_f1_bar.png     모델별 입력 전처리 변인 막대 (MobileNetV3 해상도·보간·채널·정규화, ViT 토큰, EffNet 해상도)
  fig3_per_class_f1_bar.png      대표 모델 3종의 클래스별 F1 막대
  fig4_training_curves.png       모델별 학습 경과 꺾은선 — epoch 별 val accuracy / val macro F1 (small multiples)
  fig5_val_f1_overlay.png        핵심 모델 val macro F1 overlay (해상도 효과)
  fig6_legacy_curves.png         기존 레시피 재학습 로그 기반 학습 곡선

입력: analysis/final_evaluation.json · analysis/fair_compare/runs/*.json · logs/retrain_*.log
"""
import json, re, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'analysis/fair_compare/runs'
OUT = ROOT / 'analysis/figures'
OUT.mkdir(exist_ok=True)

# ── 스타일 (dataviz 참조 팔레트, light) ────────────────────────────
SURFACE, TEXT1, TEXT2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']  # 고정 순서
for f in ('Malgun Gothic', 'NanumGothic', 'Apple SD Gothic Neo'):
    if any(x.name == f for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break
plt.rcParams.update({'axes.unicode_minus': False, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                     'axes.edgecolor': GRID, 'axes.labelcolor': TEXT2, 'xtick.color': TEXT2, 'ytick.color': TEXT2,
                     'text.color': TEXT1, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'savefig.dpi': 160})

CLASS_ORDER = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']


def load_runs():
    runs = {}
    for p in sorted(RUNS.glob('*.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        runs.setdefault(r['model_key'], []).append(r)
    return runs


def agg(rs):
    f1 = np.array([r['test']['f1_macro'] for r in rs]); acc = np.array([r['test']['accuracy'] for r in rs])
    ci = rs[0]['test']['f1_macro_ci95'] if len(rs) == 1 else None
    return dict(f1=f1.mean(), std=f1.std(ddof=1) if len(rs) > 1 else 0.0, ci=ci, acc=acc.mean(), n=len(rs))


def hbar(ax, labels, vals, errs, colors, xlim=(0.70, 0.92), fmt='{:.4f}'):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.55, color=colors, xerr=errs, error_kw=dict(ecolor=TEXT2, elinewidth=1, capsize=3))
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.003, yi, fmt.format(v), va='center', ha='left', fontsize=9, color=TEXT1)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(*xlim); ax.grid(axis='y', visible=False)
    ax.set_xlabel('Test macro F1  (오차막대: 3 seeds ± std 또는 bootstrap 95% CI)')


# ── Fig 1: 기존 레시피 vs 동일 레시피 ─────────────────────────────
def fig1(runs):
    fe = json.loads((ROOT / 'analysis/final_evaluation.json').read_text(encoding='utf-8'))['models']
    pairs = [  # (표시명, 기존 레시피 키, 동일 레시피 키)
        ('WaferCNN (64px)', 'WaferCNN', 'wafercnn'),
        ('MobileNetV3-S (64px)', 'MobileNetV3-Small', 'mv3_pre64'),
        ('MobileNetV3-S (128px)', None, 'mv3_pre128'),
        ('EfficientNet-B0 (128px)', 'EfficientNet-B0', 'effb0_pre128'),
        ('ViT-Tiny (64px)', 'ViT-Tiny', 'vit_pre64'),
        ('AdvancedDefectPredictor', 'AdvancedDefectPredictor', None),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(pairs))[::-1]; h = 0.36
    for i, (name, lk, fk) in enumerate(pairs):
        if lk and lk in fe:
            m = fe[lk]; lo, hi = m['test_f1_macro_ci95']
            ax.barh(y[i] + h / 2, m['test_f1_macro'], height=h, color=C[1],
                    xerr=[[m['test_f1_macro'] - lo], [hi - m['test_f1_macro']]], error_kw=dict(ecolor=TEXT2, elinewidth=1, capsize=3),
                    label='기존 레시피 (WaferCNN: Adam 3e-4 · 사전학습: 2-Phase 백본 lr 1e-5)' if i == 0 else None)
            ax.text(hi + 0.003, y[i] + h / 2, f"{m['test_f1_macro']:.4f}", va='center', fontsize=9)
        if fk and fk in runs:
            a = agg(runs[fk]); err = a['std'] if a['n'] > 1 else (a['f1'] - a['ci'][0] + a['ci'][1] - a['f1']) / 2
            ax.barh(y[i] - h / 2, a['f1'], height=h, color=C[0], xerr=err, error_kw=dict(ecolor=TEXT2, elinewidth=1, capsize=3),
                    label='동일 레시피 (전 모델 AdamW 3e-4 · 25 ep · 조기종료 없음)' if fk == 'wafercnn' else None)
            ax.text(a['f1'] + err + 0.003, y[i] - h / 2, f"{a['f1']:.4f}" + (f" ±{a['std']:.4f} (n={a['n']})" if a['n'] > 1 else ''), va='center', fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([p[0] for p in pairs]); ax.set_xlim(0.70, 0.94); ax.grid(axis='y', visible=False)
    ax.set_ylim(-0.7, len(pairs) - 0.3)
    for tgt in (0.80, 0.88):
        ax.axvline(tgt, color=TEXT2, lw=1, ls=':')
        ax.text(tgt + 0.001, len(pairs) - 0.35, f'목표 {tgt:.2f}', fontsize=8, color=TEXT2, va='top')
    ax.set_xlabel('Test macro F1 (25,943개 · 오차막대 = 95% CI 또는 seeds ± std)')
    ax.set_title('모델별 macro F1 — 학습 레시피를 통일하면 순위가 뒤집힌다', loc='left', fontsize=12, fontweight='bold', pad=34)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, ncol=1)
    fig.tight_layout(); fig.savefig(OUT / 'fig1_model_f1_bar.png'); plt.close(fig)


# ── Fig 2: 입력 전처리 변인 ──────────────────────────────────────────
PREP_ROWS = [  # (표시명, 키, 그룹색 인덱스)
    ('WaferCNN 64px (4×4)', 'wafercnn', 3), ('WaferCNN 128px↑ (8×8)', 'wafercnn_128', 3),
    ('MobileNetV3-S 64px (2×2)', 'mv3_pre64', 0), ('MobileNetV3-S 128px nearest (4×4)', 'mv3_pre128', 0),
    ('MobileNetV3-S 128px bilinear', 'mv3_pre128_bil', 0), ('MobileNetV3-S 128px 3ch+ImageNet norm', 'mv3_pre128_3ch', 0),
    ('MobileNetV3-S 128px 1ch+ImageNet norm', 'mv3_pre128_norm', 0), ('MobileNetV3-S 160px (5×5)', 'mv3_pre160', 0),
    ('MobileNetV3-S 224px native (7×7)', 'mv3_pre224', 0), ('MobileNetV3-S scratch 64px', 'mv3_scratch64', 0),
    ('EfficientNet-B0 128px (4×4)', 'effb0_pre128', 2), ('EfficientNet-B0 224px native (7×7)', 'effb0_pre224', 2),
    ('ViT-Tiny 64px (16 tokens)', 'vit_pre64', 6), ('ViT-Tiny 128px (64 tokens)', 'vit_pre128', 6), ('ViT-Tiny 224px native (196 tokens)', 'vit_pre224', 6),
]


def fig2(runs):
    rows = [(n, k, c) for n, k, c in PREP_ROWS if k in runs]
    labels, vals, errs, cols = [], [], [], []
    for n, k, c in rows:
        a = agg(runs[k]); labels.append(n + (f'  [n={a["n"]}]' if a['n'] > 1 else '')); vals.append(a['f1']); cols.append(C[c])
        errs.append(a['std'] if a['n'] > 1 else (a['ci'][1] - a['ci'][0]) / 2)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 2.4))
    hbar(ax, labels, vals, errs, cols, xlim=(0.74, 0.93))
    ax.set_title('모델별 입력 전처리 변인 — 해상도가 유일하게 큰 변인 (동일 레시피 · seed 42, n 표기는 3 seeds 평균)', loc='left', fontsize=12, fontweight='bold', pad=28)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C[3], label='WaferCNN'), Patch(color=C[0], label='MobileNetV3-S'), Patch(color=C[2], label='EfficientNet-B0'), Patch(color=C[6], label='ViT-Tiny')],
              loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, ncol=4)
    fig.tight_layout(); fig.savefig(OUT / 'fig2_preprocess_f1_bar.png'); plt.close(fig)


# ── Fig 3: 클래스별 F1 ─────────────────────────────────────────────
def fig3(runs):
    picks = [('WaferCNN 64px', 'wafercnn', C[3]), ('MobileNetV3-S 64px', 'mv3_pre64', C[1]), ('MobileNetV3-S 128px', 'mv3_pre128', C[0])]
    best = max((k for k in runs if k.startswith(('mv3_pre', 'effb0'))), key=lambda k: agg(runs[k])['f1'])
    picks.append((runs[best][0]['description'], best, C[2]))
    fig, ax = plt.subplots(figsize=(12, 4.8)); x = np.arange(len(CLASS_ORDER)); w = 0.8 / len(picks)
    for i, (name, k, col) in enumerate(picks):
        vals = [np.mean([r['test']['per_class'][c]['f1'] for r in runs[k]]) for c in CLASS_ORDER]
        ax.bar(x + (i - len(picks) / 2 + 0.5) * w, vals, width=w * 0.92, color=col, label=name)
    ax.set_xticks(x); ax.set_xticklabels([f'{c}\n(n={runs["wafercnn"][0]["test"]["per_class"][c]["support"]:,})' for c in CLASS_ORDER], fontsize=9)
    ax.set_ylim(0.2, 1.0); ax.set_ylabel('Test F1'); ax.grid(axis='x', visible=False)
    ax.set_title('클래스별 F1 — 해상도 상향의 이득은 Scratch · Loc · Edge-Loc 에 집중된다', loc='left', fontsize=12, fontweight='bold', pad=28)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, ncol=4)
    fig.tight_layout(); fig.savefig(OUT / 'fig3_per_class_f1_bar.png'); plt.close(fig)


# ── Fig 4: 학습 경과 small multiples ─────────────────────────────────
def fig4(runs):
    keys = [k for _, k, _ in PREP_ROWS if k in runs]
    n = len(keys); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.9 * nrow), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, keys):
        r = sorted(runs[k], key=lambda r: r['seed'])[0]  # seed 42 대표
        ep = [h['epoch'] for h in r['history']]
        acc = [h['val_acc'] for h in r['history']]; f1 = [h['val_f1_macro'] for h in r['history']]
        ax.plot(ep, acc, color=C[0], lw=2, solid_capstyle='round'); ax.plot(ep, f1, color=C[1], lw=2, solid_capstyle='round')
        ax.scatter([ep[-1]], [acc[-1]], s=36, color=C[0], edgecolor=SURFACE, lw=2, zorder=3)
        ax.scatter([ep[-1]], [f1[-1]], s=36, color=C[1], edgecolor=SURFACE, lw=2, zorder=3)
        ax.axvline(r['best_epoch'], color=GRID, lw=1)
        ax.set_title(r['description'], fontsize=9, loc='left')
        ax.text(0.98, 0.04, f"test F1 {r['test']['f1_macro']:.4f}", transform=ax.transAxes, ha='right', fontsize=8, color=TEXT2)
        ax.set_ylim(0.6, 1.0); ax.set_xlim(1, max(ep))
    for ax in axes.flat[n:]: ax.axis('off')
    for ax in axes[-1]: ax.set_xlabel('epoch')
    for ax in axes[:, 0]: ax.set_ylabel('validation')
    fig.suptitle('모델별 학습 경과 (seed 42) — 파란선 val accuracy · 주황선 val macro F1 · 세로선 best-val epoch', x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.legend(handles=[plt.Line2D([], [], color=C[0], lw=2, label='val accuracy'), plt.Line2D([], [], color=C[1], lw=2, label='val macro F1')],
               loc='upper right', frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / 'fig4_training_curves.png'); plt.close(fig)


# ── Fig 5: 핵심 모델 val F1 overlay ───────────────────────────────────
def fig5(runs):
    picks = [('WaferCNN 64px', 'wafercnn'), ('MobileNetV3-S 64px', 'mv3_pre64'), ('MobileNetV3-S 128px', 'mv3_pre128'),
             ('MobileNetV3-S 160px', 'mv3_pre160'), ('MobileNetV3-S 224px', 'mv3_pre224'), ('EfficientNet-B0 128px', 'effb0_pre128')]
    picks = [(n, k) for n, k in picks if k in runs][:6]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ttl in zip(axes, ('val_acc', 'val_f1_macro'), ('validation accuracy', 'validation macro F1')):
        placed = []  # 끝점 라벨 충돌 방지: 가까운 값은 라벨 생략 (범례·툴팁이 대신함)
        gap = 0.004 if metric == 'val_acc' else 0.008
        for i, (name, k) in enumerate(picks):
            r = sorted(runs[k], key=lambda r: r['seed'])[0]; ep = [h['epoch'] for h in r['history']]; v = [h[metric] for h in r['history']]
            ax.plot(ep, v, color=C[i], lw=2, solid_capstyle='round', label=name)
            ax.scatter([ep[-1]], [v[-1]], s=40, color=C[i], edgecolor=SURFACE, lw=2, zorder=3)
            if all(abs(v[-1] - p) > gap for p in placed):
                ax.text(ep[-1] + 0.4, v[-1], f'{v[-1]:.3f}', va='center', fontsize=8, color=TEXT2); placed.append(v[-1])
        ax.set_xlabel('epoch'); ax.set_title(ttl, loc='left', fontsize=11); ax.set_xlim(1, 27)
        ax.set_ylim(0.85, 1.0) if metric == 'val_acc' else ax.set_ylim(0.65, 0.92)
    axes[1].legend(loc='lower right', frameon=False, fontsize=9)
    fig.suptitle('학습 경과 비교 (seed 42) — 해상도가 높을수록 초반부터 위에서 시작해 끝까지 유지된다', x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / 'fig5_val_f1_overlay.png'); plt.close(fig)


# ── Fig 6: 기존 레시피 로그 곡선 ───────────────────────────────────────
def parse_legacy():
    out = {}
    t = (ROOT / 'logs/retrain_baseline.log').read_text(encoding='utf-8', errors='ignore')
    rows = re.findall(r'^\s*(\d+) \|\s*[\d.e-]+ \|\s*([\d.]+) \|\s*([\d.]+) \|\s*([\d.]+) \|', t, re.M)
    out['WaferCNN (Adam 3e-4 · 40 ep)'] = {'ep': [int(r[0]) for r in rows], 'acc': [float(r[2]) for r in rows], 'f1': [float(r[3]) for r in rows]}
    t = (ROOT / 'logs/retrain_finetune.log').read_text(encoding='utf-8', errors='ignore')
    for name in ('MobileNetV3', 'EfficientNet-B0', 'ViT-Tiny'):
        seg = t.split(f'모델: {name}')[1] if f'모델: {name}' in t else ''
        seg = seg.split('모델: ')[0]
        p1 = re.findall(r'P1 Ep(\d+) \|.*?VlF1=([\d.]+)', seg); p2 = re.findall(r'P2 Ep(\d+) \|.*?VlF1=([\d.]+)', seg)
        ep = list(range(1, len(p1) + len(p2) + 1)); f1 = [float(x[1]) for x in p1] + [float(x[1]) for x in p2]
        out[f'{name} (2-Phase · 백본 lr 1e-5)'] = {'ep': ep, 'f1': f1, 'p1_end': len(p1)}
    t = (ROOT / 'logs/retrain_advanced.log').read_text(encoding='utf-8', errors='ignore')
    rows = re.findall(r'^\s*(\d+) \|\s*([\d.]+) \|\s*([\d.]+) \|', t, re.M)
    out['AdvancedDefectPredictor (Focal · 50 ep)'] = {'ep': [int(r[0]) for r in rows], 'f1': [float(r[2]) for r in rows]}
    return out


def fig6():
    L = parse_legacy()
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, d) in enumerate(L.items()):
        ax.plot(d['ep'], d['f1'], color=C[i], lw=2, solid_capstyle='round', label=name)
        ax.scatter([d['ep'][-1]], [d['f1'][-1]], s=40, color=C[i], edgecolor=SURFACE, lw=2, zorder=3)
        ax.text(d['ep'][-1] + 0.5, d['f1'][-1], f"{d['f1'][-1]:.3f}", va='center', fontsize=8, color=TEXT2)
        if 'p1_end' in d:
            ax.axvline(d['p1_end'] + 0.5, color=GRID, lw=1)
    if 'WaferCNN (Adam 3e-4 · 40 ep)' in L:
        d = L['WaferCNN (Adam 3e-4 · 40 ep)']; ax.plot(d['ep'], d['acc'], color=C[0], lw=1.2, ls='--', label='WaferCNN val accuracy (참고)')
    ax.set_xlabel('epoch (사전학습 모델: Phase 1 5 ep 포함, 세로선 = Phase 2 시작)'); ax.set_ylabel('validation macro F1')
    ax.set_ylim(0.2, 1.0); ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.set_title('기존 레시피 재학습 곡선 — 사전학습 모델은 백본 lr 1e-5 탓에 낮은 곳에서 느리게 오른다', loc='left', fontsize=12, fontweight='bold')
    fig.tight_layout(); fig.savefig(OUT / 'fig6_legacy_curves.png'); plt.close(fig)


if __name__ == '__main__':
    runs = load_runs()
    fig1(runs); fig2(runs); fig3(runs); fig4(runs); fig5(runs); fig6()
    print('저장:', *sorted(p.name for p in OUT.glob('fig*.png')), sep='\n  ')
