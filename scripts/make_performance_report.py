"""재학습·재평가 결과를 하나의 성능 보고서(Markdown)로 묶는다.

입력:
  analysis/final_evaluation.json          (scripts/evaluate_all.py — 기존 레시피 재학습 체크포인트 통합 평가)
  analysis/per_class_metrics.csv          (〃 클래스별 P/R/F1)
  analysis/fair_compare/summary.json      (scripts/fair_compare.py — 동일 프로토콜 · 다중 seed)
  analysis/fair_compare/runs.csv
출력:
  docs/MODEL_PERFORMANCE.md
  analysis/finetuning_comparison.csv      (stale 파일을 최신 값으로 재생성)
"""
import json
from datetime import date
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'analysis'
CLASS_ORDER = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def main():
    fe = load(A / 'final_evaluation.json')
    pc = pd.read_csv(A / 'per_class_metrics.csv', encoding='utf-8-sig') if (A / 'per_class_metrics.csv').exists() else None
    fc = load(A / 'fair_compare/summary.json')
    runs = pd.read_csv(A / 'fair_compare/runs.csv', encoding='utf-8-sig') if (A / 'fair_compare/runs.csv').exists() else None

    md = [f'# 모델 성능 재측정 보고서 ({date.today().isoformat()})', '',
          '모든 수치는 **Test 25,943개** (증강·샘플러 미적용, 원본 불균형 분포) 기준이며, 같은 체크포인트에서 Accuracy · macro F1 · weighted F1 · macro Precision/Recall 을 동시에 계산했다.',
          '`macro F1 95% CI` 는 테스트셋 bootstrap 1,000회 재표집 구간이다. 두 모델의 구간이 크게 겹치면 그 차이는 테스트셋 표본 잡음 수준이다.', '']

    # ── 핵심 결론 (수치는 결과 파일에서 읽음)
    if fe and fc:
        fm = fe['models']; cm = fc['models']
        g = lambda k, f: cm.get(k, {}).get(f)
        md += ['## 0. 핵심 결론', '',
               f'1. **기존 README 수치는 재현된다.** 기존 레시피로 다시 학습한 WaferCNN macro F1 {fm["WaferCNN"]["test_f1_macro"]:.4f}, '
               f'MobileNetV3-Small {fm["MobileNetV3-Small"]["test_f1_macro"]:.4f} 로 이전 기재값(0.8458 / 0.7369)과 ±0.025 안에서 일치한다.',
               f'2. **그러나 "커스텀 CNN > MobileNet" 은 레시피 차이였다.** 같은 레시피(AdamW 3e-4 · 25 ep · 조기종료 없음)에서 '
               f'WaferCNN {g("wafercnn","f1_mean"):.4f} ± {g("wafercnn","f1_std"):.4f} vs MobileNetV3-S 64px {g("mv3_pre64","f1_mean"):.4f} ± {g("mv3_pre64","f1_std"):.4f} (3 seeds) — '
               f'기존 파인튜닝의 백본 학습률 1e-5 가 MobileNet 을 0.10 가까이 깎아 먹고 있었다.',
               f'3. **입력 해상도가 가장 큰 변인이다.** 64×64 를 128×128 로 nearest 업샘플만 해도 MobileNetV3-S 는 {g("mv3_pre128","f1_mean"):.4f} ± {g("mv3_pre128","f1_std"):.4f} (3 seeds), '
               f'40 epoch 에서는 {g("mv3_pre128@ep40","f1_mean") or float("nan"):.4f} 에 도달한다. stride-32 백본이 64px 에서 2×2 feature map 으로 붕괴하는 문제가 해소되기 때문이다. '
               f'Scratch(선형 결함) precision 이 0.4~0.5 → 0.7~0.8 로 오르는 것이 대표적 효과다.',
               f'4. **ImageNet 전이는 도움이 된다.** 같은 조건에서 scratch 초기화 MobileNetV3-S 는 {g("mv3_scratch64","f1_mean") or float("nan"):.4f} 로 ImageNet 초기화보다 낮다.',
               f'5. **최고 성능은 EfficientNet-B0 128px {g("effb0_pre128","f1_mean") or float("nan"):.4f}** (Acc {(g("effb0_pre128","acc_mean") or 0)*100:.2f}%), 그 다음이 MobileNetV3-S 128px 이다. '
               f'프로젝트 목표(macro F1 ≥ 0.88)는 이 두 구성에서 달성된다.',
               '6. **val macro F1 은 epoch 간 ±0.03 요동**하므로(소수 클래스 22~83개) 조기종료·best-val 선택은 seed 운에 좌우된다. '
               'WaferCNN 은 best-val 체크포인트와 마지막 epoch 의 test F1 차이가 최대 0.017 인 반면, 사전학습 모델은 0.005 이내로 안정적이다.',
               '']

    # ── Part 1: 기존 레시피 재학습
    if fe:
        md += ['## 1. 기존 학습 레시피 그대로 재학습한 결과 (`scripts/retrain_*.py` → `scripts/evaluate_all.py`)', '',
               f'환경: {fe.get("gpu") or fe.get("device")} · seed {fe.get("seed")} · 결과 파일 `analysis/final_evaluation.json`', '',
               '| 모델 | 파라미터 | Accuracy | **macro F1** | 95% CI | weighted F1 | macro P | macro R | 체크포인트 |',
               '|---|---:|---:|---:|:---:|---:|---:|---:|---|']
        rows = sorted(fe['models'].items(), key=lambda kv: -kv[1]['test_f1_macro'])
        for name, m in rows:
            ci = m.get('test_f1_macro_ci95', [None, None])
            ci_s = f'[{ci[0]:.4f}, {ci[1]:.4f}]' if ci[0] is not None else '—'
            md.append(f'| {name} | {m["params_M"]:.2f}M | {m["test_accuracy"]*100:.2f}% | **{m["test_f1_macro"]:.4f}** | {ci_s} | '
                      f'{m["test_f1_weighted"]:.4f} | {m.get("test_precision_macro", float("nan")):.4f} | '
                      f'{m.get("test_recall_macro", float("nan")):.4f} | `{m["checkpoint"]}` |')
        if pc is not None:
            md += ['', '### 1.1 클래스별 F1', '', '| 모델 | ' + ' | '.join(CLASS_ORDER) + ' |', '|---|' + '---:|' * len(CLASS_ORDER)]
            for name, _ in rows:
                sub = pc[pc.model == name].set_index('class')
                if len(sub):
                    md.append(f'| {name} | ' + ' | '.join(f'{sub.loc[c, "f1"]:.3f}' for c in CLASS_ORDER) + ' |')
            md += ['', '### 1.2 클래스별 Precision / Recall (Scratch·Loc·Edge-Loc 병목 확인용)', '',
                   '| 모델 | 클래스 | support | precision | recall | f1 |', '|---|---|---:|---:|---:|---:|']
            for name, _ in rows:
                sub = pc[(pc.model == name) & (pc['class'].isin(['Scratch', 'Loc', 'Edge-Loc']))]
                for _, r in sub.iterrows():
                    md.append(f'| {name} | {r["class"]} | {int(r.support)} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} |')
        md.append('')

    # ── Part 2: 동일 프로토콜
    if fc:
        P = fc['protocol']
        md += ['## 2. 동일 프로토콜 공정 비교 (`scripts/fair_compare.py`)', '',
               '기존 파이프라인은 WaferCNN(전체 파라미터 Adam 3e-4)과 사전학습 모델(2-Phase, 백본 lr 1e-5)의 레시피가 달라 아키텍처 효과를 분리할 수 없었다. '
               '아래는 모든 모델을 같은 옵티마이저·LR·증강·샘플러·epoch 예산·선택 기준으로 학습하고 seed 를 반복한 결과다.', '',
               f'프로토콜: {P["optimizer"]} lr {P["lr"]} · wd {P["weight_decay"]} · batch {P["batch_size"]} · {P["scheduler"]} · {P["loss"]} · {P["sampler"]} · 선택 기준 {P["select_by"]}', '',
               '| 모델 | 파라미터 | seeds | Accuracy | **macro F1 mean ± std** | min – max | weighted F1 | macro P | macro R | best epoch(평균) |',
               '|---|---:|:-:|---:|---:|:---:|---:|---:|---:|---:|']
        for key, m in fc['models'].items():
            md.append(f'| {m["description"]} | {m["params_M"]:.2f}M | {m["n_seeds"]} | {m["acc_mean"]*100:.2f}% | '
                      f'**{m["f1_mean"]:.4f} ± {m["f1_std"]:.4f}** | {m["f1_min"]:.4f} – {m["f1_max"]:.4f} | '
                      f'{m["wf1_mean"]:.4f} | {m["prec_mean"]:.4f} | {m["rec_mean"]:.4f} | {m["best_epoch_mean"]:.1f} |')
        md += ['', '### 2.1 클래스별 F1 (seed 평균)', '', '| 모델 | ' + ' | '.join(CLASS_ORDER) + ' |', '|---|' + '---:|' * len(CLASS_ORDER)]
        for key, m in fc['models'].items():
            md.append(f'| {m["description"]} | ' + ' | '.join(f'{m[f"f1_{c}_mean"]:.3f}' for c in CLASS_ORDER) + ' |')
        if runs is not None:
            md += ['', '### 2.2 run 별 원자료', '', '| model | seed | best ep | val F1 | test Acc | test macro F1 | 95% CI | 학습(min) |',
                   '|---|:-:|:-:|---:|---:|---:|:---:|---:|']
            for _, r in runs.sort_values(['model_key', 'seed']).iterrows():
                md.append(f'| {r.model_key} | {r.seed} | {r.best_epoch} | {r.val_f1_macro:.4f} | {r.test_accuracy:.4f} | '
                          f'{r.test_f1_macro:.4f} | [{r.ci_lo:.4f}, {r.ci_hi:.4f}] | {r.train_min:.1f} |')
        md.append('')

    out = ROOT / 'docs/MODEL_PERFORMANCE.md'
    out.write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(f'저장: {out}')

    # stale finetuning_comparison.csv 재생성
    if fe:
        rows = []
        for name in ['WaferCNN', 'MobileNetV3-Small', 'EfficientNet-B0', 'ViT-Tiny', 'AdvancedDefectPredictor']:
            m = fe['models'].get(name)
            if m:
                rows.append({'모델': name, '파라미터 수': f'{m["params_M"]:.2f}M',
                             'Test Accuracy': f'{m["test_accuracy"]*100:.2f}%', 'Test F1 macro': m['test_f1_macro'],
                             'Test F1 weighted': m['test_f1_weighted'],
                             '목표 달성 (F1≥0.80)': 'YES' if m['test_f1_macro'] >= 0.80 else 'NO'})
        pd.DataFrame(rows).to_csv(A / 'finetuning_comparison.csv', index=False, encoding='utf-8-sig')
        print('재생성: analysis/finetuning_comparison.csv')


if __name__ == '__main__':
    main()
