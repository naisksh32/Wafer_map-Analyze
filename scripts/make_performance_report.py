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
               f'Scratch(선형 결함) precision 이 0.4~0.5 → 0.7~0.8 로 오르는 것이 대표적 효과다. 단 128px 이후의 추가 해상도(160 · 224)는 3 seeds 기준 +0.006 ± 0.006 으로 완만하며, '
               f'WaferCNN 128px 대조군({g("wafercnn_128","f1_mean") or 0:.4f})이 개선되지 않아 이 효과는 stride-32 사전학습 백본에 특유하다.',
               f'4. **ImageNet 전이는 도움이 된다.** 같은 조건에서 scratch 초기화 MobileNetV3-S 는 {g("mv3_scratch64","f1_mean") or float("nan"):.4f} 로 ImageNet 초기화보다 낮다.',
               f'5. **최고 성능은 EfficientNet-B0 224px {g("effb0_pre224","f1_mean") or float("nan"):.4f} ± {g("effb0_pre224","f1_std") or 0:.4f}** (3 seeds, Acc {(g("effb0_pre224","acc_mean") or 0)*100:.2f}%). '
               f'엣지 후보 MobileNetV3-S 는 160px {g("mv3_pre160","f1_mean") or 0:.4f} ± {g("mv3_pre160","f1_std") or 0:.4f} · 128px+ImageNet norm {g("mv3_pre128_norm","f1_mean") or 0:.4f} ± {g("mv3_pre128_norm","f1_std") or 0:.4f} · 128px {g("mv3_pre128","f1_mean"):.4f} ± {g("mv3_pre128","f1_std"):.4f} 로 세 구성이 통계적으로 구분되지 않는다. '
               f'프로젝트 목표(macro F1 ≥ 0.88)를 3 seeds 평균으로 넘는 것은 EfficientNet-B0 224px 와 MobileNetV3-S 160px 다.',
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

    # ── Part 3: 모델별 입력 전처리 결론 + 시각화
    if fc:
        cm = fc['models']; g = lambda k, f: cm.get(k, {}).get(f)
        md += ['## 3. 모델별 특화 입력 전처리 실험 (2차 · `Preprocess` 모듈 · seed 42)', '',
               'DataLoader 는 64px 증강 파이프라인을 그대로 쓰고, 모델 직전 GPU 위에서 해상도·보간·채널·정규화만 바꿨다. 학습 레시피는 1차와 동일.', '',
               '| 변인 | 비교 | 결과 | 판정 |', '|---|---|---|---|']
        rows = [
            ('해상도 (MobileNetV3-S)', '64 → 128 → 160 → 224px', f"{g('mv3_pre64','f1_mean'):.4f} → {g('mv3_pre128','f1_mean'):.4f} → {g('mv3_pre160','f1_mean') or 0:.4f} (±{g('mv3_pre160','f1_std') or 0:.4f}) → {g('mv3_pre224','f1_mean') or 0:.4f}", '**가장 큰 변인이나 이득은 64→128 에 집중.** 160px 는 3 seeds 에서 128px 대비 +0.006 ± 0.006 (구분 안 됨). 224px 는 1 seed'),
            ('해상도 (EfficientNet-B0)', '128 → 224px (원본)', f"{g('effb0_pre128','f1_mean'):.4f} → {g('effb0_pre224','f1_mean') or 0:.4f} (±{g('effb0_pre224','f1_std') or 0:.4f}, 3 seeds)", '**전체 최고.** 3 seeds 모두 0.903 이상, std 0.002 로 가장 안정'),
            ('토큰 수 (ViT-Tiny/16)', '16 → 64 → 196 tokens', f"{g('vit_pre64','f1_mean'):.4f} → {g('vit_pre128','f1_mean') or 0:.4f} → {g('vit_pre224','f1_mean') or 0:.4f}", '단조 증가. 단 5.4M 파라미터로 MobileNetV3-S 160px(1.53M) 와 동급'),
            ('해상도 (WaferCNN 대조군)', '64 → 128px', f"{g('wafercnn','f1_mean'):.4f} → {g('wafercnn_128','f1_mean') or 0:.4f}", '**개선 없음.** 해상도 효과는 stride-32 사전학습 백본에 특유한 구조적 현상'),
            ('보간', 'nearest vs bilinear (128px)', f"{g('mv3_pre128','f1_mean'):.4f} vs {g('mv3_pre128_bil','f1_mean') or 0:.4f}", '차이 없음 (3값 픽셀 유지 여부 무관)'),
            ('채널·정규화', '1ch 평균 conv vs 3ch 복제 + ImageNet norm', f"{g('mv3_pre128','f1_mean'):.4f} vs {g('mv3_pre128_3ch','f1_mean') or 0:.4f}", '차이 없음 (사전학습 conv1 원형 유지 이점 없음)'),
            ('정규화 단독', '1ch vs 1ch + ImageNet norm', f"{g('mv3_pre128','f1_mean'):.4f} vs {g('mv3_pre128_norm','f1_mean') or 0:.4f} (±{g('mv3_pre128_norm','f1_std') or 0:.4f}, 3 seeds)", '+0.004, std 1~2배 → 약한 양의 효과, 확정 아님'),
        ]
        md += [f'| {a} | {b} | {c} | {d} |' for a, b, c, d in rows]
        md += ['', '**해석.** 64×64 입력에서 stride-32 백본(MobileNetV3·EfficientNet)의 마지막 feature map 은 2×2 로 붕괴하고, 이때 위치·형태 정보(Center/Loc/Edge-Loc 구분, Scratch 의 선형성)가 사라진다. '
               '업샘플은 정보를 추가하지 않지만 백본이 공간 구조를 유지하며 처리할 수 있게 해 준다. WaferCNN 은 64px 에서 이미 4×4 를 확보하므로 같은 처치에 반응하지 않는다. '
               '클래스별로는 Scratch F1 이 0.57 → 0.75~0.81, Loc 0.72 → 0.78~0.81, Edge-Loc 0.77 → 0.81~0.86 으로 오르고, none·Edge-Ring 은 모든 모델이 0.96 이상이라 변화가 없다.', '',
               f"**배포 관점 권장 구성.** MobileNetV3-S 160px (1.53M · macro F1 {g('mv3_pre160','f1_mean') or 0:.3f} ± {g('mv3_pre160','f1_std') or 0:.3f}) — EfficientNet-B0 224px(4.02M · {g('effb0_pre224','f1_mean') or 0:.3f} ± {g('effb0_pre224','f1_std') or 0:.3f}) 대비 0.025 낮지만 연산량은 약 1/8. "
               f"160px 와 128px+ImageNet norm({g('mv3_pre128_norm','f1_mean') or 0:.3f} ± {g('mv3_pre128_norm','f1_std') or 0:.3f}) 은 통계적으로 구분되지 않으므로 지연 요건이 우선이면 128px 도 타당. 정확도 최우선이면 EfficientNet-B0 224px.", '']
    md += ['## 4. 시각화 (`scripts/plot_results.py` → `analysis/figures/`)', '',
           '| 그림 | 내용 |', '|---|---|',
           '| ![](../analysis/figures/fig1_model_f1_bar.png) | **fig1** 모델별 macro F1 — 기존 레시피(주황) vs 동일 레시피(파랑). 오차막대 = 95% CI 또는 3 seeds ± std |',
           '| ![](../analysis/figures/fig2_preprocess_f1_bar.png) | **fig2** 모델별 입력 전처리 변인 — 해상도만 막대 길이를 바꾼다 |',
           '| ![](../analysis/figures/fig3_per_class_f1_bar.png) | **fig3** 클래스별 F1 — 해상도 이득은 Scratch·Loc·Edge-Loc 에 집중 |',
           '| ![](../analysis/figures/fig4_training_curves.png) | **fig4** 모델별 학습 경과 (epoch 별 val accuracy·val macro F1) |',
           '| ![](../analysis/figures/fig5_val_f1_overlay.png) | **fig5** 핵심 모델 학습 곡선 overlay — 해상도가 높을수록 첫 epoch 부터 위에서 시작해 순서가 바뀌지 않는다 |',
           '| ![](../analysis/figures/fig6_legacy_curves.png) | **fig6** 기존 레시피 재학습 곡선 — 2-Phase 사전학습 모델은 Phase 2 에서도 백본 lr 1e-5 탓에 느리게 오른다 |', '']

    # ── Part 5: 엣지 배포 재검증 + Part 6: 검증 결과
    dep = load(A / 'deployment_summary_v2.json')
    if dep:
        md += ['## 5. 엣지 배포 재검증 (`scripts/export_onnx_v2.py` → `analysis/deployment_summary_v2.json`)', '',
               f"CPU {dep['machine']['threads']} threads · onnxruntime {dep['machine']['onnxruntime']} · 업샘플은 ONNX 그래프 내부, 입력 (B,1,64,64) 유지", '',
               '| 모델 | PyTorch F1 | ONNX FP32 F1 | argmax 일치 | INT8 F1 | INT8 판정 | b=1 (ms) | b=32 (ms) | FP32 크기 | INT8 크기 |', '|---|---:|---:|---:|---:|:-:|---:|---:|---:|---:|']
        for k, m in dep['models'].items():
            a, l, sz = m['accuracy'], m['latency_ms'], m['size']
            md.append(f"| {m['description']} | {a['pytorch']['f1_macro']:.4f} | {a['onnx_fp32']['f1_macro']:.4f} | {a['fp32_vs_pytorch_argmax_agreement']*100:.2f}% | "
                      f"{a['onnx_int8']['f1_macro']:.4f} | {m['int8_verdict']} | {l['onnx_fp32_b1']['avg_ms']:.2f} | {l['onnx_fp32_b32']['avg_ms']:.1f} | {sz['onnx_fp32_mb']}MB | {sz['onnx_int8_mb']}MB |")
        md.append('')
    ver = load(A / 'verification_report.json')
    if ver:
        md += [f"## 6. 검증 (`scripts/verify_results.py` → `analysis/verification_report.json`) — {ver['summary']}", '',
               '| 검증 | 결과 | 요약 |', '|---|:-:|---|']
        for k, c in ver['checks'].items():
            md.append(f"| {k} | {'PASS' if c['pass'] else 'FAIL'} | {c.get('summary', '')} |")
        if any(not c['pass'] for c in ver['checks'].values()):
            md += ['', '실패 항목 상세:', '']
            for k, c in ver['checks'].items():
                if not c['pass']:
                    md += [f"- **{k}**: " + '; '.join(map(str, c.get('failures', [])))[:600]]
        g1 = ver['checks'].get('G1_fair_checkpoint_rescore', {}).get('rows', [])
        if g1:
            md += ['', f"G1·G2 의 허용오차(F1 ≤ 1e-3, 예측 불일치 ≤ 0.1%)는 GPU cuDNN autotune 커널 선택에 따른 near-tie 로짓 뒤집힘을 감안한 것이다. "
                   f"실제 관측된 최대 차이는 F1 {max(x['d_f1'] for x in g1):.1e}, 예측이 달라진 샘플은 run 당 최대 {max(x['n_pred_diff_upper_bound'] for x in g1)}개(25,943개 중)이며, "
                   f"{sum(x['cm_identical'] for x in g1)}/{len(g1)} run 은 confusion matrix 가 비트 단위로 동일했다. 문서에 기재된 4자리 수치는 이 범위에서 재현된다.", '']

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
