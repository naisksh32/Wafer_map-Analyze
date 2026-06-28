"""재학습 결과로 Report.md 업데이트 스크립트

재학습 완료 후 실행:
  .venv/Scripts/python scripts/update_report.py
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = ROOT / 'analysis'

def load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'  [WARN] {path.name} 로드 실패: {e}')
        return None


def main():
    baseline  = load_json(ANALYSIS_DIR / 'baseline_results.json')
    finetune  = load_json(ANALYSIS_DIR / 'finetuning_results.json')
    advanced  = load_json(ANALYSIS_DIR / 'advanced_model_results.json')

    if not (baseline and finetune and advanced):
        print('일부 결과 파일이 없습니다. 재학습 완료 후 다시 실행하세요.')
        return

    mv3  = finetune['models'].get('MobileNetV3', {})
    eff  = finetune['models'].get('EfficientNet-B0', {})
    vit  = finetune['models'].get('ViT-Tiny', {})

    print('=== 재학습 결과 요약 ===')
    print(f'WaferCNN:')
    print(f'  Test Acc  : {baseline["test_accuracy"]*100:.2f}%')
    print(f'  Test F1   : {baseline["test_f1_macro"]:.4f}  (목표 ≥ 0.80)  {"✅" if baseline["test_f1_macro"]>=0.80 else "❌"}')
    print(f'  Best Val  : {baseline["best_val_f1"]:.4f}')
    print()
    print(f'파인튜닝:')
    for name, res in [('MobileNetV3', mv3), ('EfficientNet-B0', eff), ('ViT-Tiny', vit)]:
        if res:
            print(f'  {name:15s}: Acc={res["test_accuracy"]*100:.2f}%  F1={res["test_f1_macro"]:.4f}')
    print()
    print(f'AdvancedDefectPredictor:')
    print(f'  Test F1   : {advanced["test_defect_f1"]:.4f}  (목표 ≥ 0.90)  {"✅" if advanced["test_defect_f1"]>=0.90 else "❌"}')
    print(f'  Best Val  : {advanced["best_val_defect_f1"]:.4f}')

    # Report.md 업데이트
    report_path = ROOT / 'Report.md'
    with open(report_path, encoding='utf-8') as f:
        report = f.read()

    # 4.1 베이스라인 섹션 업데이트
    old_baseline_table = (
        "| 최적 epoch | 1 |\n"
        "| **Test Accuracy** | **10.65%** |\n"
        "| **Test F1-macro** | **0.5014** |\n"
        "| 목표 달성 (F1≥0.80) | ❌ |"
    )
    new_baseline_table = (
        f"| 최적 epoch | {baseline['best_epoch']} |\n"
        f"| **Test Accuracy** | **{baseline['test_accuracy']*100:.2f}%** |\n"
        f"| **Test F1-macro** | **{baseline['test_f1_macro']:.4f}** |\n"
        f"| 목표 달성 (F1≥0.80) | {'✅' if baseline['test_f1_macro']>=0.80 else '❌'} |\n"
        f"| 버그 수정 | WeightedRandomSampler+class_weight 이중 보정 제거, lr 3e-4 |"
    )
    if old_baseline_table in report:
        report = report.replace(old_baseline_table, new_baseline_table)
        print('  [✓] 베이스라인 섹션 업데이트')
    else:
        print('  [!] 베이스라인 섹션 패턴 불일치 - 수동 확인 필요')

    # 4.2 파인튜닝 테이블 업데이트
    old_ft_table = (
        "| WaferCNN (베이스라인) | 0.5024 | 10.65% | 0.5014 | — |\n"
        "| MobileNetV3 Small | 0.5736 | 37.19% | 0.5618 | 5.95 MB |\n"
        "| ViT-Tiny | 0.6578 | 75.57% | 0.6473 | — |\n"
        "| **EfficientNet-B0** | **0.6775** | **79.37%** | **0.6673** | — |"
    )
    best_ft_name = finetune.get('best_model', 'EfficientNet-B0')
    new_ft_table = (
        f"| WaferCNN (베이스라인) | {baseline['best_val_f1']:.4f} | {baseline['test_accuracy']*100:.2f}% | {baseline['test_f1_macro']:.4f} | — |\n"
        f"| MobileNetV3 Small | {mv3.get('best_val_f1',0):.4f} | {mv3.get('test_accuracy',0)*100:.2f}% | {mv3.get('test_f1_macro',0):.4f} | 5.95 MB |\n"
        f"| ViT-Tiny | {vit.get('best_val_f1',0):.4f} | {vit.get('test_accuracy',0)*100:.2f}% | {vit.get('test_f1_macro',0):.4f} | — |\n"
        f"| **EfficientNet-B0** | **{eff.get('best_val_f1',0):.4f}** | **{eff.get('test_accuracy',0)*100:.2f}%** | **{eff.get('test_f1_macro',0):.4f}** | — |"
    )
    if old_ft_table in report:
        report = report.replace(old_ft_table, new_ft_table)
        print('  [✓] 파인튜닝 테이블 업데이트')
    else:
        print('  [!] 파인튜닝 테이블 패턴 불일치 - 수동 확인 필요')

    # 11.1 종합 성과 테이블 업데이트
    old_summary_table = (
        "| Step 4 | WaferCNN (베이스라인) | 0.5014 | 커스텀 CNN |\n"
        "| Step 5 | MobileNetV3 | 0.5618 | 파인튜닝 |\n"
        "| Step 5 | ViT-Tiny | 0.6473 | 파인튜닝 |\n"
        "| Step 5 | EfficientNet-B0 | 0.6673 | **최고 분류 성능** |\n"
        "| Step 6 | WaferCNN (HPO) | 0.5987 | Optuna 최적화 |"
    )
    new_summary_table = (
        f"| Step 4 | WaferCNN (베이스라인, 수정) | {baseline['test_f1_macro']:.4f} | 이중 보정 버그 수정 |\n"
        f"| Step 5 | MobileNetV3 (수정) | {mv3.get('test_f1_macro',0):.4f} | 파인튜닝 |\n"
        f"| Step 5 | ViT-Tiny (수정) | {vit.get('test_f1_macro',0):.4f} | 파인튜닝 |\n"
        f"| Step 5 | EfficientNet-B0 (수정) | {eff.get('test_f1_macro',0):.4f} | **최고 분류 성능** |\n"
        f"| Step 11 | AdvancedDefectPredictor | {advanced['test_defect_f1']:.4f} | Focal Loss + CosineWarmRestart |"
    )
    if old_summary_table in report:
        report = report.replace(old_summary_table, new_summary_table)
        print('  [✓] 종합 성과 테이블 업데이트')
    else:
        print('  [!] 종합 성과 테이블 패턴 불일치 - 수동 확인 필요')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nReport.md 업데이트 완료: {report_path}')


if __name__ == '__main__':
    main()
