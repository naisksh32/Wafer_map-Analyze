"""전체 재학습 마스터 스크립트: WaferCNN → 파인튜닝 → Advanced"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
scripts = [
    ('WaferCNN 재학습', ROOT / 'scripts/retrain_baseline.py'),
    ('파인튜닝 모델 재학습', ROOT / 'scripts/retrain_finetune.py'),
    ('Advanced 재학습', ROOT / 'scripts/retrain_advanced.py'),
]

venv_python = ROOT / '.venv/Scripts/python.exe'
log_dir = ROOT / 'logs'
log_dir.mkdir(exist_ok=True)

for name, script in scripts:
    print(f'\n{"="*60}')
    print(f'[START] {name}')
    print(f'{"="*60}', flush=True)
    log_file = log_dir / (script.stem + '.log')
    t0 = time.time()
    env = {**__import__('os').environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8', 'NO_ALBUMENTATIONS_UPDATE': '1'}
    with open(log_file, 'w', encoding='utf-8') as lf:
        result = subprocess.run(
            [str(venv_python), str(script)],
            cwd=str(ROOT),
            stdout=lf, stderr=subprocess.STDOUT,
            env=env,
        )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f'[DONE] {name}  ({elapsed/60:.1f}분)')
    else:
        print(f'[FAIL] {name}  returncode={result.returncode}')
        print(f'  → 로그 확인: {log_file}')
    # 로그 마지막 20줄 출력 (인코딩 안전 처리)
    with open(log_file, encoding='utf-8', errors='replace') as lf:
        lines = lf.readlines()
    for line in lines[-20:]:
        print(line.encode('ascii', errors='replace').decode('ascii'), end='')

print('\n모든 재학습 완료')

# ── Report.md 자동 업데이트
print(f'\n{"="*60}')
print('[START] Report.md 업데이트')
print(f'{"="*60}', flush=True)
update_script = ROOT / 'scripts/update_report.py'
result = subprocess.run(
    [str(venv_python), str(update_script)],
    cwd=str(ROOT),
    capture_output=True, text=True, encoding='utf-8', errors='replace'
)
print(result.stdout)
if result.returncode == 0:
    print('[DONE] Report.md 업데이트 완료')
else:
    print(f'[FAIL] Report.md 업데이트 실패\n{result.stderr}')
