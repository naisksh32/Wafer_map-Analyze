"""WaferCNN 완료 후 나머지 2개 스크립트 실행: 파인튜닝 + Advanced"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
scripts = [
    ('파인튜닝 모델 재학습', ROOT / 'scripts/retrain_finetune.py'),
    ('Advanced 재학습',     ROOT / 'scripts/retrain_advanced.py'),
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
        print(f'[DONE] {name}  ({elapsed/60:.1f}min)', flush=True)
    else:
        print(f'[FAIL] {name}  returncode={result.returncode}')
    with open(log_file, encoding='utf-8', errors='replace') as lf:
        lines = lf.readlines()
    for line in lines[-25:]:
        sys.stdout.write(line.encode('ascii', errors='replace').decode('ascii'))
    sys.stdout.flush()

print('\n=== All remaining training done ===', flush=True)

# Report.md 업데이트
print(f'\n[START] Report.md update', flush=True)
update_script = ROOT / 'scripts/update_report.py'
result = subprocess.run(
    [str(venv_python), str(update_script)],
    cwd=str(ROOT),
    capture_output=True,
)
out = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
err = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
print(out.encode('ascii', errors='replace').decode('ascii'))
if result.returncode == 0:
    print('[DONE] Report.md updated', flush=True)
else:
    print(f'[FAIL] {err[:200]}', flush=True)
