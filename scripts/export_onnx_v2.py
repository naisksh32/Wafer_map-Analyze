"""엣지 배포 재검증 v2 — 동일 레시피 실험의 MobileNetV3-S 체크포인트를 ONNX 로 변환·검증·벤치마크.

v1(`export_onnx.py`) 과의 차이
  - 대상: `checkpoints/fair/*.pth` (Preprocess 래퍼 포함 → 업샘플이 ONNX 그래프 안에 들어가고 입력은 그대로 (B,1,64,64))
  - 비교군: 기존 배포 모델(64px, 2-Phase 레시피) 을 같은 절차로 변환해 같은 머신에서 상대 속도 측정
  - INT8 dynamic quantization 은 정확도 검증 후 채택/기각

사용:
  python scripts/export_onnx_v2.py                       # 기본: mv3_pre160_seed42 + 비교군 legacy 64px
  python scripts/export_onnx_v2.py --targets mv3_pre160_seed42 mv3_pre128_seed42
산출:
  checkpoints/deploy/<name>.onnx, <name>_int8.onnx
  analysis/deployment_summary_v2.json
"""
import os, sys, json, time, pickle, argparse, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from src.data_loader import CLASS_ORDER  # noqa: E402
import fair_compare as fc  # noqa: E402  (MODEL_ZOO · Preprocess · _mv3)

NUM_CLASSES = len(CLASS_ORDER)
DEPLOY = ROOT / 'checkpoints/deploy'
DEPLOY.mkdir(exist_ok=True)


def load_test():
    maps = np.load(ROOT / 'data/processed/all_maps_resized.npy', mmap_mode='r')
    with open(ROOT / 'data/processed/split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
    idx = split['test_idx']
    x = np.clip(np.asarray(maps[idx]).astype(np.float32) / 2.0, 0, 1)[:, None]
    y = split['encoded_labels'][idx].astype(int)
    return x, y


def build_fair(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    key = ck['model_key'].split('@')[0]
    desc, builder = fc.MODEL_ZOO[key]
    m = builder(); m.load_state_dict(ck['model_state']); m.eval()
    return m, dict(model_key=ck['model_key'], seed=ck['seed'], epoch=ck['epoch'], description=desc,
                   preprocess=getattr(m, 'spec', dict(size=64, mode=None, channels=1, imagenet_norm=False)))


def build_legacy(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    m = fc._mv3(pretrained=False, in_ch=1); m.load_state_dict(ck['model_state']); m.eval()
    return m, dict(model_key='legacy_mv3_64', seed=42, epoch=ck.get('epoch'),
                   description='MobileNetV3-S (기존 2-Phase 레시피, 64px) — 이전 배포 모델',
                   preprocess=dict(size=64, mode=None, channels=1, imagenet_norm=False))


@torch.no_grad()
def torch_predict(m, x, bs=256):
    out = []
    for i in range(0, len(x), bs):
        out.append(m(torch.from_numpy(x[i:i + bs])).argmax(1).numpy())
    return np.concatenate(out)


def onnx_predict(sess, x, bs=256):
    name = sess.get_inputs()[0].name; out = []
    for i in range(0, len(x), bs):
        out.append(sess.run(None, {name: x[i:i + bs]})[0].argmax(1))
    return np.concatenate(out)


def bench(fn, n=200, warmup=20):
    for _ in range(warmup): fn()
    t = []
    for _ in range(n):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    t = np.array(t); return {'avg_ms': round(float(t.mean()), 3), 'p50_ms': round(float(np.median(t)), 3), 'p95_ms': round(float(np.percentile(t, 95)), 3)}


def metrics(y, p):
    return {'accuracy': round(float(accuracy_score(y, p)), 4), 'f1_macro': round(float(f1_score(y, p, average='macro', zero_division=0)), 4)}


def export_one(name, model, meta, x, y, threads):
    import onnx, onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    fp32 = DEPLOY / f'{name}.onnx'; int8 = DEPLOY / f'{name}_int8.onnx'
    dummy = torch.zeros(1, 1, 64, 64)
    torch.onnx.export(model, dummy, str(fp32), input_names=['input'], output_names=['logits'], opset_version=17,
                      dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}, dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp32)))
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)

    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    s32 = ort.InferenceSession(str(fp32), so, providers=['CPUExecutionProvider'])
    s8 = ort.InferenceSession(str(int8), so, providers=['CPUExecutionProvider'])
    torch.set_num_threads(threads)

    p_t = torch_predict(model, x); p_32 = onnx_predict(s32, x); p_8 = onnx_predict(s8, x)
    acc = {'pytorch': metrics(y, p_t), 'onnx_fp32': metrics(y, p_32), 'onnx_int8': metrics(y, p_8),
           'fp32_vs_pytorch_argmax_agreement': round(float((p_t == p_32).mean()), 5),
           'int8_vs_pytorch_argmax_agreement': round(float((p_t == p_8).mean()), 5)}
    acc['fp32_f1_delta'] = round(acc['onnx_fp32']['f1_macro'] - acc['pytorch']['f1_macro'], 4)
    acc['int8_f1_delta'] = round(acc['onnx_int8']['f1_macro'] - acc['pytorch']['f1_macro'], 4)
    int8_verdict = 'ADOPT' if acc['int8_f1_delta'] >= -0.01 else 'REJECT'

    x1, x32 = x[:1].copy(), x[:32].copy(); t1, t32 = torch.from_numpy(x1), torch.from_numpy(x32)
    lat = {}
    with torch.no_grad():
        lat['pytorch_cpu_b1'] = bench(lambda: model(t1)); lat['pytorch_cpu_b32'] = bench(lambda: model(t32), n=100)
    lat['onnx_fp32_b1'] = bench(lambda: s32.run(None, {'input': x1})); lat['onnx_fp32_b32'] = bench(lambda: s32.run(None, {'input': x32}), n=100)
    lat['onnx_int8_b1'] = bench(lambda: s8.run(None, {'input': x1})); lat['onnx_int8_b32'] = bench(lambda: s8.run(None, {'input': x32}), n=100)
    size = {'onnx_fp32_mb': round(fp32.stat().st_size / 1048576, 2), 'onnx_int8_mb': round(int8.stat().st_size / 1048576, 2)}
    size['int8_reduction_pct'] = round(100 * (1 - size['onnx_int8_mb'] / size['onnx_fp32_mb']), 1)
    return {**meta, 'files': {'onnx_fp32': fp32.relative_to(ROOT).as_posix(), 'onnx_int8': int8.relative_to(ROOT).as_posix()},
            'accuracy': acc, 'int8_verdict': int8_verdict, 'latency_ms': lat, 'size': size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['mv3_pre160_seed42'], help='checkpoints/fair/<name>.pth')
    ap.add_argument('--legacy', default='checkpoints/MobileNetV3_34_0.7431.pth')
    ap.add_argument('--threads', type=int, default=4, help='CPU 스레드 (엣지 디바이스 근사: 4)')
    args = ap.parse_args()
    x, y = load_test(); print(f'test {len(y):,} · threads {args.threads}')
    results = {}
    if args.legacy and (ROOT / args.legacy).exists():
        m, meta = build_legacy(ROOT / args.legacy); meta['checkpoint'] = args.legacy
        results['legacy_mv3_64'] = export_one('legacy_mv3_64', m, meta, x, y, args.threads)
        r = results['legacy_mv3_64']; print(f"legacy 64px    F1 torch {r['accuracy']['pytorch']['f1_macro']} onnx {r['accuracy']['onnx_fp32']['f1_macro']} int8 {r['accuracy']['onnx_int8']['f1_macro']}  b1 {r['latency_ms']['onnx_fp32_b1']['avg_ms']}ms")
    for t in args.targets:
        p = ROOT / 'checkpoints/fair' / f'{t}.pth'
        m, meta = build_fair(p); meta['checkpoint'] = p.relative_to(ROOT).as_posix()
        results[t] = export_one(t, m, meta, x, y, args.threads)
        r = results[t]; print(f"{t:<14} F1 torch {r['accuracy']['pytorch']['f1_macro']} onnx {r['accuracy']['onnx_fp32']['f1_macro']} int8 {r['accuracy']['onnx_int8']['f1_macro']} [{r['int8_verdict']}]  b1 {r['latency_ms']['onnx_fp32_b1']['avg_ms']}ms  b32 {r['latency_ms']['onnx_fp32_b32']['avg_ms']}ms")
    base = results.get('legacy_mv3_64')
    for k, r in results.items():
        if base and k != 'legacy_mv3_64':
            r['relative_to_legacy'] = {'f1_gain': round(r['accuracy']['onnx_fp32']['f1_macro'] - base['accuracy']['onnx_fp32']['f1_macro'], 4),
                                       'latency_b1_ratio': round(r['latency_ms']['onnx_fp32_b1']['avg_ms'] / base['latency_ms']['onnx_fp32_b1']['avg_ms'], 2),
                                       'latency_b32_ratio': round(r['latency_ms']['onnx_fp32_b32']['avg_ms'] / base['latency_ms']['onnx_fp32_b32']['avg_ms'], 2)}
    import platform, onnxruntime
    out = {'description': 'ONNX 배포 재검증 v2 — 동일 레시피 실험 체크포인트. 업샘플은 그래프 내부, 입력 (B,1,64,64) 유지',
           'machine': {'cpu': platform.processor(), 'threads': args.threads, 'onnxruntime': onnxruntime.__version__, 'torch': torch.__version__},
           'test_set_size': int(len(y)), 'models': results}
    (ROOT / 'analysis/deployment_summary_v2.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('저장: analysis/deployment_summary_v2.json')


if __name__ == '__main__':
    main()
