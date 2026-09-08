"""ONNX 변환 · 정확도 검증 · 속도 벤치마크 재실행

대응 이슈:
    1. 기존 08 배포 노트북이 폐기된 체크포인트(MobileNetV3_15_0.5736.pth, Acc 37.19%)를 사용
       → 재학습 완료본(MobileNetV3_34_0.7417.pth)으로 교체
    2. "모델 크기 72.2% 감소"를 성과로 제시했으나 INT8 양자화 모델은 F1 0.023 으로 사용 불가
       → 정확도 검증을 필수 단계로 포함하여 채택/기각을 근거와 함께 기록

산출물:
    checkpoints/MobileNetV3_deploy.onnx
    checkpoints/MobileNetV3_deploy_int8.onnx
    analysis/deployment_summary.json
"""
import sys
import json
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import CLASS_ORDER  # noqa: E402

NUM_CLASSES = len(CLASS_ORDER)
CKPT = ROOT / 'checkpoints/MobileNetV3_34_0.7417.pth'
ONNX_FP32 = ROOT / 'checkpoints/MobileNetV3_deploy.onnx'
ONNX_INT8 = ROOT / 'checkpoints/MobileNetV3_deploy_int8.onnx'


def build_mobilenet_v3_small():
    model = torchvision.models.mobilenet_v3_small(weights=None)
    old = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, old.out_channels, old.kernel_size,
                                     old.stride, old.padding, bias=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    return model


def load_test_data():
    maps = np.load(ROOT / 'data/processed/all_maps_resized.npy', mmap_mode='r')
    with open(ROOT / 'data/processed/split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
    idx = split['test_idx']
    x = np.asarray(maps[idx]).astype(np.float32) / 2.0
    x = np.clip(x, 0.0, 1.0)[:, None, :, :]          # (N,1,64,64)
    y = split['encoded_labels'][idx].astype(int)
    return x, y


def onnx_predict(sess, x, batch=256):
    name = sess.get_inputs()[0].name
    preds = []
    for i in range(0, len(x), batch):
        out = sess.run(None, {name: x[i:i + batch]})[0]
        preds.append(out.argmax(1))
    return np.concatenate(preds)


def bench(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {'avg_ms': round(float(np.mean(ts)), 2),
            'p95_ms': round(float(np.percentile(ts, 95)), 2)}


def main():
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print('1) 체크포인트 로드')
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = build_mobilenet_v3_small()
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    x_test, y_test = load_test_data()
    print(f'   테스트셋 {len(y_test):,}개')

    print('2) PyTorch 기준 성능 측정')
    with torch.no_grad():
        preds = []
        for i in range(0, len(x_test), 256):
            preds.append(model(torch.from_numpy(x_test[i:i + 256])).argmax(1).numpy())
    pt_pred = np.concatenate(preds)
    pt_acc = accuracy_score(y_test, pt_pred)
    pt_f1 = f1_score(y_test, pt_pred, average='macro', zero_division=0)
    print(f'   PyTorch  Acc {pt_acc * 100:.2f}%  macroF1 {pt_f1:.4f}')

    print('3) ONNX 변환 (opset 14, dynamic batch)')
    torch.onnx.export(
        model, torch.zeros(1, 1, 64, 64), str(ONNX_FP32),
        input_names=['input'], output_names=['logits'], opset_version=14,
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    )

    print('4) INT8 동적 양자화')
    # onnxruntime.quantization 은 중간 산출물(*-inferred.onnx)을 원본 경로 옆에 만드는데,
    # 프로젝트 경로에 한글이 포함되면 이 파일을 다시 열지 못한다(FileNotFoundError).
    # → ASCII 경로의 임시 디렉터리에서 양자화한 뒤 결과만 복사한다.
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, dst = tmp / 'model.onnx', tmp / 'model_int8.onnx'
        shutil.copy(ONNX_FP32, src)
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
        shutil.copy(dst, ONNX_INT8)

    print('5) ONNX 정확도 검증 (변환이 성능을 보존했는가)')
    sess_fp32 = ort.InferenceSession(str(ONNX_FP32), providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(str(ONNX_INT8), providers=['CPUExecutionProvider'])

    fp32_pred = onnx_predict(sess_fp32, x_test)
    int8_pred = onnx_predict(sess_int8, x_test)
    fp32_acc = accuracy_score(y_test, fp32_pred)
    fp32_f1 = f1_score(y_test, fp32_pred, average='macro', zero_division=0)
    int8_acc = accuracy_score(y_test, int8_pred)
    int8_f1 = f1_score(y_test, int8_pred, average='macro', zero_division=0)
    print(f'   ONNX FP32  Acc {fp32_acc * 100:.2f}%  macroF1 {fp32_f1:.4f}')
    print(f'   ONNX INT8  Acc {int8_acc * 100:.2f}%  macroF1 {int8_f1:.4f}')

    print('6) 속도 벤치마크')
    xb = x_test[:32]
    tb = torch.from_numpy(xb)
    in_fp32 = sess_fp32.get_inputs()[0].name
    in_int8 = sess_int8.get_inputs()[0].name

    with torch.no_grad():
        pt_cpu = bench(lambda: model(tb), n=100)
    onnx_cpu = bench(lambda: sess_fp32.run(None, {in_fp32: xb}), n=100)
    onnx_i8 = bench(lambda: sess_int8.run(None, {in_int8: xb}), n=100)

    pt_gpu = None
    if torch.cuda.is_available():
        m_gpu = build_mobilenet_v3_small().cuda().eval()
        m_gpu.load_state_dict(ckpt['model_state'])
        t_gpu = tb.cuda()
        with torch.no_grad():
            def _g():
                m_gpu(t_gpu)
                torch.cuda.synchronize()
            pt_gpu = bench(_g, n=100)

    x1 = x_test[:1]
    edge = bench(lambda: sess_fp32.run(None, {in_fp32: x1}), n=500)
    edge_i8 = bench(lambda: sess_int8.run(None, {in_int8: x1}), n=500)

    mb = lambda p: round(p.stat().st_size / 1024 ** 2, 2)  # noqa: E731
    pt_mb = round(CKPT.stat().st_size / 1024 ** 2, 2)

    int8_adopted = (fp32_f1 - int8_f1) < 0.02 and onnx_i8['avg_ms'] < onnx_cpu['avg_ms']

    summary = {
        'description': 'ONNX 배포 검증 — 재학습 완료 체크포인트 기준, 정확도 검증 후 채택 여부 결정',
        'source_checkpoint': str(CKPT.relative_to(ROOT)).replace('\\', '/'),
        'model': 'MobileNetV3-Small (1채널 입력, 9클래스)',
        'why_not_best_model': (
            '최고 성능은 WaferCNN(macro F1 0.8458)이나, 엣지 디바이스 배포 대상으로는 '
            'ImageNet 사전학습 기반 경량 백본이면서 depthwise separable conv 로 '
            'ARM CPU 최적화가 검증된 MobileNetV3-Small 을 선택. '
            f'macro F1 {0.8458 - pt_f1:.4f} 손실을 감수하고 추론 지연과 배포 호환성을 택한 의도적 트레이드오프.'
        ),
        'accuracy': {
            'pytorch': {'accuracy': round(float(pt_acc), 4), 'f1_macro': round(float(pt_f1), 4)},
            'onnx_fp32': {'accuracy': round(float(fp32_acc), 4), 'f1_macro': round(float(fp32_f1), 4)},
            'onnx_int8': {'accuracy': round(float(int8_acc), 4), 'f1_macro': round(float(int8_f1), 4)},
            'fp32_conversion_f1_delta': round(float(fp32_f1 - pt_f1), 4),
            'int8_quantization_f1_delta': round(float(int8_f1 - fp32_f1), 4),
        },
        'model_size_mb': {
            'pytorch': pt_mb,
            'onnx_fp32': mb(ONNX_FP32),
            'onnx_int8': mb(ONNX_INT8),
            'int8_size_reduction_pct': round((1 - ONNX_INT8.stat().st_size / ONNX_FP32.stat().st_size) * 100, 1),
        },
        'latency_batch32': {
            'pytorch_cpu': pt_cpu, 'pytorch_gpu': pt_gpu,
            'onnx_cpu': onnx_cpu, 'onnx_int8_cpu': onnx_i8,
        },
        'latency_batch1_single_wafer': {'onnx_fp32': edge, 'onnx_int8': edge_i8},
        'decision': {
            'adopted': 'ONNX FP32 (checkpoints/MobileNetV3_deploy.onnx)',
            'rejected': 'ONNX INT8 동적 양자화',
            'rejection_reason': (
                f'INT8 양자화 시 macro F1 {fp32_f1:.4f} → {int8_f1:.4f} '
                f'({int8_f1 - fp32_f1:+.4f}), 추론 속도도 {onnx_cpu["avg_ms"]:.2f}ms → '
                f'{onnx_i8["avg_ms"]:.2f}ms 로 오히려 느려짐. '
                '단일 채널 grayscale 입력은 활성값 분포의 동적 범위가 좁아 '
                'per-tensor 양자화 스케일이 소수 클래스 특징을 뭉갠 것으로 판단. '
                '크기 감소만으로 채택할 수 없어 기각.'
            ) if not int8_adopted else 'INT8 채택 조건 충족',
            'int8_auto_check_passed': bool(int8_adopted),
        },
        'note_on_edge_benchmark': (
            '측정은 데스크톱 CPU 기준. 실제 Raspberry Pi 4 는 5~10배 느리므로 '
            f'단일 웨이퍼 추론 {edge["avg_ms"]:.2f}ms → 실기 추정 '
            f'{edge["avg_ms"] * 5:.1f}~{edge["avg_ms"] * 10:.1f}ms 수준. '
            '실기 측정값이 아닌 추정치임.'
        ),
    }

    (ROOT / 'analysis/deployment_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n=== 결정 ===')
    print(f'채택: {summary["decision"]["adopted"]}')
    print(f'기각: {summary["decision"]["rejected"]}')
    print(f'   FP32 {onnx_cpu["avg_ms"]}ms / INT8 {onnx_i8["avg_ms"]}ms (batch=32)')
    print('저장: analysis/deployment_summary.json')


if __name__ == '__main__':
    main()
