"""
Airflow DAG — wafer_defect_pipeline
Phase 1 Step 7 | WM-811K 웨이퍼 불량 검출 MLOps 파이프라인

실행 흐름 (8 Tasks):
  load_data → eda_analysis → data_classification → data_augmentation
  → model_training → model_evaluation → onnx_conversion → model_deployment

스케줄: @weekly (매주 자동 실행)
재시도: 1회 (실패 후 5분 대기)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

# ── 프로젝트 루트 (Airflow Variable → 환경 변수 → 상대경로 순 fallback)
def _get_project_root() -> Path:
    try:
        return Path(Variable.get("wafer_project_root"))
    except Exception:
        env_root = os.environ.get("AIRFLOW_VAR_WAFER_PROJECT_ROOT")
        if env_root:
            return Path(env_root)
        # 로컬 개발 시: dags/ 두 단계 위 = 프로젝트 루트
        return Path(__file__).parent.parent.parent


# ── 공통 DAG 기본값
default_args = {
    "owner": "wafer-mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 6, 21),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# Task 1 — load_data
# ═══════════════════════════════════════════════════════
def load_data(**context):
    """LSWMD.pkl 파일 존재 및 기본 구조 검증"""
    import pandas as pd
    import numpy as np

    root = _get_project_root()
    pkl_path = root / "data" / "raw" / "LSWMD.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(f"데이터 파일 없음: {pkl_path}")

    log.info("LSWMD.pkl 로드 중...")
    df = pd.read_pickle(pkl_path)

    required_cols = {"waferMap", "failureType"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # 레이블 정제 (중첩 리스트 해제)
    def extract_label(val):
        if isinstance(val, (list, np.ndarray)):
            flat = np.array(val).flatten()
            return str(flat[0]) if len(flat) > 0 else "unknown"
        return str(val)

    df["failureType_clean"] = df["failureType"].apply(extract_label)
    n_labeled = (df["failureType_clean"] != "0").sum()

    info = {
        "total_samples": int(len(df)),
        "n_labeled": int(n_labeled),
        "n_unlabeled": int(len(df) - n_labeled),
        "columns": list(df.columns),
    }

    log.info(f"데이터 로드 완료: 전체 {info['total_samples']:,}개 / 레이블 {info['n_labeled']:,}개")
    context["task_instance"].xcom_push(key="data_info", value=info)
    return info


# ═══════════════════════════════════════════════════════
# Task 2 — eda_analysis
# ═══════════════════════════════════════════════════════
def eda_analysis(**context):
    """클래스 분포 분석 및 통계 리포트 생성"""
    import json
    import numpy as np
    import pandas as pd

    root = _get_project_root()
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    data_info = context["task_instance"].xcom_pull(key="data_info", task_ids="load_data")
    log.info(f"EDA 시작 — 레이블 샘플: {data_info['n_labeled']:,}개")

    df = pd.read_pickle(root / "data" / "raw" / "LSWMD.pkl")

    def extract_label(val):
        if isinstance(val, (list, np.ndarray)):
            flat = np.array(val).flatten()
            return str(flat[0]) if len(flat) > 0 else "unknown"
        return str(val)

    df["failureType_clean"] = df["failureType"].apply(extract_label)
    labeled_df = df[df["failureType_clean"] != "0"]
    dist = labeled_df["failureType_clean"].value_counts().to_dict()

    max_cnt = max(dist.values())
    min_cnt = min(dist.values())
    imbalance_ratio = round(max_cnt / min_cnt, 2)

    eda_stats = {
        "run_date": datetime.now().isoformat(),
        "total_samples": int(len(df)),
        "labeled_samples": int(len(labeled_df)),
        "class_distribution": {k: int(v) for k, v in dist.items()},
        "imbalance_ratio": imbalance_ratio,
        "dominant_class": max(dist, key=dist.get),
        "rarest_class": min(dist, key=dist.get),
    }

    out_path = analysis_dir / "pipeline_eda_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eda_stats, f, ensure_ascii=False, indent=2)

    log.info(f"EDA 완료 — Imbalance Ratio: {imbalance_ratio}x → {out_path}")
    context["task_instance"].xcom_push(key="eda_stats", value=eda_stats)
    return eda_stats


# ═══════════════════════════════════════════════════════
# Task 3 — data_classification
# ═══════════════════════════════════════════════════════
def data_classification(**context):
    """전처리 결과물(split_indices.pkl, all_maps_resized.npy) 검증"""
    import pickle

    root = _get_project_root()
    processed_dir = root / "data" / "processed"

    split_path   = processed_dir / "split_indices.pkl"
    maps_path    = processed_dir / "all_maps_resized.npy"

    missing = [str(p) for p in [split_path, maps_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"전처리 결과물 없음: {missing}\n"
            "03_preprocessing.ipynb 를 먼저 실행하세요."
        )

    with open(split_path, "rb") as f:
        split = pickle.load(f)

    split_info = {
        "train_size": int(len(split["train_idx"])),
        "val_size":   int(len(split["val_idx"])),
        "test_size":  int(len(split["test_idx"])),
        "num_classes": int(len(split.get("class_order", []) or [])),
        "maps_path": str(maps_path),
    }

    log.info(
        f"분류 검증 완료 — "
        f"Train: {split_info['train_size']:,} / "
        f"Val: {split_info['val_size']:,} / "
        f"Test: {split_info['test_size']:,}"
    )
    context["task_instance"].xcom_push(key="split_info", value=split_info)
    return split_info


# ═══════════════════════════════════════════════════════
# Task 4 — data_augmentation
# ═══════════════════════════════════════════════════════
def data_augmentation(**context):
    """증강 설정 검증 및 DataLoader 동작 확인"""
    import pickle
    import numpy as np
    import yaml

    import torch
    from torch.utils.data import Dataset, DataLoader

    root = _get_project_root()
    config_path = root / "configs" / "augmentation_config.yaml"

    aug_config = {}
    if config_path.exists():
        with open(config_path) as f:
            aug_config = yaml.safe_load(f) or {}
        log.info(f"증강 설정 로드: {config_path}")
    else:
        log.warning("augmentation_config.yaml 없음 — 기본값 사용")
        aug_config = {
            "rotate_limit": 20, "flip_h": True, "flip_v": True,
            "gauss_noise": True, "coarse_dropout": True,
        }

    # 간단한 DataLoader 동작 테스트
    split_info = context["task_instance"].xcom_pull(key="split_info", task_ids="data_classification")
    all_maps = np.load(split_info["maps_path"])
    train_idx = pickle.load(open(root / "data" / "processed" / "split_indices.pkl", "rb"))["train_idx"]

    class _QuickDataset(Dataset):
        def __init__(self, arr):
            self.arr = arr
        def __len__(self):
            return len(self.arr)
        def __getitem__(self, i):
            return torch.from_numpy(self.arr[i].astype(np.float32) / 2.0).unsqueeze(0)

    sample = all_maps[train_idx[:200]]
    loader = DataLoader(_QuickDataset(sample), batch_size=32, shuffle=True, num_workers=0)
    batch  = next(iter(loader))

    assert batch.shape == (32, 1, 64, 64), f"배치 shape 오류: {batch.shape}"
    log.info(f"DataLoader 검증 완료 — batch shape: {tuple(batch.shape)}")

    context["task_instance"].xcom_push(key="aug_config", value=aug_config)
    return aug_config


# ═══════════════════════════════════════════════════════
# Task 5 — model_training
# ═══════════════════════════════════════════════════════
def model_training(**context):
    """WaferCNN 학습 (HPO 최적 파라미터 적용, 없으면 기본값)"""
    import json
    import pickle
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from sklearn.metrics import f1_score

    root = _get_project_root()
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ── 하이퍼파라미터 (HPO 결과 우선, 없으면 기본값)
    hpo_path = root / "analysis" / "hpo_results.json"
    if hpo_path.exists():
        with open(hpo_path) as f:
            hpo = json.load(f)
        best_p = hpo["best_params"]
        log.info(f"HPO 파라미터 로드: {best_p}")
    else:
        best_p = {"lr": 1e-3, "batch_size": 64, "dropout": 0.3, "weight_decay": 1e-4}
        log.info("HPO 결과 없음 — 기본 파라미터 사용")

    PIPELINE_EPOCHS = 10
    PATIENCE        = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 데이터 로드
    split_info = context["task_instance"].xcom_pull(key="split_info", task_ids="data_classification")
    all_maps   = np.load(split_info["maps_path"])

    with open(root / "data" / "processed" / "split_indices.pkl", "rb") as f:
        split = pickle.load(f)

    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    y         = split["encoded_labels"]
    cw        = split["class_weights"]

    class WaferDS(Dataset):
        def __init__(self, maps, labels):
            self.maps = maps
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, i):
            x = torch.from_numpy(self.maps[i].astype(np.float32) / 2.0).unsqueeze(0)
            return x, torch.tensor(int(self.labels[i]), dtype=torch.long)

    train_labels = y[train_idx].astype(int)
    val_labels   = y[val_idx].astype(int)

    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(cw[train_labels]),
        num_samples=len(train_labels), replacement=True
    )
    bs = int(best_p["batch_size"])
    t_loader = DataLoader(WaferDS(all_maps[train_idx], train_labels), batch_size=bs, sampler=sampler, num_workers=0)
    v_loader = DataLoader(WaferDS(all_maps[val_idx],   val_labels),   batch_size=bs, shuffle=False, num_workers=0)

    # ── 모델
    class WaferCNN(nn.Module):
        def __init__(self, num_classes=9, dropout=0.3):
            super().__init__()
            def _blk(i, o, d=0.1):
                return nn.Sequential(
                    nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                    nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                    nn.MaxPool2d(2), nn.Dropout2d(d),
                )
            self.net = nn.Sequential(
                _blk(1, 32, dropout*0.33), _blk(32, 64, dropout*0.33),
                _blk(64, 128, dropout*0.5), _blk(128, 256, dropout*0.67),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(256, 128), nn.ReLU(True),
                nn.Dropout(dropout), nn.Linear(128, num_classes),
            )
        def forward(self, x):
            return self.head(self.net(x))

    model     = WaferCNN(num_classes=9, dropout=float(best_p["dropout"])).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=float(best_p["lr"]), weight_decay=float(best_p["weight_decay"]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PIPELINE_EPOCHS, eta_min=1e-7)

    best_f1, patience_cnt, ckpt_path = 0.0, 0, None

    for ep in range(1, PIPELINE_EPOCHS + 1):
        # train
        model.train()
        for imgs, lbls in t_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(imgs), lbls).backward()
            optimizer.step()
        scheduler.step()

        # val
        model.eval()
        preds_all, true_all = [], []
        with torch.no_grad():
            for imgs, lbls in v_loader:
                preds_all.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
                true_all.extend(lbls.numpy())
        val_f1 = f1_score(true_all, preds_all, average="macro", zero_division=0)
        log.info(f"  Epoch {ep:>2}/{PIPELINE_EPOCHS} — Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1  = val_f1
            patience_cnt = 0
            ckpt_path = str(ckpt_dir / f"WaferCNN_pipeline_{ep:02d}_{val_f1:.4f}.pth")
            torch.save({"epoch": ep, "model_state": model.state_dict(), "val_f1": val_f1}, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                log.info(f"  Early Stopping (ep={ep})")
                break

    log.info(f"학습 완료 — Best Val F1: {best_f1:.4f}  ckpt: {ckpt_path}")
    context["task_instance"].xcom_push(key="ckpt_path", value=ckpt_path)
    context["task_instance"].xcom_push(key="best_val_f1", value=float(best_f1))
    return {"ckpt_path": ckpt_path, "best_val_f1": float(best_f1)}


# ═══════════════════════════════════════════════════════
# Task 6 — model_evaluation
# ═══════════════════════════════════════════════════════
def model_evaluation(**context):
    """체크포인트 로드 → Test Set 최종 평가"""
    import json
    import pickle
    import numpy as np

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

    root = _get_project_root()
    analysis_dir = root / "analysis"

    ckpt_path  = context["task_instance"].xcom_pull(key="ckpt_path", task_ids="model_training")
    split_info = context["task_instance"].xcom_pull(key="split_info", task_ids="data_classification")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_maps = np.load(split_info["maps_path"])
    with open(root / "data" / "processed" / "split_indices.pkl", "rb") as f:
        split = pickle.load(f)

    test_idx    = split["test_idx"]
    test_labels = split["encoded_labels"][test_idx].astype(int)
    class_order = split.get("class_order", [f"cls_{i}" for i in range(9)])

    class WaferDS(Dataset):
        def __init__(self, maps, labels):
            self.maps = maps; self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return torch.from_numpy(self.maps[i].astype(np.float32) / 2.0).unsqueeze(0), \
                   torch.tensor(int(self.labels[i]), dtype=torch.long)

    test_loader = DataLoader(WaferDS(all_maps[test_idx], test_labels), batch_size=128, shuffle=False, num_workers=0)

    class WaferCNN(nn.Module):
        def __init__(self, num_classes=9, dropout=0.3):
            super().__init__()
            def _blk(i, o, d=0.1):
                return nn.Sequential(
                    nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                    nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                    nn.MaxPool2d(2), nn.Dropout2d(d),
                )
            self.net = nn.Sequential(_blk(1,32), _blk(32,64), _blk(64,128), _blk(128,256))
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(128, num_classes),
            )
        def forward(self, x): return self.head(self.net(x))

    model = WaferCNN(num_classes=9).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds_all, true_all = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            preds_all.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
            true_all.extend(lbls.numpy())

    test_f1  = f1_score(true_all, preds_all, average="macro", zero_division=0)
    test_acc = sum(p == t for p, t in zip(preds_all, true_all)) / len(true_all)

    eval_result = {
        "run_date":     datetime.now().isoformat(),
        "checkpoint":   ckpt_path,
        "test_accuracy": round(float(test_acc), 4),
        "test_f1_macro": round(float(test_f1),  4),
        "classification_report": classification_report(
            true_all, preds_all, target_names=class_order, output_dict=True, zero_division=0
        ),
    }

    out_path = analysis_dir / "pipeline_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    log.info(f"평가 완료 — Test Acc: {test_acc*100:.2f}%  F1: {test_f1:.4f}")
    context["task_instance"].xcom_push(key="eval_result", value={"test_acc": float(test_acc), "test_f1": float(test_f1)})
    return eval_result


# ═══════════════════════════════════════════════════════
# Task 7 — onnx_conversion
# ═══════════════════════════════════════════════════════
def onnx_conversion(**context):
    """PyTorch 모델 → ONNX 변환 및 검증"""
    import numpy as np
    import torch
    import torch.nn as nn

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        raise ImportError("pip install onnx onnxruntime 필요")

    root     = _get_project_root()
    ckpt_dir = root / "checkpoints"
    ckpt_path = context["task_instance"].xcom_pull(key="ckpt_path", task_ids="model_training")

    DEVICE = torch.device("cpu")  # ONNX 변환은 CPU에서 수행

    class WaferCNN(nn.Module):
        def __init__(self, num_classes=9, dropout=0.0):  # dropout=0 (추론 시)
            super().__init__()
            def _blk(i, o, d=0.0):
                return nn.Sequential(
                    nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                    nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                    nn.MaxPool2d(2), nn.Dropout2d(d),
                )
            self.net = nn.Sequential(_blk(1,32), _blk(32,64), _blk(64,128), _blk(128,256))
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(128, num_classes),
            )
        def forward(self, x): return self.head(self.net(x))

    model = WaferCNN(num_classes=9, dropout=0.0).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ONNX 변환 (opset 14, 동적 배치)
    dummy_input = torch.zeros(1, 1, 64, 64)
    onnx_path   = str(ckpt_dir / "wafer_cnn_pipeline.onnx")

    torch.onnx.export(
        model, dummy_input, onnx_path,
        opset_version=14,
        input_names=["wafer_map"],
        output_names=["class_logits"],
        dynamic_axes={"wafer_map": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
    )

    # ONNX 모델 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # ONNX Runtime 추론 속도 테스트 (100 샘플)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    test_input = np.random.rand(1, 1, 64, 64).astype(np.float32)

    import time
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, {"wafer_map": test_input})
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = round(float(np.mean(times)), 2)
    p95_ms = round(float(np.percentile(times, 95)), 2)

    log.info(f"ONNX 변환 완료: {onnx_path}")
    log.info(f"  추론 속도 — 평균: {avg_ms}ms / P95: {p95_ms}ms / 목표: <50ms")

    result = {"onnx_path": onnx_path, "avg_ms": avg_ms, "p95_ms": p95_ms, "target_met": bool(p95_ms < 50)}
    context["task_instance"].xcom_push(key="onnx_result", value=result)
    return result


# ═══════════════════════════════════════════════════════
# Task 8 — model_deployment
# ═══════════════════════════════════════════════════════
def model_deployment(**context):
    """MLFlow에 파이프라인 실행 결과 기록 + 배포 요약 저장"""
    import json
    import mlflow

    root = _get_project_root()
    analysis_dir = root / "analysis"

    eval_result  = context["task_instance"].xcom_pull(key="eval_result",  task_ids="model_evaluation")
    onnx_result  = context["task_instance"].xcom_pull(key="onnx_result",  task_ids="onnx_conversion")
    best_val_f1  = context["task_instance"].xcom_pull(key="best_val_f1",  task_ids="model_training")
    ckpt_path    = context["task_instance"].xcom_pull(key="ckpt_path",    task_ids="model_training")

    mlflow.set_tracking_uri(str(root / "mlruns"))
    mlflow.set_experiment("wafer-defect-detection")

    with mlflow.start_run(run_name="pipeline_weekly_run", tags={"type": "pipeline", "step": "7"}):
        mlflow.log_params({"pipeline": "wafer_defect_pipeline", "schedule": "weekly"})
        mlflow.log_metrics({
            "val_f1_macro":  float(best_val_f1),
            "test_f1_macro": float(eval_result.get("test_f1", 0)),
            "test_accuracy": float(eval_result.get("test_acc", 0)),
            "onnx_avg_ms":   float(onnx_result.get("avg_ms", 0)),
            "onnx_p95_ms":   float(onnx_result.get("p95_ms", 0)),
        })
        mlflow.log_artifact(str(analysis_dir / "pipeline_eval_results.json"))

    # 배포 요약 저장
    deployment_summary = {
        "pipeline_name": "wafer_defect_pipeline",
        "run_date":      datetime.now().isoformat(),
        "checkpoint":    ckpt_path,
        "onnx_model":    onnx_result.get("onnx_path"),
        "performance": {
            "val_f1_macro":  float(best_val_f1),
            "test_f1_macro": float(eval_result.get("test_f1", 0)),
            "test_accuracy": float(eval_result.get("test_acc", 0)),
        },
        "inference": {
            "onnx_avg_ms": float(onnx_result.get("avg_ms", 0)),
            "onnx_p95_ms": float(onnx_result.get("p95_ms", 0)),
            "target_50ms_met": bool(onnx_result.get("target_met", False)),
        },
        "status": "deployed",
    }

    out_path = analysis_dir / "pipeline_deployment_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deployment_summary, f, ensure_ascii=False, indent=2)

    log.info(f"배포 완료 — Test F1: {eval_result.get('test_f1', 0):.4f} | ONNX P95: {onnx_result.get('p95_ms', 0):.2f}ms")
    log.info(f"배포 요약: {out_path}")
    return deployment_summary


# ═══════════════════════════════════════════════════════
# DAG 정의
# ═══════════════════════════════════════════════════════
with DAG(
    dag_id="wafer_defect_pipeline",
    description="WM-811K 웨이퍼 불량 검출 MLOps 파이프라인 (8 Tasks)",
    default_args=default_args,
    schedule_interval="@weekly",
    catchup=False,
    tags=["wafer", "mlops", "sk-hynix"],
    doc_md="""
    ## wafer_defect_pipeline

    WM-811K 데이터셋 기반 웨이퍼 불량 검출 자동화 파이프라인.

    **실행 순서:**
    ```
    load_data → eda_analysis → data_classification → data_augmentation
    → model_training → model_evaluation → onnx_conversion → model_deployment
    ```

    **스케줄:** 매주 자동 실행
    **목표 성능:** Test F1-macro ≥ 0.80, ONNX 추론 P95 < 50ms
    """,
) as dag:

    t1_load_data = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    t2_eda = PythonOperator(
        task_id="eda_analysis",
        python_callable=eda_analysis,
    )

    t3_classify = PythonOperator(
        task_id="data_classification",
        python_callable=data_classification,
    )

    t4_augment = PythonOperator(
        task_id="data_augmentation",
        python_callable=data_augmentation,
    )

    t5_train = PythonOperator(
        task_id="model_training",
        python_callable=model_training,
        execution_timeout=timedelta(hours=2),
    )

    t6_eval = PythonOperator(
        task_id="model_evaluation",
        python_callable=model_evaluation,
    )

    t7_onnx = PythonOperator(
        task_id="onnx_conversion",
        python_callable=onnx_conversion,
    )

    t8_deploy = PythonOperator(
        task_id="model_deployment",
        python_callable=model_deployment,
    )

    # ── Task 의존성 (순차 실행)
    t1_load_data >> t2_eda >> t3_classify >> t4_augment \
        >> t5_train >> t6_eval >> t7_onnx >> t8_deploy
