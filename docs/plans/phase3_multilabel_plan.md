# WM-811K Phase 3 — 멀티레이블 합성 데이터 + 다중 CNN 아키텍처 확장 계획

> **작성일:** 2026-07-03  
> **기반:** WM-811K v2.0 (Phase 1~2 완료 기준)  
> **목표 직무:** SK하이닉스 Device Engineering / AI 반도체 검사 시스템

---

## 0. 현황 및 문제 정의

### 0-1. 현재 모델 아키텍처 정정

> **사용자 참고:** 현재 프로젝트의 베이스 모델은 ResNet-18이 아닌 **커스텀 WaferCNN**입니다.

| 모델 | F1 (macro) | 비고 |
|------|-----------|------|
| **WaferCNN (커스텀)** | **0.8458 ✅** | 4 Conv Block + GAP, ~2M params, from scratch |
| MobileNetV3-Small | 0.7369 ❌ | ImageNet 사전학습 → 웨이퍼 도메인 불일치 |
| EfficientNet-B0 | 0.7944 ❌ | 동일 원인 |
| ViT-Tiny | 0.8352 ❌ | 어텐션 효과, but 소규모 데이터 과적합 |

**커스텀 CNN이 사전학습 모델을 이긴 이유:**  
웨이퍼 맵은 픽셀값이 `{0, 1, 2}` 3가지뿐인 이진 공간 패턴 이미지.  
ImageNet 사전학습의 귀납적 편향(texture, color, edge)이 오히려 학습을 방해함.  
→ 다중 CNN 비교 실험 시 이 인사이트를 전략에 반영해야 함.

### 0-2. 현재 데이터의 한계

```
WM-811K 레이블 분포:
  - 단일 결함 패턴만 존재 (1개 웨이퍼 = 1개 결함 유형)
  - 실제 FAB: 복수 결함이 동시 발생 (Center + Edge-Ring, Loc + Random 등)
  - 전체 811,457개 중 638,507개 (78.7%)가 미라벨
```

---

## 1. Phase 3-A: 멀티레이블 합성 데이터 생성

### 1-1. 왜 합성이 필요한가

WM-811K는 **단일 결함(Single-defect)** 레이블만 제공한다.  
현실 공정에서는 복수 결함이 동시 발생한다:

| 실제 복합 결함 예시 | 물리적 원인 |
|--------------------|------------|
| Edge-Ring + Center | 온도 구배 + 열처리 동시 이상 |
| Loc + Random | 파티클 오염 + 챔버 국소 결함 |
| Scratch + Edge-Loc | CMP 과연마 + 리테이닝 링 이상 |
| Donut + Random | PR 두께 불균일 + 오염 |

### 1-2. 합성 전략 3단계

#### 전략 A — Pixel-Level Overlay (즉시 구현 가능)

두 단일 결함 맵을 픽셀별 OR 연산으로 합성.

```python
# src/synthesis/overlay.py
import numpy as np
from typing import Tuple, List

def overlay_defect_maps(
    map_a: np.ndarray,   # (64, 64) uint8, 값: {0,1,2}
    map_b: np.ndarray,
    label_a: int,
    label_b: int,
) -> Tuple[np.ndarray, List[int]]:
    """
    두 웨이퍼 맵을 합성: 불량다이(2)는 양 맵 모두에서 보존.
    비활성(0)은 어느 한 맵이 활성(1 or 2)이면 활성으로 덮어씀.
    """
    # 비활성 마스크: 두 맵 모두 0인 픽셀만 비활성
    active_mask = (map_a > 0) | (map_b > 0)
    
    combined = np.ones_like(map_a)         # 기본값: 정상다이(1)
    combined[~active_mask] = 0             # 비활성 영역 복원
    combined[map_a == 2] = 2              # map_a 불량 이식
    combined[map_b == 2] = 2              # map_b 불량 이식 (OR)
    
    # 멀티레이블 벡터 (9차원 멀티-핫)
    # 클래스 인덱스: 0=none, 1=Center, 2=Donut, 3=Edge-Loc,
    #               4=Edge-Ring, 5=Loc, 6=Near-full, 7=Random, 8=Scratch
    labels = [label_a, label_b]
    return combined, labels
```

#### 전략 B — Region Copy-Paste (공간 보존 합성)

공여(donor) 맵에서 결함 영역만 마스크로 추출 → 기본 맵에 이식.

```python
def copy_paste_defect(
    base_map: np.ndarray,
    donor_map: np.ndarray,
    donor_label: int,
    jitter: int = 3,    # 위치 무작위 이동 (±3 픽셀)
) -> Tuple[np.ndarray, List[int]]:
    """
    donor_map의 결함 영역(==2)을 base_map에 이식.
    jitter: 이식 위치를 소폭 랜덤 이동 → 다양성 증가.
    """
    defect_mask = (donor_map == 2)
    dy = np.random.randint(-jitter, jitter + 1)
    dx = np.random.randint(-jitter, jitter + 1)
    shifted_mask = np.roll(np.roll(defect_mask, dy, axis=0), dx, axis=1)
    
    result = base_map.copy()
    result[shifted_mask] = 2
    
    base_labels = [] if base_map.max() < 2 else [get_dominant_label(base_map)]
    return result, base_labels + [donor_label]
```

#### 전략 C — GAN 기반 합성 (고급, Phase 3-C 이후)

- **DefectGAN / Conditional DCGAN**: 결함 유형 조합을 조건 벡터로 입력 → 현실적 합성 맵 생성  
- 구현 복잡도 높음 → 본 Phase에서는 A·B 검증 후 진행

### 1-3. 합성 데이터셋 설계

```
합성 비율 설계 (목표 총 샘플 수: ~250,000)
  ─────────────────────────────────────────────
  단일 결함 (원본 유지):    172,950개  (기존)
  2중 결함 합성 (A+B 조합): 60,000개   (+)
  3중 결함 합성:            15,000개   (+)
  미라벨 → 준지도 학습:    638,507개  (Phase 3-B에서 활용)
  ─────────────────────────────────────────────
  합계:                    ~247,950개
```

**합성 조합 우선순위** (Pearson 상관계수 기반 공정 연관성):

| 조합 | 발생 확률 | 공정 원인 |
|------|---------|---------|
| Edge-Ring + Center | 높음 | 열처리 전반 붕괴 |
| Loc + Random | 높음 | 파티클 복합 오염 |
| Edge-Loc + Scratch | 중간 | CMP 복합 이상 |
| Donut + Edge-Loc | 중간 | 포토/CMP 동시 이상 |
| Near-full + * | 낮음 | 이미 전면 불량이므로 드묾 |

### 1-4. 레이블 스키마 변환

```python
# 기존: 단일 정수 레이블
label = 3   # Edge-Loc

# 변경: 9차원 멀티-핫 벡터
# [none, Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch]
label = torch.tensor([0, 0, 0, 1, 0, 1, 0, 0, 0], dtype=torch.float32)
# → Edge-Loc + Loc 복합 결함
```

---

## 2. Phase 3-B: 멀티레이블 분류 모델 구조

### 2-1. 손실 함수 변경

```python
# 기존: CrossEntropyLoss (단일 레이블)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 변경 1: BCEWithLogitsLoss (기본 멀티레이블)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# 변경 2: AsymmetricLoss (불균형 강화, 권장)
# - 양성 샘플(불량)에 높은 가중치, 쉬운 음성(정상)은 무시
# - ASL: L = -(1-p)^γ⁺ · log(p)  for positive
#          -p^γ⁻ · log(1-p)       for negative  (γ⁻ > γ⁺)
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        loss_pos = targets       * torch.log(probs_pos.clamp(min=1e-8)) * (1 - probs_pos) ** self.gamma_pos
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=1e-8)) * probs_neg       ** self.gamma_neg
        return -(loss_pos + loss_neg).mean()
```

### 2-2. 평가 지표 변경

```python
# 단일레이블 지표 (기존)
from sklearn.metrics import f1_score
f1_macro = f1_score(y_true, y_pred, average='macro')

# 멀티레이블 지표 (변경)
from sklearn.metrics import (
    average_precision_score,    # mAP: 각 클래스 AP의 평균
    hamming_loss,               # 잘못 예측된 레이블 비율
    f1_score,                   # Micro/Macro/Sample F1
)

metrics = {
    'mAP':          average_precision_score(y_true, y_prob, average='macro'),
    'hamming_loss': hamming_loss(y_true, y_pred),
    'f1_micro':     f1_score(y_true, y_pred, average='micro'),
    'f1_macro':     f1_score(y_true, y_pred, average='macro'),
    'f1_sample':    f1_score(y_true, y_pred, average='samples'),
}
```

### 2-3. 멀티레이블 WaferCNN 헤드 수정

```python
# src/model_multilabel.py
class MultiLabelWaferHead(nn.Module):
    """단일레이블 head를 멀티레이블 sigmoid head로 교체."""
    def __init__(self, in_features: int, num_classes: int = 9):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            # Sigmoid는 BCEWithLogitsLoss에 포함 → 학습 시 생략
            # 추론 시: torch.sigmoid(logits) > threshold
        )
    
    def forward(self, x):
        return self.fc(x)

# 추론 시 임계값 최적화 (per-class 최적화 가능)
def predict_multilabel(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    return (probs > threshold).int()
```

---

## 3. Phase 3-C: 다중 CNN 아키텍처 통합 비교 실험

### 3-1. 현 WaferCNN이 ResNet보다 유리한 이유 분석

```
웨이퍼 맵 특성:
  픽셀값:   {0, 1, 2}  → 3가지 이산값 (연속 텍스처 없음)
  채널:     1-ch 그레이스케일
  정보 유형: 공간 패턴 (어디에 불량이 있는가)
  
ImageNet 사전학습 모델의 문제:
  - RGB 3채널 → 1채널 변환 시 특징 손실
  - 텍스처·색상·엣지 특징 → 웨이퍼 이진 패턴과 불일치
  - 사전학습 가중치의 귀납적 편향이 오히려 수렴 방해
```

→ **전략: 모든 모델을 from-scratch + 웨이퍼 도메인 특화 전처리로 공정 비교**

### 3-2. 비교 대상 아키텍처

| 모델 | 파라미터 수 | 특징 | 기대 효과 |
|------|-----------|------|---------|
| **WaferCNN (베이스라인)** | ~2M | 커스텀 4-Conv, 도메인 최적화 | 현 최고 성능 기준선 |
| **ResNet-18** | 11.7M | 잔차(skip) 연결, 그래디언트 안정화 | 깊이 증가 시 성능 한계 극복 |
| **ResNet-34** | 21.8M | 더 깊은 잔차 블록 | 공간 패턴 계층적 추출 |
| **SE-ResNet-18** | 11.8M | 채널 어텐션 (Squeeze-Excitation) | 결함별 채널 중요도 적응 |
| **DenseNet-121** | 8.0M | 밀집 연결 (Dense Connection) | 그래디언트 흐름 최적화, 소규모 데이터 강점 |
| **ConvNeXt-Tiny** | 28M | ViT 설계 원칙을 CNN에 적용 | 최신 CNN 패러다임, LayerNorm + GELU |
| **EfficientNet-B2** | 9.2M | 복합 스케일링 (width·depth·resolution) | 파라미터 효율성 |
| **MobileNetV3-Small** | 2.5M | 하드스위시, SE 블록, 경량화 | ONNX 배포 목표 |

> **제외 결정:** ViT-Tiny → 이미 실험(F1=0.8352), 학습 데이터 부족 시 어텐션 과적합 확인됨.  
> EfficientNetV2-S → 학습 효율 개선이지만 데이터 스케줄링 요구, Phase 4로 이연.

### 3-3. 통합 모델 팩토리 설계

```python
# src/model_factory.py
import timm
import torch.nn as nn

BACKBONE_REGISTRY = {
    'wafer_cnn':      lambda: WaferCNN(num_classes=1, head=False),
    'resnet18':       lambda: timm.create_model('resnet18',      in_chans=1, num_classes=0),
    'resnet34':       lambda: timm.create_model('resnet34',      in_chans=1, num_classes=0),
    'se_resnet18':    lambda: timm.create_model('seresnet18',    in_chans=1, num_classes=0),
    'densenet121':    lambda: timm.create_model('densenet121',   in_chans=1, num_classes=0),
    'convnext_tiny':  lambda: timm.create_model('convnext_tiny', in_chans=1, num_classes=0),
    'efficientnet_b2':lambda: timm.create_model('efficientnet_b2', in_chans=1, num_classes=0),
    'mobilenetv3':    lambda: timm.create_model('mobilenetv3_small_100', in_chans=1, num_classes=0),
}

def build_multilabel_model(backbone_name: str, num_classes: int = 9) -> nn.Module:
    """팩토리 패턴: 백본 + 멀티레이블 헤드 조합."""
    backbone  = BACKBONE_REGISTRY[backbone_name]()
    in_feats  = backbone.num_features  # timm 공통 속성
    head      = MultiLabelWaferHead(in_feats, num_classes)
    return nn.Sequential(
        nn.Sequential(backbone),   # feature extractor
        nn.AdaptiveAvgPool2d(1),   # (B, C, 1, 1)
        nn.Flatten(),              # (B, C)
        head,                      # (B, num_classes)
    )

# 사용 예시
model = build_multilabel_model('resnet18', num_classes=9)
```

### 3-4. 도메인 특화 전처리 파이프라인

```python
# 웨이퍼 맵 특성을 고려한 Albumentations 전략
import albumentations as A

train_transform = A.Compose([
    # 공간 변환 (웨이퍼는 회전 대칭성 有)
    A.Rotate(limit=360, p=0.8),           # 임의 회전 (웨이퍼 노치 제외 시)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    
    # 픽셀값 보존 변환 (0/1/2 이산값 깨지지 않도록)
    A.GridDistortion(distort_limit=0.05, p=0.3),  # 미세 왜곡
    A.CoarseDropout(                              # 일부 다이 무작위 마스킹
        max_holes=8, max_height=4, max_width=4,
        fill_value=1, p=0.3
    ),
    
    # 정규화: {0,1,2} → [-1, 1]
    A.Normalize(mean=[0.5], std=[0.5]),
])

# 주의: 일반 밝기·대비 변환 금지 → 픽셀값 {0,1,2} 의미 파괴
```

### 3-5. MLflow 비교 실험 설계

```python
# 06_multilabel_comparison.ipynb 구조
import mlflow

EXPERIMENT_NAME = "wafer-multilabel-comparison"

MODELS_TO_COMPARE = [
    'wafer_cnn', 'resnet18', 'resnet34', 'se_resnet18',
    'densenet121', 'convnext_tiny', 'efficientnet_b2', 'mobilenetv3',
]

for model_name in MODELS_TO_COMPARE:
    with mlflow.start_run(run_name=f"multilabel_{model_name}"):
        # 공통 하이퍼파라미터
        mlflow.log_params({
            'model':      model_name,
            'loss':       'AsymmetricLoss',
            'optimizer':  'AdamW',
            'lr':         3e-4,
            'epochs':     50,
            'batch_size': 64,
            'scheduler':  'CosineAnnealingLR',
        })
        
        # 학습 실행 → 체크포인트 저장
        # ...
        
        # 멀티레이블 지표 기록
        mlflow.log_metrics({
            'test_mAP':          mAP,
            'test_f1_macro':     f1_macro,
            'test_f1_micro':     f1_micro,
            'test_hamming_loss': h_loss,
            'params_M':          count_params(model) / 1e6,
            'inference_ms':      measure_inference_time(model),
        })
```

---

## 4. Phase 3-D: 준지도 학습으로 미라벨 데이터 활용

### 4-1. 동기

```
WM-811K 미라벨 데이터: 638,507개 (78.7%)
현재 활용: 0% (완전 폐기)
→ 멀티레이블 맥락에서 Teacher-Student 방식으로 활용 가능
```

### 4-2. Mean Teacher 방식 (권장)

```
Student Model                    Teacher Model
  ↓ (학습 중 업데이트)            ↓ (EMA로 업데이트: θ_t = α·θ_t + (1-α)·θ_s)
  ↓                              ↓
[라벨 데이터] → 지도 손실         [미라벨 데이터] → 일관성 손실
                          (Student와 Teacher 예측 차이 최소화)
```

```python
# 손실 함수 조합
total_loss = supervised_loss + λ * consistency_loss
# λ: 0 → 1 으로 ramp-up (초기에는 지도 학습 위주)
```

---

## 5. 단계별 구현 로드맵

```
Phase 3-A: 합성 데이터 생성         [2~3일]
  ├── synthesis/overlay.py         Pixel-Level Overlay 구현
  ├── synthesis/copy_paste.py      Copy-Paste 구현  
  ├── synthesis/dataset.py         멀티레이블 Dataset 클래스
  └── 합성 결과 시각화 + EDA

Phase 3-B: 멀티레이블 모델         [3~4일]
  ├── src/model_multilabel.py      MultiLabelWaferHead
  ├── src/losses.py                AsymmetricLoss
  ├── 13_multilabel_baseline.ipynb WaferCNN → 멀티레이블 변환
  └── 목표: mAP ≥ 0.85

Phase 3-C: 다중 CNN 비교           [5~7일]
  ├── src/model_factory.py         통합 팩토리
  ├── 14_multilabel_comparison.ipynb  8개 모델 MLflow 실험
  ├── 목표: 최고 성능 모델 mAP ≥ 0.88
  └── 대시보드: 모델별 성능 시각화

Phase 3-D: 준지도 학습             [4~5일]
  ├── src/mean_teacher.py          EMA Teacher 구현
  ├── 15_semisupervised.ipynb      638K 미라벨 활용
  └── 목표: F1 ≥ 0.90

Phase 3-E: ONNX 재배포             [1~2일]
  ├── 최고 성능 모델 ONNX 변환
  ├── 멀티레이블 출력 검증
  └── 대시보드 업데이트 (복합 결함 표시)
```

---

## 6. 예상 성능 목표

| 단계 | 모델 | 지표 | 목표값 |
|------|------|------|------|
| 현재 | WaferCNN (단일레이블) | F1 macro | 0.8458 ✅ |
| Phase 3-A+B | WaferCNN (멀티레이블) | mAP | ≥ 0.82 |
| Phase 3-C | 최적 CNN (멀티레이블) | mAP | ≥ 0.88 |
| Phase 3-D | 최적 CNN + 준지도 | mAP | ≥ 0.90 |

---

## 7. 핵심 기술 판단 기준

### 7-1. 어떤 CNN이 웨이퍼 도메인에 적합할까?

| 아키텍처 특징 | 웨이퍼 적합성 | 이유 |
|-------------|------------|------|
| 잔차 연결 (ResNet) | ★★★★☆ | 그래디언트 안정화 → 깊은 공간 패턴 학습 가능 |
| 밀집 연결 (DenseNet) | ★★★★★ | 소규모 데이터에서 특징 재사용 효율 높음 |
| 채널 어텐션 (SE-Net) | ★★★★☆ | 결함 유형별 채널 중요도 적응적 조정 |
| 복합 스케일링 (EfficientNet) | ★★★☆☆ | 파라미터 효율 우수, 도메인 이탈 위험 |
| Modern CNN (ConvNeXt) | ★★★☆☆ | 최신 설계지만 작은 데이터셋에서 불확실 |

→ **DenseNet-121 + SE-ResNet-18이 웨이퍼 멀티레이블에서 유망**하다고 예상.

### 7-2. 합성 데이터 품질 검증 방법

```python
# 합성 전후 품질 비교 체크리스트
def validate_synthesis(original_map, synthesized_map, labels):
    checks = {
        '활성 다이 비율 유지': check_active_ratio(original_map, synthesized_map),
        '불량 다이 겹침 해소': check_no_conflict(synthesized_map),
        '웨이퍼 원형 마스크 보존': check_circle_mask(synthesized_map),
        '레이블 정합성': check_label_consistency(synthesized_map, labels),
    }
    return all(checks.values()), checks
```

---

## 8. 산업 연결점 (포트폴리오 관점)

| 본 Phase 과제 | 산업 대응 기술 | SK하이닉스 직무 연관성 |
|-------------|------------|------------------|
| 멀티레이블 합성 | 실제 FAB 복합 결함 데이터 증강 | 실제 공정에서는 단일 결함이 드묾 |
| AsymmetricLoss | 불량률 < 0.1% 극불균형 처리 | 양산 FAB의 실제 불량 분포 |
| 다중 CNN 비교 | ADC 엔진 선택 기준 | 장비 선정 + 알고리즘 평가 역량 |
| DenseNet + SE 어텐션 | KLA DefectWise AI 유사 개념 | 결함 유형별 가중치 적응 |
| 준지도 학습 | 미라벨 FAB 데이터 활용 | 실제 FAB 라벨 부족 문제 해결 |
| 멀티레이블 ONNX | 인라인 ADC 배포 | 실시간 추론 요구사항 대응 |

---

## 참고 자료

- [Asymmetric Loss for Multi-Label Classification (ICCV 2021)](https://arxiv.org/abs/2009.14119)
- [DenseNet: Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993)
- [Squeeze-and-Excitation Networks (CVPR 2018)](https://arxiv.org/abs/1709.01507)
- [ConvNeXt: A ConvNet for the 2020s (CVPR 2022)](https://arxiv.org/abs/2201.03545)
- [Mean Teacher Semi-Supervised Learning](https://arxiv.org/abs/1703.01780)
- [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [WM-811K Dataset — Wafer Map Failure Pattern Recognition](https://www.researchgate.net/publication/340624801)
