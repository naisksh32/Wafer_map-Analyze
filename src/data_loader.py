"""WM-811K 웨이퍼 맵 Dataset & DataLoader 유틸리티 (Phase 1 Step 3)

사용법:
    all_maps_resized = np.load('data/processed/all_maps_resized.npy')  # (N,64,64) uint8
    with open('data/processed/split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
    train_loader, val_loader, test_loader = get_dataloaders(
        all_maps_resized, split, batch_size=64
    )

불균형 보정 전략:
    WeightedRandomSampler 단독 사용 (split['class_weights'] 활용)
    CrossEntropyLoss에는 weight 인자를 넣지 말 것 — 이중 보정 시 소수 클래스 편향 심화
"""

import pickle
import numpy as np
import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

CLASS_ORDER  = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
                'Loc', 'Near-full', 'Random', 'Scratch']
LABEL_MAP    = {cls: idx for idx, cls in enumerate(CLASS_ORDER)}
NUM_CLASSES  = len(CLASS_ORDER)
NORM_DIVISOR = 2.0


class WaferMapDataset(Dataset):
    """WM-811K 웨이퍼 맵 PyTorch Dataset.

    maps_array : (N, 64, 64) uint8 — 미리 64×64 리사이즈된 배열
    __getitem__: uint8 → float32 ÷2 정규화만 수행 (resize 없음)
    반환: (1, 64, 64) float32 tensor, long label (0~8)
    """

    def __init__(self, maps_array, labels, transform=None,
                 norm_divisor=NORM_DIVISOR):
        self.maps_array   = maps_array
        self.labels       = labels
        self.transform    = transform
        self.norm_divisor = norm_divisor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wmap = self.maps_array[idx].astype(np.float32) / self.norm_divisor

        if self.transform is not None:
            wmap = self.transform(
                image=wmap[:, :, np.newaxis]
            )['image'][:, :, 0]

        tensor = torch.from_numpy(np.clip(wmap, 0.0, 1.0).copy()).unsqueeze(0)
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


def build_train_transform():
    """학습용 Albumentations 증강 파이프라인 반환."""
    return A.Compose([
        A.Rotate(limit=20, border_mode=0, value=0.0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
        A.CoarseDropout(
            max_holes=4, max_height=8, max_width=8,
            min_holes=1, min_height=4, min_width=4,
            fill_value=0.0, p=0.3,
        ),
    ])


def get_dataloaders(all_maps_resized, split, batch_size=64, num_workers=0):
    """(N,64,64) uint8 배열과 split_indices.pkl 기반 DataLoader 반환.

    num_workers 기본값=0: Windows/Jupyter 환경 최적화
        Linux 서버에선 2~4로 변경 가능.
    """
    train_idx      = split['train_idx']
    val_idx        = split['val_idx']
    test_idx       = split['test_idx']
    encoded_labels = split['encoded_labels']
    class_weights  = split['class_weights']

    train_maps   = all_maps_resized[train_idx]
    val_maps     = all_maps_resized[val_idx]
    test_maps    = all_maps_resized[test_idx]
    train_labels = encoded_labels[train_idx].astype(int)
    val_labels   = encoded_labels[val_idx].astype(int)
    test_labels  = encoded_labels[test_idx].astype(int)

    train_transform = build_train_transform()

    train_dataset = WaferMapDataset(train_maps, train_labels, transform=train_transform)
    val_dataset   = WaferMapDataset(val_maps,   val_labels)
    test_dataset  = WaferMapDataset(test_maps,  test_labels)

    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(train_labels),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
