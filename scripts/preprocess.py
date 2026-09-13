"""03_preprocessing.ipynb 의 핵심 단계를 스크립트로 재현 — 학습 데이터 산출물 생성

입력 : data/raw/LSWMD.pkl                (WM-811K 원본, kagglehub 로 다운로드)
출력 : data/processed/all_maps_resized.npy   (172,950, 64, 64) uint8
       data/processed/split_indices.pkl      train/val/test 인덱스 + 레이블 + class_weights
       data/processed/class_weights.npy

노트북과 동일한 SEED=42 · StratifiedShuffleSplit 70/15/15 이므로 테스트셋(25,943개)
분포가 README 에 기재된 값과 일치해야 한다.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH     = ROOT / 'data/raw/LSWMD.pkl'
PROCESSED_DIR = ROOT / 'data/processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CLASS_ORDER = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
               'Loc', 'Near-full', 'Random', 'Scratch']
LABEL_MAP = {c: i for i, c in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER)
RESIZE = (64, 64)


def extract_label(val):
    if isinstance(val, (list, np.ndarray)):
        flat = np.array(val).flatten()
        return str(flat[0]) if len(flat) > 0 else 'unknown'
    return str(val)


class _LegacyPandasUnpickler(pickle.Unpickler):
    """LSWMD.pkl 은 pandas 0.1x 로 저장되어 `pandas.indexes.*` 모듈 경로를 참조한다.
    pandas 2.x 에는 해당 모듈이 없으므로 현재 위치로 재매핑한다."""
    _MODULE_MAP = {
        'pandas.indexes': 'pandas.core.indexes',
        'pandas.core.internals.managers': 'pandas.core.internals.managers',
        'pandas.core.internals': 'pandas.core.internals',
    }
    _CLASS_MAP = {  # 제거된 인덱스 클래스 → Index
        'Int64Index': 'Index', 'UInt64Index': 'Index', 'Float64Index': 'Index',
    }

    def find_class(self, module, name):
        for old, new in self._MODULE_MAP.items():
            if module == old or module.startswith(old + '.'):
                module = new + module[len(old):]
                break
        if module.startswith('pandas.core.indexes') and name in self._CLASS_MAP:
            module, name = 'pandas.core.indexes.base', self._CLASS_MAP[name]
        # pandas 자체 호환 테이블(옛 경로 → 새 경로) 우선 적용
        from pandas.compat import pickle_compat
        module, name = getattr(pickle_compat, '_class_locations_map', {}).get((module, name), (module, name))
        return super().find_class(module, name)


def load_lswmd(path):
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as e:
        print(f'  pd.read_pickle 실패 ({e}) → 레거시 pandas 호환 Unpickler 로 재시도')
        with open(path, 'rb') as f:
            return _LegacyPandasUnpickler(f, encoding='latin1').load()  # Python2 pickle 문자열 호환


def main():
    if not DATA_PATH.exists():
        sys.exit(f'원본 없음: {DATA_PATH}')
    t0 = time.time()
    print(f'pickle 로드: {DATA_PATH} ({DATA_PATH.stat().st_size/1024**2:.0f} MB)')
    df = load_lswmd(DATA_PATH)
    print(f'  전체 {len(df):,}행  ({time.time()-t0:.0f}s)')

    df['failureType_clean'] = df['failureType'].apply(extract_label)
    labeled = df[df['failureType_clean'] != 'unknown'].reset_index(drop=True)
    print(f'레이블 샘플: {len(labeled):,}개')
    dist = labeled['failureType_clean'].value_counts()
    for c in CLASS_ORDER:
        print(f'  {c:<10} {dist.get(c, 0):>7,}')

    n = len(labeled)
    maps = labeled['waferMap'].values
    encoded = np.array([LABEL_MAP[l] for l in labeled['failureType_clean'].values])
    # lotName 도 함께 저장 (향후 GroupShuffleSplit 누수 점검용)
    lot_names = labeled['lotName'].astype(str).values

    print(f'\n{n:,}개 맵 → 64×64 uint8 리사이즈 (INTER_NEAREST)')
    all_maps_resized = np.empty((n, 64, 64), dtype=np.uint8)
    t0 = time.time()
    for i, w in enumerate(maps):
        all_maps_resized[i] = cv2.resize(np.asarray(w).astype(np.uint8), RESIZE,
                                         interpolation=cv2.INTER_NEAREST)
        if (i + 1) % 40000 == 0:
            print(f'  {i+1:>7,}/{n:,}  {time.time()-t0:.0f}s')
    print(f'  완료 {time.time()-t0:.0f}s  고유 픽셀값: {np.unique(all_maps_resized)}')

    X = np.arange(n); y = encoded
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(sss1.split(X, y))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel, test_rel = next(sss2.split(temp_idx, y[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]
    print(f'\nTrain {len(train_idx):,} / Val {len(val_idx):,} / Test {len(test_idx):,}')
    print(f"{'class':<10}{'train':>8}{'val':>8}{'test':>8}")
    for ci, c in enumerate(CLASS_ORDER):
        print(f'{c:<10}{(y[train_idx]==ci).sum():>8}{(y[val_idx]==ci).sum():>8}{(y[test_idx]==ci).sum():>8}')

    class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y[train_idx])

    split = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
             'encoded_labels': encoded, 'label_map': LABEL_MAP, 'class_order': CLASS_ORDER,
             'class_weights': class_weights,
             'split_ratio': {'train': 0.70, 'val': 0.15, 'test': 0.15}, 'seed': SEED,
             'lot_names': lot_names}
    with open(PROCESSED_DIR / 'split_indices.pkl', 'wb') as f:
        pickle.dump(split, f)
    np.save(PROCESSED_DIR / 'class_weights.npy', class_weights)
    np.save(PROCESSED_DIR / 'all_maps_resized.npy', all_maps_resized)
    print(f'\n저장 완료 → {PROCESSED_DIR}')
    print(f'  all_maps_resized.npy  {all_maps_resized.nbytes/1024**2:.0f} MB')
    print(f'  split_indices.pkl / class_weights.npy')


if __name__ == '__main__':
    main()
