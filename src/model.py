"""WM-811K 웨이퍼 불량 분류 모델 모음 (Phase 1)"""

import torch
import torch.nn as nn

NUM_CLASSES = 9


class WaferCNN(nn.Module):
    """커스텀 CNN 베이스라인 — 4 Conv Block + Global Avg Pool + FC Head.

    입력: (B, 1, 64, 64) float32
    출력: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        def _block(in_ch, out_ch, drop=0.1):
            return nn.Sequential(
                nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(drop),
            )

        self.block1 = _block(1,   32,  drop=0.10)
        self.block2 = _block(32,  64,  drop=0.10)
        self.block3 = _block(64,  128, drop=0.15)
        self.block4 = _block(128, 256, drop=0.20)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_wafer_cnn(checkpoint_path: str, device=None) -> WaferCNN:
    """체크포인트에서 WaferCNN 로드."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get('config', {'num_classes': NUM_CLASSES, 'dropout': 0.3})
    model = WaferCNN(**cfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model
