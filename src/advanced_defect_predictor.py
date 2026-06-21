import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class AdvancedDefectPredictor(nn.Module):
    """
    MobileNetV3 Small 백본 + 3-헤드 Multi-Task 출력
    Head1: 불량 분류 (num_defect_classes)
    Head2: 심각도 분류 (Critical/High/Medium/None = 4)
    Head3: 신뢰도 스코어 (0~1)
    """

    BACKBONE_DIM = 576
    SHARED_DIM   = 256

    def __init__(self, num_defect_classes=9, num_severity_classes=4, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        if pretrained:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        base.features[0][0] = new_conv

        self.backbone = base.features
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM, self.SHARED_DIM),
            nn.Hardswish(),
            nn.Dropout(0.3),
        )
        self.defect_head = nn.Sequential(
            nn.Linear(self.SHARED_DIM, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_defect_classes),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(self.SHARED_DIM, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_severity_classes),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(self.SHARED_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        feat   = self.backbone(x)
        feat   = self.gap(feat).flatten(1)
        shared = self.shared(feat)
        return {
            'defect':     self.defect_head(shared),
            'severity':   self.severity_head(shared),
            'confidence': self.confidence_head(shared),
        }

    @classmethod
    def from_checkpoint(cls, ckpt_path, device='cpu', **kwargs):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = cls(**kwargs)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()
        return model