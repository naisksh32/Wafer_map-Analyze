import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAM:
    """AdvancedDefectPredictor defect head 기준 Grad-CAM"""

    def __init__(self, model, target_layer):
        self.model  = model
        self._acts  = None
        self._grads = None
        self._fwd   = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd   = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, inp, out): self._acts  = out.detach()
    def _bwd_hook(self, m, gi, go):  self._grads = go[0].detach()

    def generate(self, x, class_idx=None, device='cpu'):
        self.model.eval()
        x = x.unsqueeze(0).to(device).requires_grad_(True)
        out = self.model(x)['defect']
        if class_idx is None:
            class_idx = out.argmax(1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()

        w   = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._acts).sum(1, keepdim=True)).squeeze().cpu().numpy()
        return cam / (cam.max() + 1e-8), class_idx

    def remove(self):
        self._fwd.remove(); self._bwd.remove()


class SHAPExplainer:
    """shap.DeepExplainer 래퍼 — defect head 기준 픽셀 기여도"""

    def __init__(self, model, background: torch.Tensor):
        import shap

        class _Wrapper(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x)['defect']

        self.explainer = shap.DeepExplainer(_Wrapper(model), background)

    def explain(self, X: torch.Tensor):
        """Returns list[9] of (N,1,64,64) SHAP value arrays"""
        return self.explainer.shap_values(X)

    @staticmethod
    def plot(img_np, shap_map, class_name, save_path=None):
        """단일 샘플 SHAP 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("원본")
        vmax = np.abs(shap_map).max() + 1e-8
        axes[1].imshow(shap_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"SHAP ({class_name})")
        axes[2].imshow(img_np, cmap="gray", alpha=0.5)
        axes[2].imshow(np.clip(shap_map, 0, None), cmap="Reds", alpha=0.5)
        axes[2].set_title("Overlay")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.show()