"""Preserve MagicBokeh's native depth conditioning and 512px tiled generator."""
import importlib
import os
import sys
from types import SimpleNamespace


def configure_attention():
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
        from xformers.ops.fmha import _set_use_fa3
        _set_use_fa3(False)


class OfficialMagicBokeh:
    def __init__(self, repository, weights, base):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        sys.path.insert(0, str(repository))
        configure_attention()
        self.module = importlib.import_module("inference")
        self.model = self.module.model_test(SimpleNamespace(
            pretrained_model_name_or_path=str(base),
            model_path=str(weights / "lora.pkl"), mixed_precision="fp16"))
        # Preserve the author's per-module training/evaluation state.

    def prepare(self, image, quantized_depth, focus, K):
        """CPU decoding/control adaptation and transfer, outside model-level timing."""
        import torch
        lq = self.module.tensor_transforms(image).unsqueeze(0).cuda()
        depth = self.module.tensor_transforms(quantized_depth).unsqueeze(0).cuda()
        defocus = (depth - focus).abs() * (K / 32)
        images, positions = self.module.extract_patches(lq)
        controls, control_positions = self.module.extract_patches(defocus)
        if positions != control_positions:
            raise ValueError("RGB/condition tile mismatch")
        # These pointwise adapters are excluded just as other methods' controls are.
        focused = 1 - (controls > 0.1).float()
        return dict(images=images * 2 - 1, controls=controls * 2 - 1,
                    focused=focused * 2 - 1, positions=positions, shape=lq.shape)

    def forward_cuda(self, prepared, prompt=""):
        """All native tile forwards and author overlap assembly to CUDA float RGB."""
        import torch
        outputs = [self.model(image, control, focused, prompt) for image, control, focused in
                   zip(prepared["images"], prepared["controls"], prepared["focused"])]
        return self.module.reconstruct_from_patches(
            torch.stack(outputs), prepared["shape"], prepared["positions"])

    def to_pil(self, output, source):
        """Original CLI conversion and AdaIN; CPU postprocessing, outside timing."""
        from utils.wavelet_color_fix import adain_color_fix
        image = self.module.transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        return adain_color_fix(target=image, source=source)
