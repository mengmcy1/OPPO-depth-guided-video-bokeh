"""Local-weight BokehDiff public inference with externally supplied Track A controls."""
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import torch


def map_controls(disparity, focus, K, candidate):
    lo, hi = float(disparity.min()), float(disparity.max())
    span = hi - lo
    assert span > 0, 'Constant disparity: no silent fallback'
    if candidate == 'shared_geometry':
        mapped, mapped_focus, scale = disparity, focus, 1.0
    else:
        assert candidate == 'normalized'
        mapped, mapped_focus, scale = (disparity - lo) / span, (focus - lo) / span, 1.0 / span
    target = np.abs(disparity - focus) * K / 10 * scale
    error = float(np.abs(np.abs(mapped - mapped_focus) * K / 10 - target).max())
    assert error < 1e-10
    return mapped.astype(np.float32), mapped_focus, dict(candidate=candidate, min=lo, max=hi, span=span,
        input_focus=focus, mapped_focus=mapped_focus, K=K, CoC_condition_scale=scale,
        max_defocus_condition=float(target.max()), mapping_error_float64=error)


class OfficialBokehDiff:
    def __init__(self, repository, weights):
        sys.path.insert(0, str(repository))
        m = importlib.import_module('inference_hf')
        self.module = m
        base, adapter = weights / 'RealVisXL_V5.0', weights / 'adapter/small'
        options = dict(local_files_only=True)
        parts = {
            'tokenizer': m.CLIPTokenizer.from_pretrained(base, subfolder='tokenizer', **options),
            'tokenizer_2': m.CLIPTokenizer.from_pretrained(base, subfolder='tokenizer_2', **options),
            'scheduler': m.DDPMScheduler.from_pretrained(base, subfolder='scheduler', **options),
        }
        for key, cls in [('text_encoder', m.CLIPTextModel),
                         ('text_encoder_2', m.CLIPTextModelWithProjection),
                         ('vae', m.AutoencoderKL), ('unet', m.UNet2DConditionModel)]:
            parts[key] = cls.from_pretrained(base, subfolder=key, use_safetensors=True, **options).eval()
            parts[key].requires_grad_(False)
        parts['vae'].enable_tiling()
        p = m.TiledStableDiffusionXLPipeline(**parts, add_watermarker=False).to(dtype=torch.float32)
        p.vae.set_attn_processor(m.AttnProcessor2_0())
        p.unet.set_attn_processor(m.AttnProcessor2_0())
        p = p.to('cuda:0')
        p.set_progress_bar_config(disable=True)
        p.load_lora_weights(str(adapter), weight_name='pytorch_lora_weights.safetensors', local_files_only=True)
        state = p.vae.load_state_dict(torch.load(adapter / 'vae.ckpt', map_location='cpu', weights_only=True), strict=False)
        self.weight_loading = dict(vae_missing=list(state.missing_keys), vae_unexpected=list(state.unexpected_keys))
        m.fn_recursive_attn_processor('unet', p.unet, m.AttnProcessorDistReciprocal(
            hard=1e7, supersampling_num=5, segment_num=7, train=False))
        self.weight_loading['PISA_processors'] = [name for name, processor in p.unet.attn_processors.items()
            if isinstance(processor, m.AttnProcessorDistReciprocal)]
        self.weight_loading['LoRA_parameter_count'] = sum(parameter.numel() for name, parameter in p.unet.named_parameters() if 'lora_' in name)
        assert self.weight_loading['PISA_processors'] and self.weight_loading['LoRA_parameter_count'] > 0
        assert not state.unexpected_keys
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.pipeline = p
        self.output_audit = {}
        self.output_float = None
        original = p.image_processor.postprocess

        def audited_postprocess(image, *args, **kwargs):
            assert torch.isfinite(image).all(), 'Non-finite decoded RGB before public postprocessing'
            self.output_audit = dict(decoded_range=[float(image.min()), float(image.max())])
            result = original(image, *args, **kwargs)
            assert isinstance(result, torch.Tensor) and torch.isfinite(result).all()
            self.output_audit.update(output_shape=list(result.shape), output_range=[float(result.min()), float(result.max())])
            self.output_float = result[0].permute(1, 2, 0).detach().cpu().numpy().copy()
            return result

        p.image_processor.postprocess = audited_postprocess

    @torch.inference_mode()
    def predict(self, rgb, disparity, focus, K, seed, *, reset_global_seed=True, quantization='public_truncate'):
        assert rgb.shape == (576, 1024, 3) and disparity.shape == (576, 1024)
        assert np.isfinite(rgb).all() and np.isfinite(disparity).all()
        image = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda() * 2 - 1
        d = torch.from_numpy(disparity)[None, None].cuda()
        # Same signed difference and K/10 scale as the public inference; choose only the supplied focus.
        control = torch.cat([d, (d - focus).abs() * (K / 10)], dim=1)
        assert torch.isfinite(control).all()
        # The legacy capacity probe resets each frame. Recovery runs seed globally once,
        # while retaining the public function's separate per-call scheduler generator.
        if reset_global_seed:
            torch.manual_seed(seed)
        output, _ = self.module.log_validation(self.pipeline, seed, 499,
            SimpleNamespace(device=torch.device('cuda:0')), aif_image=image, disp_coc=control)
        if quantization == 'round_png8':
            from PIL import Image
            output = Image.fromarray(np.uint8(np.round(np.clip(self.output_float, 0, 1) * 255)))
        else:
            assert quantization == 'public_truncate'
        audit = dict(self.output_audit, disparity_range=[float(d.min()), float(d.max())],
                     defocus_condition_range=[float(control[:, 1].min()), float(control[:, 1].max())],
                     disparity_equal_one_fraction=float((d == 1).float().mean()), focus=focus, K=K, seed=seed,
                     reset_global_seed=reset_global_seed, quantization=quantization)
        assert output.size == (1024, 576) and output.mode == 'RGB'
        return output, audit
