"""Local-weight BokehDiff public inference with externally supplied Track A controls."""
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import torch


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
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.pipeline = p
        self.output_audit = {}
        original = p.image_processor.postprocess

        def audited_postprocess(image, *args, **kwargs):
            assert torch.isfinite(image).all(), 'Non-finite decoded RGB before public postprocessing'
            self.output_audit = dict(decoded_range=[float(image.min()), float(image.max())])
            result = original(image, *args, **kwargs)
            assert isinstance(result, torch.Tensor) and torch.isfinite(result).all()
            self.output_audit.update(output_shape=list(result.shape), output_range=[float(result.min()), float(result.max())])
            return result

        p.image_processor.postprocess = audited_postprocess

    @torch.inference_mode()
    def predict(self, rgb, disparity, focus, K, seed):
        assert rgb.shape == (576, 1024, 3) and disparity.shape == (576, 1024)
        assert np.isfinite(rgb).all() and np.isfinite(disparity).all()
        image = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda() * 2 - 1
        d = torch.from_numpy(disparity)[None, None].cuda()
        # Same signed difference and K/10 scale as the public inference; choose only the supplied focus.
        control = torch.cat([d, (d - focus).abs() * (K / 10)], dim=1)
        assert torch.isfinite(control).all()
        # Public PISA uses global torch.rand in addition to the pipeline generator.
        torch.manual_seed(seed)
        output, _ = self.module.log_validation(self.pipeline, seed, 499,
            SimpleNamespace(device=torch.device('cuda:0')), aif_image=image, disp_coc=control)
        audit = dict(self.output_audit, disparity_range=[float(d.min()), float(d.max())],
                     defocus_condition_range=[float(control[:, 1].min()), float(control[:, 1].max())],
                     disparity_equal_one_fraction=float((d == 1).float().mean()), focus=focus, K=K, seed=seed)
        assert output.size == (1024, 576) and output.mode == 'RGB'
        return output, audit
