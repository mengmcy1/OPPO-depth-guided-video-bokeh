"""Load the pinned BokehMe public demo functions without running its top-level CLI."""
import argparse
import ast
import sys

import numpy as np
import torch
import torch.nn.functional as F


def demo_functions(repository):
    source = repository / 'demo.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('gaussian_blur', 'pipeline'):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == 'parser' and fn.attr == 'add_argument':
                nodes.append(node)
    namespace = dict(argparse=argparse, np=np, torch=torch, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['pipeline'], namespace['parser'].parse_args([])


def setup(repository, checkpoints, protocol):
    import cupy
    if not hasattr(cupy, 'int'):
        cupy.int = np.int32
    if not hasattr(cupy.cuda, 'compile_with_cache'):
        cupy.cuda.compile_with_cache = lambda source: cupy.RawModule(code=source)
    sys.path.insert(0, str(repository))
    from neural_renderer import ARNet, IUNet
    from classical_renderer.scatter import ModuleRenderScatter
    pipeline, args = demo_functions(repository)
    args.gamma = protocol['gamma']
    args.highlight = protocol['highlight']
    args.defocus_scale = protocol['defocus_scale']
    args.save_intermediate = False
    assert not args.highlight and args.gamma_min == 1 and args.gamma_max == 5
    nets = {'classical': ModuleRenderScatter().cuda().eval()}
    counts = {}
    for name, cls in [('arnet', ARNet), ('iunet', IUNet)]:
        # The demo constructor defaults are taken from the pinned CLI, not redefined.
        kwargs = {k.removeprefix(name+'_'): v for k, v in vars(args).items()
                  if k.startswith(name+'_') and not k.endswith('checkpoint_path')}
        net = cls(**kwargs)
        state = torch.load(checkpoints/f'{name}.pth', map_location='cpu', weights_only=False)['model']
        net.load_state_dict(state, strict=True)
        nets[name] = net.cuda().eval()
        counts[name] = sum(p.numel() for p in net.parameters())
    return pipeline, args, nets, counts


def defocus_input(disparity, focus, K, candidate, scale):
    lo, hi = float(disparity.min()), float(disparity.max())
    span = hi-lo
    assert span > 0, 'Constant disparity requires an explicit convention; do not silently normalize'
    shared = K*(disparity-focus)
    if candidate == 'shared_defocus':
        mapped_focus, signed = focus, shared
    elif candidate == 'normalized':
        mapped_focus = (focus-lo)/span
        signed = K*((disparity-lo)/span-mapped_focus)
        assert np.allclose(signed, shared/span, atol=1e-5, rtol=1e-5)
    else:
        raise ValueError(candidate)
    assert np.isfinite(signed).all()
    # Both mappings preserve zero and sign; this identity also catches a reversed focus convention.
    assert np.allclose(signed*(span if candidate == 'normalized' else 1), shared, atol=1e-5, rtol=1e-5)
    stats = dict(disparity_range=[lo,hi], span=span, original_focus=focus, mapped_focus=mapped_focus,
                 signed_defocus_range=[float(signed.min()),float(signed.max())],
                 pipeline_defocus_range=[float(signed.min()/scale),float(signed.max()/scale)])
    return signed/scale, stats


@torch.no_grad()
def predict(pipeline, args, nets, rgb, defocus):
    image = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).cuda()
    control = torch.from_numpy(defocus).unsqueeze(0).unsqueeze(0).cuda()
    values = pipeline(nets['classical'], nets['arnet'], nets['iunet'], image, control, args.gamma, args)
    assert len(values) == 4
    assert all(torch.isfinite(x).all() for x in values)
    assert all(x.shape == image.shape for x in values[:3])
    assert values[3].shape == (1,1,*rgb.shape[:2])
    output = values[0][0].permute(1,2,0).cpu().numpy()
    assert 0 <= output.min() <= output.max() <= 1
    stats = {name: [float(value.min()),float(value.max())]
             for name,value in zip(('prediction','classical','neural','error_map'),values)}
    return output, stats
