"""Use public MPIB/BokehDiff kernels unchanged; not the missing Any-to-Bokeh implementation."""
import importlib.util
from pathlib import Path
import cupy
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]

def load_renderer(kind='MPIB'):
    if not hasattr(cupy,'int'):cupy.int=np.int32
    path=ROOT.parent/('MPIB/bokeh_generator.py' if kind=='MPIB' else 'BokehDiff/classical_renderer/mpi_multi_reverse.py')
    spec=importlib.util.spec_from_file_location('public_ray_'+kind,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ModuleRenderRT().cuda()

def render(module,images,alphas,depths,K,focus,samples=101,kind='MPIB'):
    # [batch, layers, RGB/alpha, H, W]; C stores axial depth for parallel planes.
    assert images.is_cuda and images.dtype==torch.float32 and images.is_contiguous()
    assert alphas.is_cuda and alphas.dtype==torch.float32 and alphas.is_contiguous()
    n,l=images.shape[:2]
    coeff=torch.zeros((n,l,3),device=images.device,dtype=torch.float32)
    coeff[:,:,2]=torch.as_tensor(depths,device=images.device,dtype=torch.float32).reshape(n,l)
    kval=torch.as_tensor(K,device=images.device,dtype=torch.float32).expand(n).reshape(n,1,1,1).contiguous()
    focal=torch.as_tensor(focus,device=images.device,dtype=torch.float32).expand(n).reshape(n,1,1,1).contiguous()
    with cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        if kind=='MPIB':out=module(images,alphas,coeff,kval,focal,samples)
        else:
            color,weight=module(images,alphas,coeff,kval,focal,0,samples)
            assert (weight>0).all(),'Public BokehDiff kernel returned zero ray weight'
            out=color[:,:3]/weight
    assert torch.isfinite(out).all()
    return out
