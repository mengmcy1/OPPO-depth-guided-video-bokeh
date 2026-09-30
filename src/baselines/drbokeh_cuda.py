"""Build an isolated Dr.Bokeh extension with only its forward launch mapped to a 2-D grid."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]


def load_grid2d():
    repo=ROOT.parent/'DrBokeh/app/cuda-src'
    major,minor=torch.cuda.get_device_capability(0)
    # Preserve existing binaries; the same frozen kernel needs a build for this GPU.
    build=ROOT/'logs'/f'drbokeh-launch-grid2d-sm{major}{minor}';build.mkdir(exist_ok=True)
    cpp=(repo/'scatter_cuda.cpp').read_text()
    cuda=(repo/'scatter_cuda_kernel.cu').read_text().replace('x.type()','x.scalar_type()')
    old='''    const int threads = 512;
    const dim3 blocks((w + threads - 1) / threads, (h+threads-1)/threads, batch_size);'''
    new='''    const dim3 threads(16, 16);
    const dim3 blocks((w - ksize + 1 + threads.x - 1) / threads.x,
                      (h - ksize + 1 + threads.y - 1) / threads.y, batch_size);'''
    assert cuda.count(old)==4
    cuda=cuda.replace(old,new,1)
    setup='''from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
setup(name='scatter_cuda_grid2d',ext_modules=[CUDAExtension('scatter_cuda_grid2d',
    ['scatter_cuda.cpp','scatter_cuda_kernel.cu'],extra_compile_args={'cxx':['-O3'],'nvcc':['-O3']})],
    cmdclass={'build_ext':BuildExtension.with_options(use_ninja=False)})
'''
    for name,text in [('scatter_cuda.cpp',cpp),('scatter_cuda_kernel.cu',cuda),('setup_grid2d.py',setup)]:
        p=build/name
        if p.exists():assert p.read_text()==text
        else:p.write_text(text)
    binaries=list(build.glob('scatter_cuda_grid2d*.so'))
    if not binaries:
        env=dict(os.environ,TORCH_CUDA_ARCH_LIST=f'{major}.{minor}',MAX_JOBS='4')
        subprocess.run([sys.executable,'setup_grid2d.py','build_ext','--inplace'],cwd=build,env=env,check=True)
        binaries=list(build.glob('scatter_cuda_grid2d*.so'))
    assert len(binaries)==1
    spec=importlib.util.spec_from_file_location('scatter_cuda_grid2d',binaries[0])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
