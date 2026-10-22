"""Cost-only scratch CNN forward/backward probe. No optimizer step or trained weights."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]


class ProbeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())
        self.e1=block(4,32);self.e2=block(32,64);self.mid=block(64,128)
        self.d2=block(192,64);self.d1=block(96,32);self.out=nn.Conv2d(32,1,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x):
        a=self.e1(x);b=self.e2(F.avg_pool2d(a,2));c=self.mid(F.avg_pool2d(b,2))
        c=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        c=self.d1(torch.cat([F.interpolate(c,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return x[:,3:4]+self.out(c)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    torch.set_num_threads(4);torch.manual_seed(20261023)
    model=ProbeUNet().cuda().train()
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    parameter_count=sum(p.numel() for p in model.parameters());rows=[]
    for batch,h,w in [(4,384,384),(1,436,1024)]:
        # Cached synthetic batch isolates compute; does not estimate disk reads or convergence.
        cpu_x=torch.rand(batch,4,h,w,pin_memory=True)
        cpu_y=torch.rand(batch,1,h,w,pin_memory=True)
        cpu_mask=torch.ones(batch,1,h,w,dtype=torch.bool,pin_memory=True)
        times=[]
        for i in range(40):
            if i==10:torch.cuda.reset_peak_memory_stats()
            model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            x=cpu_x.cuda(non_blocking=True);y=cpu_y.cuda(non_blocking=True);mask=cpu_mask.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(x)
                # Geometry and keep arithmetic both evaluated; scratch target has no research meaning.
                loss=(out-y).abs()[mask].mean()+.1*F.relu((out-x[:,3:4]).abs()-.028637895)[mask].mean()
            loss.backward();torch.cuda.synchronize()
            if i>=10:times.append(time.perf_counter()-start)
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        row=dict(batch=batch,height=h,width=w,warmup=10,repeats=30,mean_seconds=float(np.mean(times)),
                 median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),
                 peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                 compute_minutes_1000_steps=float(np.mean(times)*1000/60),
                 compute_minutes_5000_steps=float(np.mean(times)*5000/60))
        rows.append(row);print(json.dumps(row),flush=True)
        del cpu_x,cpu_y,cpu_mask,x,y,mask,out,loss;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    result=dict(status='COST_PROBE_ONLY_NO_TRAINING',GPU=torch.cuda.get_device_name(0),torch=torch.__version__,
                model='candidate two-downsample UNet channels32/64/128, residual output; architecture not frozen',
                parameters=parameter_count,precision='BF16 autocast, float32 parameters',rows=rows,
                optimizer_updates=0,weights_unchanged=True,checkpoint_saved=False,
                timing_scope='pinned cached batch H2D + forward + simple geometry/keep loss + backward; no optimizer, data decoding, validation, saving or ATB',
                caveats=['Cold-cache data preparation and full experiment elapsed time are separate.',
                         'No learning curve or convergence-time claim; number of steps not chosen by this probe.',
                         'Real training may require nonzero residual initialization or optimization details; probe shows only candidate cost.'])
    p=ROOT/'03_实验记录/records/ours-student-cost-probe-20261023.json'
    with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PARAMETERS',parameter_count,flush=True)


if __name__=='__main__':main()
