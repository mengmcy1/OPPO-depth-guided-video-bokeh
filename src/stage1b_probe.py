"""Stage1b numeric adapter intervention; Stage1a implementation stays untouched."""
import torch
from torch import nn
import torch.nn.functional as F
from src.bokeh_generation_probe import VideoGenerationProbe


class StrongNumericProbe(VideoGenerationProbe):
    def __init__(self,seed=20261029):
        super().__init__('Depth',seed)
        torch.manual_seed(seed+1)
        modules=[]
        for a,b in zip([2,128,224,128],[128,224,128,32]):
            modules.extend([nn.Conv2d(a,b,1),nn.SiLU()])
        self.proj=nn.Sequential(*modules)
        self.interaction=nn.Sequential(nn.Conv2d(128,32,1),nn.SiLU())

    def forward(self,rgb,mask,strength,condition=None,zero_condition=False):
        n,t,_,h,w=rgb.shape
        x=rgb.flatten(0,1);mf=mask[:,None].expand(-1,t,-1,-1,-1).flatten(0,1)
        k=(strength.float()/32)[:,None,None,None,None].expand(n,t,1,h,w).flatten(0,1)
        e1=self.enc1(torch.cat([x,mf,k],1));e2=self.enc2(F.avg_pool2d(e1,2));e3=self.enc3(F.avg_pool2d(e2,2))
        if zero_condition:c=e3.new_zeros((n*t,32,*e3.shape[-2:]))
        else:
            f=self.proj(condition.flatten(0,1).float())
            q=(f*mf).sum((-1,-2),keepdim=True)/mf.sum((-1,-2),keepdim=True).clamp_min(1e-8)
            q=q.expand_as(f);c=self.interaction(torch.cat([f,q,f-q,f*q],1))
            c=F.interpolate(c,size=e3.shape[-2:],mode='bilinear',align_corners=False)
        z=F.silu(self.fusion(torch.cat([e3,c],1)))
        z=z.reshape(n,t,64,*z.shape[-2:]).permute(0,2,1,3,4)
        z=z+self.temporal(z);z=z.permute(0,2,1,3,4).flatten(0,1)
        z=self.dec2(torch.cat([F.interpolate(z,size=e2.shape[-2:],mode='bilinear',align_corners=False),e2],1))
        z=self.dec1(torch.cat([F.interpolate(z,size=e1.shape[-2:],mode='bilinear',align_corners=False),e1],1))
        delta=self.output(z).float()
        return (x.float()+k.float()*delta).clamp(0,1).reshape(n,t,3,h,w)


def make_model(mode,seed=20261029):
    if mode.endswith('Strong'):return StrongNumericProbe(seed)
    return VideoGenerationProbe('RGB' if mode=='RGB' else 'Tokens' if mode in ['DINO','VGGT'] else 'Oracle',seed)
