"""Small controlled RGB-video generator. Not ATB or the final Ours model."""
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1),nn.SiLU(),nn.Conv2d(cout,cout,3,padding=1),nn.SiLU())


class VideoGenerationProbe(nn.Module):
    def __init__(self,kind,seed=20261029):
        super().__init__();self.kind=kind
        # Shared trunk initialization independent of adapter parameter counts.
        torch.manual_seed(seed)
        self.enc1=Block(5,16);self.enc2=Block(16,32);self.enc3=Block(32,64)
        self.fusion=nn.Conv2d(96,64,3,padding=1)
        self.temporal=nn.Sequential(nn.Conv3d(64,64,3,padding=1),nn.SiLU(),nn.Conv3d(64,64,3,padding=1))
        self.dec2=Block(96,32);self.dec1=Block(48,16);self.output=nn.Conv2d(16,3,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        torch.manual_seed(seed+1)
        if kind=='Tokens':
            self.proj=nn.Conv2d(2048,32,1)
            self.interaction=nn.Sequential(nn.Conv2d(128,32,1),nn.SiLU())
        elif kind in ['Depth','Oracle']:
            self.proj=nn.Sequential(nn.Conv2d(2,32,1),nn.SiLU())
        else:assert kind=='RGB'

    def forward(self,rgb,mask,strength,condition=None,zero_condition=False):
        n,t,_,h,w=rgb.shape
        x=rgb.flatten(0,1);mf=mask[:,None].expand(-1,t,-1,-1,-1).flatten(0,1)
        k=(strength.float()/32)[:,None,None,None,None].expand(n,t,1,h,w).flatten(0,1)
        e1=self.enc1(torch.cat([x,mf,k],1));e2=self.enc2(F.avg_pool2d(e1,2));e3=self.enc3(F.avg_pool2d(e2,2))
        if self.kind=='RGB' or zero_condition:
            c=e3.new_zeros((n*t,32,*e3.shape[-2:]))
        elif self.kind=='Tokens':
            f=condition.flatten(0,1).float();f=F.normalize(f,dim=1)*2048**.5
            f=self.proj(f);m=F.interpolate(mf,size=f.shape[-2:],mode='area')
            q=(f*m).sum((-1,-2),keepdim=True)/m.sum((-1,-2),keepdim=True).clamp_min(1e-8)
            q=q.expand_as(f);c=self.interaction(torch.cat([f,q,f-q,f*q],1))
            c=F.interpolate(c,size=e3.shape[-2:],mode='bilinear',align_corners=False)
        else:
            c=self.proj(condition.flatten(0,1).float())
            c=F.interpolate(c,size=e3.shape[-2:],mode='bilinear',align_corners=False)
        z=F.silu(self.fusion(torch.cat([e3,c],1)))
        z=z.reshape(n,t,64,*z.shape[-2:]).permute(0,2,1,3,4)
        z=z+self.temporal(z);z=z.permute(0,2,1,3,4).flatten(0,1)
        z=self.dec2(torch.cat([F.interpolate(z,size=e2.shape[-2:],mode='bilinear',align_corners=False),e2],1))
        z=self.dec1(torch.cat([F.interpolate(z,size=e1.shape[-2:],mode='bilinear',align_corners=False),e1],1))
        delta=self.output(z).float()
        return (x.float()+k.float()*delta).clamp(0,1).reshape(n,t,3,h,w)
