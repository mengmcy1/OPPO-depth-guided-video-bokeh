"""First pilot residual geometry student; RGB + actual disparity only."""
import torch
from torch import nn
from torch.nn import functional as F


class GeometryResidualUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())
        self.e1=block(4,32);self.e2=block(32,64);self.mid=block(64,128)
        self.d2=block(192,64);self.d1=block(96,32);self.out=nn.Conv2d(32,1,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x):
        assert x.ndim==4 and x.shape[1]==4
        a=self.e1(x);b=self.e2(F.avg_pool2d(a,2));c=self.mid(F.avg_pool2d(b,2))
        c=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        c=self.d1(torch.cat([F.interpolate(c,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.out(c)


def predict(model,x):
    """Pre-clamp geometry; residual adds to the actual fourth input channel."""
    return x[:,3:4].float()+model(x).float()
