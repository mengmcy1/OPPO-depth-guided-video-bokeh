"""Independent native-resolution E-L geometry fixer."""
import math

import torch
from torch import nn
from torch.nn import functional as F

ORDER_RATIO = .985
RESIDUAL_LIMIT = 1.5
SCALE_FLOOR = .02
LOG_SQRT_2PI = .5*math.log(2*math.pi)


def rear_base(rho1, rho2, prior, confidence, margin):
    """Use the first qualifying multiview source, DepthSplat then triangulation."""
    log_front = rho1.log()
    valid = (confidence[:,1:] > 0)&torch.isfinite(prior[:,1:])&(prior[:,1:] > 0)
    log_prior = prior[:,1:].clamp_min(1e-8).log()
    accept = valid & (log_prior <= log_front+math.log(ORDER_RATIO)-margin)
    source = torch.where(accept[:,0:1],1,torch.where(accept[:,1:2],2,0))
    beta = torch.where(source==1,log_prior[:,0:1],torch.where(source==2,log_prior[:,1:2],rho2.log()))
    return beta, source, valid.any(1,keepdim=True)


def sobel(value):
    x = value.new_tensor([[-1.,0,1],[-2,0,2],[-1,0,1]])/8
    filters = torch.stack((x,x.T))[:,None]
    return F.conv2d(F.pad(value,(1,1,1,1),mode="replicate"),filters)


def fixed_channels(rgb, rho1, rho2, prior, confidence):
    log_front = rho1.log()
    delta = torch.cat([(prior[:,a:a+1]-prior[:,b:b+1]).abs()
                       * ((confidence[:,a:a+1]>0)&(confidence[:,b:b+1]>0))
                       for a,b in ((0,1),(0,2),(1,2))],1)
    dxy = sobel(log_front)
    luminance = (rgb*rgb.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
    structural = (sobel(luminance).norm(dim=1,keepdim=True)>=.04) | (dxy.norm(dim=1,keepdim=True)>=.03)
    structural |= delta.amax(1,keepdim=True)/prior.amax(1,keepdim=True).clamp_min(1e-8) > .10
    mask = F.max_pool2d(structural.float(),7,1,3)
    data = torch.cat((rgb,log_front,dxy,prior,confidence,delta,rho2.log()),1)
    return data, mask


def gaussian_log_density(value, mu, scale):
    return -.5*((value-mu)/scale).square()-scale.log()-LOG_SQRT_2PI


def log_mixture(value, means, scales, logit):
    mass = torch.cat((F.logsigmoid(logit),F.logsigmoid(-logit)),1)
    return torch.logsumexp(mass+gaussian_log_density(value,means,scales),1,keepdim=True)


def modes(means, scales, logit):
    scores = [log_mixture(means[:,i:i+1],means,scales,logit) for i in (0,1)]
    use_back = scores[1]>scores[0]
    return torch.where(use_back, means[:,1:2],means[:,0:1]).exp(), use_back


def block(widths, stride=1):
    layers=[]
    for i,(a,b) in enumerate(zip(widths,widths[1:])):
        layers.extend((nn.Conv2d(a,b,3,padding=1,stride=stride if i==0 else 1),nn.ReLU(inplace=True)))
    return nn.Sequential(*layers)


class ELight(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Conv2d(64,16,1)
        self.enc1 = block((32,32,32))
        self.enc2 = block((32,48,48),2)
        self.bottleneck = block((48,64,64,64),2)
        self.dec2 = block((112,48,48))
        self.dec1 = block((80,32,32))
        self.mixture = nn.Conv2d(32,5,1)
        self.gate = nn.Conv2d(33,1,1)
        self.segment = nn.Conv2d(32,3,1)
        nn.init.zeros_(self.mixture.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        with torch.no_grad():
            self.mixture.bias.copy_(torch.tensor([0.,0.,-2.5,-2.5,2.]))

    def forward(self, features, phi, mask, rho1, beta, attenuation):
        e1 = self.enc1(torch.cat((features,self.phi(phi)),1))
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        def up(x, other):
            return F.interpolate(x, size=other.shape[-2:],mode="bilinear",align_corners=False)
        d2 = self.dec2(torch.cat((up(b,e2),e2),1))
        d1 = self.dec1(torch.cat((up(d2,e1),e1),1))
        h = self.mixture(d1).float()
        gate = self.gate(torch.cat((d1,mask),1)).float().sigmoid()
        raw_tanh = h[:,:2].tanh()
        delta = RESIDUAL_LIMIT*attenuation*gate*raw_tanh
        front, back = rho1.log()+delta[:,0:1], beta+delta[:,1:2]
        order = front+math.log(ORDER_RATIO)
        means = torch.cat((front,torch.minimum(back,order)),1)
        scales = SCALE_FLOOR+F.softplus(h[:,2:4])
        decoded,is_back = modes(means,scales,h[:,4:5])
        decoded = torch.where(gate==0,rho1,decoded)
        return dict(means=means, scales=scales, logit=h[:,4:5], gate=gate,
                    decoded=decoded, selected_back=is_back & (gate>0), raw_tanh=raw_tanh,
                    order_gap=back-order, segmentation=self.segment(d1))
