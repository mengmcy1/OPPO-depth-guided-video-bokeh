"""Independent native-resolution E-L geometry fixer."""
import math

import torch
from torch import nn
from torch.nn import functional as F

ORDER_RATIO = .985
RESIDUAL_LIMIT = 1.5
SCALE_FLOOR = .02
LOG_SQRT_2PI = .5*math.log(2*math.pi)
REAR_MARGINS = (.902, .057)
FEATURE_MODES = ("E-L", "E-H1", "E-H2")
ENCODER_CHANNELS = 4 * 384
LOCAL_SIDE = 112


def rear_base(rho1, rho2, prior, confidence, margin=None):
    """Select a qualifying rear anchor without changing the front anchor."""
    log_front = rho1.log()
    valid = (confidence[:,1:] > 0)&torch.isfinite(prior[:,1:])&(prior[:,1:] > 0)
    log_prior = torch.where(valid, prior[:,1:], 1.).log()
    if margin is None:
        margins = prior.new_tensor(REAR_MARGINS)[None,:,None,None]
        accept = valid & (log_prior <= log_front+math.log(ORDER_RATIO)-margins)
        source = torch.where(accept[:,1:2],2,torch.where(accept[:,0:1],1,0))
    else:
        # The preparatory overfit checkpoint used a shared margin and DS priority.
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


def log1mexp(value):
    """Stable log(1-exp(value)) in either half of the negative domain."""
    split = -math.log(2)
    low = torch.log1p(-value.clamp_max(split).exp())
    high = torch.log(-torch.expm1(value.clamp_min(split)))
    return torch.where(value < split, low, high)


def gated_log_mass(front_logit, gate_logit):
    log_back = F.logsigmoid(gate_logit) + F.logsigmoid(-front_logit)
    log_back = log_back.clamp_max(-torch.finfo(log_back.dtype).tiny)
    return torch.cat((log1mexp(log_back), log_back), 1)


def mixture_density(value, means, scales, log_mass):
    return torch.logsumexp(log_mass+gaussian_log_density(value,means,scales),1,keepdim=True)


def log_mixture(value, means, scales, logit):
    mass = torch.cat((F.logsigmoid(logit),F.logsigmoid(-logit)),1)
    return mixture_density(value,means,scales,mass)


def modes(means, scales, logit):
    scores = [log_mixture(means[:,i:i+1],means,scales,logit) for i in (0,1)]
    use_back = scores[1]>scores[0]
    return torch.where(use_back, means[:,1:2],means[:,0:1]).exp(), use_back


def decode_gated(rho1, beta, attenuation, residual, scales, front_logit, gate_logit):
    gate = gate_logit.sigmoid()
    delta = RESIDUAL_LIMIT * attenuation * gate * residual
    front = rho1.log() + delta[:,0:1]
    back = beta + delta[:,1:2]
    order = front + math.log(ORDER_RATIO)
    means = torch.cat((front, torch.minimum(back,order)), 1)
    mass = gated_log_mass(front_logit, gate_logit)
    first, second = (mixture_density(means[:,i:i+1],means,scales,mass) for i in (0,1))
    selected = second > first
    # This is the front mean in linear inverse depth, exact when its residual is zero.
    decoded = torch.where(selected,means[:,1:2].exp(),rho1 * delta[:,0:1].exp())
    return dict(means=means,scales=scales,log_mass=mass,gate=gate,decoded=decoded,
                selected_back=selected,order_gap=back-order)


def feature_grid(origins, size, stride):
    height,width = size
    y,x = torch.meshgrid((torch.arange(height,device=origins.device)+.5)*stride,
                         (torch.arange(width,device=origins.device)+.5)*stride,indexing="ij")
    return torch.stack((x,y),-1)[None] + origins[:,None,None]


def local_context(project, grid, window_tokens):
    rows = []
    for space,(tokens,origins) in zip(grid,window_tokens):
        if not len(origins):
            rows.append(space.new_zeros((project.out_channels,*space.shape[:2])))
            continue
        local = (space[None]-origins[:,None,None])/LOCAL_SIDE
        inside = ((local>=0)&(local<1)).all(-1)[:,None]
        feature = project(tokens)
        sampled = F.grid_sample(feature.float(),2*local-1,align_corners=False,padding_mode="border")
        rows.append((sampled*inside).sum(0)/inside.sum(0).clamp_min(1))
    return torch.stack(rows)


def block(widths, stride=1):
    layers=[]
    for i,(a,b) in enumerate(zip(widths,widths[1:])):
        layers.extend((nn.Conv2d(a,b,3,padding=1,stride=stride if i==0 else 1),nn.ReLU(inplace=True)))
    return nn.Sequential(*layers)


class ELight(nn.Module):
    def __init__(self, feature_mode="E-L", *, legacy_mixture=False):
        super().__init__()
        if feature_mode not in FEATURE_MODES:
            raise ValueError(feature_mode)
        self.feature_mode = feature_mode
        self.legacy_mixture = legacy_mixture
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
        if feature_mode != "E-L":
            self.global_projection = nn.ModuleList(nn.Conv2d(ENCODER_CHANNELS,16,1) for _ in range(2))
            channels = 16
            if feature_mode == "E-H2":
                self.local_projection = nn.ModuleList(nn.Conv2d(ENCODER_CHANNELS,16,1) for _ in range(2))
                channels += 16
            self.context_fusion = nn.ModuleList(nn.Conv2d(width+channels,width,1) for width in (64,112))
            with torch.no_grad():
                for blend in self.context_fusion:
                    width = blend.out_channels
                    blend.bias.zero_()
                    blend.weight[:,:width,0,0].copy_(torch.eye(width))

    def context(self, value, mask, context, level):
        tokens, origins, windows = context
        grid = feature_grid(origins,value.shape[-2:],(4,2)[level])
        projected = self.global_projection[level](tokens)
        features = [value,F.grid_sample(projected.float(),grid/256.-1.,align_corners=False,padding_mode="border")]
        if self.feature_mode == "E-H2":
            local = local_context(self.local_projection[level],grid,windows)
            active = F.interpolate(mask,size=value.shape[-2:],mode="nearest-exact")
            features.append(local*active)
        return self.context_fusion[level](torch.cat(features,1))

    def forward(self, features, phi, mask, rho1, beta, attenuation, context=None):
        e1 = self.enc1(torch.cat((features,self.phi(phi)),1))
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        def up(x, other):
            return F.interpolate(x, size=other.shape[-2:],mode="bilinear",align_corners=False)
        if self.feature_mode != "E-L":
            b = self.context(b,mask,context,0)
        d2_input = torch.cat((up(b,e2),e2),1)
        if self.feature_mode != "E-L":
            d2_input = self.context(d2_input,mask,context,1)
        d2 = self.dec2(d2_input)
        d1 = self.dec1(torch.cat((up(d2,e1),e1),1))
        h = self.mixture(d1).float()
        gate_logit = self.gate(torch.cat((d1,mask),1)).float()
        raw_tanh = h[:,:2].tanh()
        scales = SCALE_FLOOR+F.softplus(h[:,2:4])
        if self.legacy_mixture:
            gate = gate_logit.sigmoid()
            delta = RESIDUAL_LIMIT*attenuation*gate*raw_tanh
            front,back = rho1.log()+delta[:,0:1],beta+delta[:,1:2]
            order = front+math.log(ORDER_RATIO)
            means = torch.cat((front,torch.minimum(back,order)),1)
            decoded,selected = modes(means,scales,h[:,4:5])
            output = dict(means=means,scales=scales,gate=gate,decoded=decoded,
                          selected_back=selected,order_gap=back-order)
        else:
            output = decode_gated(rho1,beta,attenuation,raw_tanh,scales,h[:,4:5],gate_logit)
        return dict(output,logit=h[:,4:5],raw_tanh=raw_tanh,segmentation=self.segment(d1),d1=d1)
