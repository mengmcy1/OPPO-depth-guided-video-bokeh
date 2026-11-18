"""OVG-Net geometry stage. Camera convention is OpenCV axial metric depth."""
import torch
from torch import nn
from torch.nn import functional as F


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                         nn.GroupNorm(8, cout), nn.GELU())


def resize(x, shape):
    return F.interpolate(x, size=shape, mode='bilinear', align_corners=False)


def rays(intrinsics, shape, dtype, device):
    h,w = shape
    y,x = torch.meshgrid(torch.arange(h,device=device,dtype=dtype)+.5,
                         torch.arange(w,device=device,dtype=dtype)+.5,indexing='ij')
    p = torch.stack([x,y,torch.ones_like(x)],-1)
    return torch.einsum('bij,hwj->bhwi',torch.linalg.inv(intrinsics),p)


def project(xyz, pose, other, K, shape):
    world = xyz @ pose[:3,:3].T + pose[:3,3]
    camera = (world-other[:3,3]) @ other[:3,:3]
    uvz = camera @ K.T
    xy = uvz[...,:2] / uvz[...,2:].clamp_min(1e-5)
    grid = 2*xy/xy.new_tensor([shape[1],shape[0]]) - 1
    valid = (camera[...,2]>0) & (abs(grid)<1).all(-1)
    return grid, valid, camera[...,2]


def sample(image, grid, mode='bilinear'):
    result = F.grid_sample(image[None], grid.reshape(1,-1,1,2),mode=mode,
                           padding_mode='zeros',align_corners=False)
    return result[0,:,:,0].T.reshape(*grid.shape[:-1], image.shape[0])


def fusion_depth(priors, confidence):
    support = confidence.sum(1)
    inverse = torch.where(confidence>0, priors.clamp_min(.05).reciprocal(), 0.)
    base = (inverse*confidence).sum(1)/support.clamp_min(1e-8)
    return torch.where(support>0,base,.2), confidence.max(1).values


def select_prior(priors, confidence, logits):
    present=(confidence>0)&torch.isfinite(priors)&(priors>0)
    any_present=present.any(dim=1,keepdim=True)
    logits=torch.where(present,logits,-torch.inf)
    weights=torch.softmax(torch.where(any_present,logits,0.),dim=1)
    weights=torch.where(present,weights,0.)
    inverse=torch.where(present,priors.clamp_min(.05).reciprocal(),0.)
    base=torch.where(any_present[:,0],(weights*inverse).sum(1),.2)
    return base,weights


class OVGNet(nn.Module):
    def __init__(self, encoder, learned_prior_selection=False):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.rgb_half=conv(3,32,2)
        self.rgb_quarter=conv(32,48,2)
        self.depth_quarter=nn.Sequential(conv(7,32,2),conv(32,48,2))
        self.semantic=nn.ModuleList(nn.Conv2d(384,64,1) for _ in range(4))
        self.dpt=nn.ModuleList(conv(128,64) for _ in range(3))
        self.quarter=conv(64+48+48,64)
        self.half_decoder=conv(64+32,64)
        self.full=conv(64+3,64)
        self.shared_view=nn.Sequential(conv(3,32,2),conv(32,32,2))
        self.match=nn.Conv2d(64,32,1)
        self.refine=nn.Sequential(conv(73,64),conv(64,64))
        self.geometry=nn.Conv2d(64,6,1)
        self.coverage=nn.Conv2d(64,1,1)
        self.footprint=nn.Conv2d(64,3,1)
        self.geometry.weight.data.mul_(.05)
        self.geometry.bias.data.zero_()
        self.coverage.bias.data.fill_(2.2)
        self.learned_prior_selection=learned_prior_selection
        if learned_prior_selection:
            self.prior_head=nn.Sequential(conv(64,32),nn.Conv2d(32,3,1))
            self.prior_pixel=nn.Conv2d(10,3,1)

    def train(self,mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def tokens(self,rgb):
        srgb=torch.where(rgb<=.0031308,rgb*12.92,1.055*rgb.clamp_min(.0031308).pow(1/2.4)-.055)
        image=(resize(srgb.clamp(0,1),(252,252))-self.mean)/self.std
        with torch.no_grad():
            return self.encoder.get_intermediate_layers(image,n=[2,5,8,11],reshape=True)

    def local_cost(self, features, rgb, neighbors, d, confident, motion, K, Kn, pose, poses, p_neighbor, c_neighbor):
        b,_,h,w=features.shape
        H,W=rgb.shape[-2:]
        base=resize(d[:,None],(h,w))[:,0]
        conf=resize(confident[:,None],(h,w))[:,0]
        gradient=F.max_pool2d(base[:,None],3,1,1)+F.max_pool2d(-base[:,None],3,1,1)
        static=F.max_pool2d(motion[:,None].float(),(H//h,W//w))[:,0]<.5
        mask=((conf<.6)|(gradient[:,0]>.03*base))&static
        other=self.shared_view(neighbors.flatten(0,1)).reshape(b,neighbors.shape[1],32,h,w)
        # Appearance features are shared across views. The target semantic
        # projection conditions the descriptor without changing source pixels.
        target=F.normalize(self.shared_view(rgb)+self.match(features),dim=1)
        other=F.normalize(other,dim=2)
        short=K.clone()
        short[:,0]*=w/W
        short[:,1]*=h/H
        ray=rays(short,(h,w),rgb.dtype,rgb.device)
        deltas,all_costs=[],[]
        for i in range(b):
            pick=mask[i].nonzero()
            y,x=pick.T
            local=base[i,y,x]
            spread=(.1+.55*(1-conf[i,y,x]))*local
            bins=(local[:,None]+spread[:,None]*torch.linspace(-1,1,8,device=rgb.device)).clamp(.005,20)
            xyz=ray[i,y,x,None]/bins[...,None]
            desc=target[i,:,y,x].T
            total=torch.zeros_like(bins)
            count=torch.zeros_like(bins)
            for j in range(neighbors.shape[1]):
                grid,safe,z=project(xyz,pose[i],poses[i,j],Kn[i,j],(H,W))
                known=sample(c_neighbor[i,j,None],grid)[:,:,0]>.1
                observed=sample(p_neighbor[i,j,None],grid,mode='nearest')[:,:,0]
                safe=safe&(~known | (z<=observed*1.05))
                correlation=(sample(other[i,j],grid)*desc[:,None]).sum(-1)
                total=total+torch.where(safe,correlation,0)
                count=count+safe
            cost=total/count.clamp_min(1)
            prob=F.softmax(torch.where(count>0,8*cost,-50),dim=-1)
            delta=torch.where((count>0).any(-1),(prob*bins).sum(-1)-local,0.)
            costs=features.new_zeros(8,h,w)
            correction=features.new_zeros(h,w)
            costs[:,y,x]=cost.T
            correction[y,x]=delta
            deltas.append(correction);all_costs.append(costs)
        return torch.stack(deltas)[:,None],torch.stack(all_costs),mask

    def forward(self,sample):
        rgb=sample['rgb']
        priors,conf=sample['prior_depth'],sample['prior_conf']
        motion=sample['motion']
        half=self.rgb_half(rgb)
        quarter=self.rgb_quarter(half)
        h,w=quarter.shape[-2:]
        inv=torch.where(conf>0,priors.clamp_min(.05).reciprocal(),0.)
        evidence=torch.cat([inv.clamp(max=5),conf,motion[:,None].float()],1)
        z=self.depth_quarter(evidence)
        tokens=self.tokens(rgb)
        levels=[(max(1,h//8),max(1,w//8)),(max(1,h//4),max(1,w//4)),
                (max(1,h//2),max(1,w//2)),(h,w)]
        maps=[resize(m(t),size) for t,m,size in zip(reversed(tokens),self.semantic,levels)]
        plane=maps[0]
        for f,skip in zip(self.dpt,maps[1:]):
            plane=f(torch.cat([resize(plane,skip.shape[-2:]),skip],1))
        phi=self.quarter(torch.cat([plane,quarter,z],1))
        prior_weights=None
        if self.learned_prior_selection:
            logits=resize(self.prior_head(phi),rgb.shape[-2:])
            logits=logits+self.prior_pixel(torch.cat([rgb,evidence],1))
            fused,prior_weights=select_prior(priors,conf,logits)
            confidence=(prior_weights*conf).sum(1)
        else:
            fused,confidence=fusion_depth(priors,conf)
        if sample['neighbors'].shape[1]:
            delta, cost, mask=self.local_cost(phi,rgb,sample['neighbors'],fused,confidence,motion,
                                  sample['K'],sample['neighbor_K'],sample['pose'],sample['neighbor_pose'],
                                  sample['neighbor_depth'],sample['neighbor_conf'])
        else:
            delta=phi.new_zeros(len(phi),1,h,w)
            cost=phi.new_zeros(len(phi),8,h,w)
            mask=motion.new_zeros(len(phi),h,w)
        refined=self.refine(torch.cat([phi,cost,delta],1))
        half=self.half_decoder(torch.cat([resize(refined,half.shape[-2:]),half],1))
        phi=self.full(torch.cat([resize(half,rgb.shape[-2:]),rgb],1))
        g=self.geometry(phi)
        conf_out=g[:,5].sigmoid()
        gate=(F.interpolate(mask[:,None].float(),rgb.shape[-2:],mode='nearest')[:,0]>.5)&~motion
        correction=resize(delta,rgb.shape[-2:])[:,0]*gate*conf_out
        inverse1=((fused+correction).clamp(.005,20)*torch.exp(1.25*g[:,0].tanh())).clamp(.005,20)
        inverse2=inverse1*(.985-.94*g[:,1].sigmoid())
        front_normal=F.normalize(torch.cat([g[:,2:4],-F.softplus(g[:,4:5])],1),dim=1)
        footprint=self.footprint(phi)
        result=dict(depth1=inverse1.reciprocal(),depth2=inverse2.reciprocal(),inverse1=inverse1,inverse2=inverse2,
                    alpha=self.coverage(phi)[:,0].sigmoid(),normal=front_normal,confidence=conf_out,
                    sigma_px=.3+1.4*footprint[:,:2].sigmoid(),opacity=footprint[:,2].sigmoid(),
                    mv_mask=gate,fused_depth=fused.reciprocal())
        if prior_weights is not None:
            result['prior_weights']=prior_weights
        return result


def masked_mean(value,mask):
    return torch.where(mask,value,0.).sum()/mask.sum().clamp_min(1)


def scene_consistency(prediction, inputs):
    depth=prediction['depth1']
    B,H,W=depth.shape
    ray=rays(inputs['K'],(H,W),depth.dtype,depth.device)
    terms, counts=[],[]
    for i in range(B):
        # Quarter-grid photometric matches. Visibility/confidence thresholds
        # are computed from the observed input priors, detached from geometry.
        xyz = ray[i,::4,::4]*depth[i,::4,::4,None]
        ref=inputs['rgb'][i,:,::4,::4].permute(1,2,0)
        static=~inputs['motion'][i,::4,::4]
        for j in range(inputs['neighbors'].shape[1]):
            grid,in_view,z=project(xyz,inputs['pose'][i],inputs['neighbor_pose'][i,j],
                                   inputs['neighbor_K'][i,j],(H,W))
            observed=sample(inputs['neighbor_depth'][i,j,None],grid,'nearest')[...,0]
            confidence=sample(inputs['neighbor_conf'][i,j,None],grid)[...,0]
            valid=(in_view & static & (confidence>.15) &
                   ((z-observed).abs()<.05*z)).detach()
            rgb=sample(inputs['neighbors'][i,j],grid)
            err=((rgb-ref).square()+1e-4).sqrt().mean(-1)-.01
            terms.append(torch.where(valid,err,0.).sum())
            counts.append(valid.sum())
    if counts:
        n=torch.stack(counts).sum()
        return torch.stack(terms).sum()/n.clamp_min(1),n
    return depth.sum()*0,depth.new_zeros((),dtype=torch.int64)


def balanced_boundary_mean(value, valid, boundary, per_stratum, boundary_weight, generator=None):
    """Equal samples where both strata exist, plus explicit boundary weight."""
    v=value.flatten()
    edge=torch.nonzero((valid&boundary).flatten()).flatten()
    interior=torch.nonzero((valid&~boundary).flatten()).flatten()
    if not len(edge) and not len(interior):
        # Keep an exact zero gradient for tiers without this label.
        return v.sum()*0
    common=min(per_stratum,len(edge),len(interior)) if len(edge) and len(interior) else per_stratum
    chunks=[]
    for rows,weight in ((edge,boundary_weight),(interior,1.)):
        if len(rows)>common:
            rows=rows[torch.randperm(len(rows),device=rows.device,generator=generator)[:common]]
        if len(rows):
            chunks.append((v[rows]*weight).sum())
    return sum(chunks)/(min(len(edge),common)+min(len(interior),common))


def geometry_loss(prediction, inputs, target, weights, *, boundary_sampling=None, generator=None):
    valid=target['valid']
    mask_alpha=target['valid_alpha']
    mask_dual=target['valid_dual']&mask_alpha
    optics=inputs['K'][:,0,0,None,None]*.05/(2*1.4)
    front=target['depth1'].clamp_min(.05).reciprocal()
    back=target['depth2'].clamp_min(.05).reciprocal()
    e1=(prediction['inverse1']-front).abs()*optics
    boundary=target['depth_boundary']
    if boundary_sampling is None:
        disp=masked_mean(e1*(1+boundary.float()),valid)
    else:
        def reduce(values, mask):
            return balanced_boundary_mean(values,mask,boundary,*boundary_sampling,generator)
        disp=reduce(e1,valid)
    alpha=prediction['alpha'].clamp(1e-5,1-1e-5)
    # C/B geometry coverages supervise alpha. G and A are masked out by the
    # loader. A's later appearance loss instead consumes its matte directly.
    b1=alpha.log()
    b2=torch.log1p(-alpha)
    log_f=torch.stack([b1-e1/.15,b2-(prediction['inverse2']-front).abs()*optics/.15],-1)
    log_b=torch.stack([b1-(prediction['inverse1']-back).abs()*optics/.15,
                       b2-(prediction['inverse2']-back).abs()*optics/.15],-1)
    mass=target['alpha']
    nll=-(mass*torch.logsumexp(log_f,-1)+(1-mass)*torch.logsumexp(log_b,-1))
    alpha_map=F.binary_cross_entropy(alpha,mass,reduction='none')
    combine=masked_mean if boundary_sampling is None else reduce
    bimodal=combine(nll,mask_dual)
    alpha_loss=combine(alpha_map,mask_alpha)
    normal=masked_mean(1-(prediction['normal']*target['normal']).sum(1).clamp(-1,1),target['valid_normal'])
    mv,mv_count=scene_consistency(prediction,inputs)
    terms=dict(disp=disp,bimodal=bimodal,alpha=alpha_loss,normal=normal,mv=mv)
    total=sum(weights[k]*term for k,term in terms.items())
    return total,terms,dict(front=valid.sum(),dual=mask_dual.sum(),alpha=mask_alpha.sum(),
                            normal=target['valid_normal'].sum(),mv=mv_count)
