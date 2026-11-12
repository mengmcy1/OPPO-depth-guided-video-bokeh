"""Light, observation-level candidate fusion. No attention or pretrained weights.

Candidate inputs: B x M(8) x19, plus six public query/lens conditions.
The two hidden surfels have separate opacity and content-confidence heads.
Only the caller's loss function accepts supervision; this module has no GT.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class LightCandidateFusion(nn.Module):
    def __init__(self,width=128,depth_bound=.5,color_bound=.05):
        super().__init__();self.depth_bound=depth_bound;self.color_bound=color_bound
        self.encoder=nn.Sequential(nn.Linear(19,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.condition=nn.Linear(6,width)
        self.head_embedding=nn.Parameter(torch.randn(2,width)*.01)
        # Two selection scores, two source-ray depth corrections, one q.
        self.geometry=nn.Linear(width,5)
        self.appearance=nn.Sequential(nn.Linear(width+3,width),nn.SiLU(),nn.Linear(width,4))
        with torch.no_grad():
            self.geometry.weight[2:4].zero_();self.geometry.bias[2:4].zero_()
            self.appearance[-1].weight.zero_();self.appearance[-1].bias.zero_();self.appearance[-1].bias[3]=-3.
        self.register_buffer('feature_scale',torch.tensor([5,5,5,5,1,1,1,16,4,4,math.radians(5),1,1,1,1,1,1,1,1],dtype=torch.float32))

    def normalize(self,features,image_width):
        x=features.clone();ratio=512/image_width;x[...,7]*=ratio
        x[...,8]=torch.where(x[...,16]>.5,torch.full_like(x[...,8],8),x[...,8]*ratio)
        x[...,9]=torch.where(x[...,17]>.5,torch.full_like(x[...,9],8),x[...,9]*ratio)
        return (x/self.feature_scale).clamp(-10,10)

    @staticmethod
    def source_colors(world,slots,context):
        """Bilinear samples grouped by source frame; never duplicate images/B.

        world Bx2x3; slots BxM. Each sampled color is from an original input
        view at the corrected fused point, not from a GT texture or label.
        """
        b,heads=world.shape[:2];m=slots.shape[1];poses=context['c2w'];R=poses[slots,:3,:3];t=poses[slots,:3,3]
        cam=torch.einsum('bhmi,bmij->bhmj',world[:,:,None]-t[:,None],R)
        uvz=cam@context['K'].T;uv=uvz[...,:2]/cam[...,2:].clamp_min(1e-8)
        h,w=context['images'].shape[-2:];inside=(cam[...,2]>1e-5)&(uv[...,0]>=0)&(uv[...,0]<w)&(uv[...,1]>=0)&(uv[...,1]<h)
        grid=uv*uv.new_tensor([2/w,2/h])-1;flat_grid=grid.reshape(-1,2);flat_slots=slots[:,None].expand(-1,heads,-1).reshape(-1);color=world.new_zeros((b*heads*m,3))
        for frame in range(len(poses)):
            idx=torch.nonzero(flat_slots==frame,as_tuple=True)[0]
            if not len(idx):continue
            value=F.grid_sample(context['images'][frame:frame+1],flat_grid[idx][None,None],mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
            color=color.index_copy(0,idx,value)
        return color.reshape(b,heads,m,3),inside

    def forward(self,features,valid,source_slots,source_xy,source_depth,condition,context,target_c2w):
        if features.shape[-1]!=19 or features.shape[1]!=8:raise ValueError('Expected Bx8x19 candidates')
        width=context['images'].shape[-1];encoded=self.encoder(self.normalize(features,width))+self.condition(condition)[:,None];g=self.geometry(encoded);qlogit=g[...,4]
        scores=g[...,:2].transpose(1,2)+F.logsigmoid(qlogit)[:,None]
        weights=torch.softmax(scores.masked_fill(~valid[:,None],-1e9),-1)*valid[:,None]
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        z=source_depth[:,None]*(1+self.depth_bound*g[...,2:4].transpose(1,2).tanh())
        rays=torch.cat([source_xy+.5,torch.ones_like(source_xy[...,:1])],-1)@torch.linalg.inv(context['K']).T
        pc=rays[:,None]*z[...,None];poses=context['c2w'];R=poses[source_slots,:3,:3];t=poses[source_slots,:3,3]
        candidate_world=torch.einsum('bhmi,bmji->bhmj',pc,R)+t[:,None];means=(candidate_world*weights[...,None]).sum(2)
        samples,inside=self.source_colors(means,source_slots,context);cw=weights*inside*valid[:,None];cw=cw/cw.sum(-1,keepdim=True).clamp_min(1e-8)
        fused=(samples*cw[...,None]).sum(2);variance=((samples-fused[:,:,None]).square()*cw[...,None]).sum(2)
        pooled=torch.einsum('bhm,bmc->bhc',weights,encoded)+self.head_embedding[None];appearance=self.appearance(torch.cat([pooled,variance],-1));colors=(fused+self.color_bound*appearance[...,:3].tanh()).clamp(0,1)
        # q guides normalized candidate selection, not the absolute opacity.
        opacity_logits=appearance[...,3];opacities=opacity_logits.sigmoid()*((inside&valid[:,None]).any(-1))
        camera=(means-target_c2w[:3,3])@target_c2w[:3,:3];candidate_camera=(candidate_world-target_c2w[:3,3])@target_c2w[:3,:3]
        return dict(means_world=means,means_camera=camera,colors=colors,opacities=opacities,opacity_logits=opacity_logits,weights=weights,selection_logits=scores.masked_fill(~valid[:,None],-1e9),q_logits=qlogit,depth=camera[...,2],candidate_camera=candidate_camera,color_variance=variance)
