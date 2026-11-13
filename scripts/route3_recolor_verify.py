"""Direct native-renderer checks for the fixed-geometry color statistics."""
import json,sys,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from gsplat import rasterization
from src.aperture_renderer import GaussianScene
from src.recolor_statistics import delta_color_statistics,surface_statistics
from src.local_gaussian_replacement import projection,ray_query_plan

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';subprocess.run(['nvidia-smi'],check=True)
    K=torch.tensor([[50.,0,8],[0,50.,8],[0,0,1]],device='cuda');E=torch.eye(4,device='cuda')
    s=GaussianScene(torch.tensor([[0.,0,1],[.04,0,1.5],[0,0,3]],device='cuda'),torch.tensor([[1.,0,0,0]]*3,device='cuda'),torch.tensor([[.1,.1,.01]]*3,device='cuda'),torch.tensor([.6,.5,.3],device='cuda'),torch.eye(3,device='cuda'),torch.tensor([0,1,1],device='cuda'))
    def native(colors):return rasterization(s.means,s.quats,s.scales,s.opacities,colors,E[None],K[None],16,16,packed=False,rasterize_mode='antialiased',eps2d=.0015,near_plane=1e-4,far_plane=1e5)
    base,alpha,_=native(s.colors);dc=torch.cat([s.colors.roll(1,0)-s.colors,torch.ones_like(s.colors)*.1],-1);pred,current,aa=delta_color_statistics(s,K,E,dc,16,16,True);errors=[]
    for i in range(2):
        changed,_,_=native(s.colors+dc[:,i*3:i*3+3]);errors.append(float((base[0]+pred[...,i*3:i*3+3]-changed[0]).abs().max()))
    assert max(errors)<2e-5,errors
    uv=torch.tensor([[8.5,8.5]],device='cuda');gmass=torch.zeros(3,device='cuda')
    stat=surface_statistics(s,projection(s,K,width=16,height=16),ray_query_plan(uv,16,16),torch.tensor([2],device='cuda',dtype=torch.int16),torch.zeros((1,3),device='cuda'),torch.ones(1,device='cuda')*.01,torch.ones(1,device='cuda'),torch.ones(1,device='cuda')*3,torch.tensor([1,1,2],device='cuda',dtype=torch.int16),torch.zeros((3,3),device='cuda'),torch.ones(3,device='cuda',dtype=torch.bool),torch.tensor([False,False,True],device='cuda'),gmass)
    assert stat[6,0]==3 and stat[7,0]==1 and stat[11,0]==1,stat
    assert gmass[2]>0 and gmass[:2].abs().max()==0,gmass
    # Native early termination excludes the sample that would cross1e-4.
    s=GaussianScene(torch.tensor([[0.,0,1],[0,0,2]],device='cuda'),torch.tensor([[1.,0,0,0]]*2,device='cuda'),torch.tensor([[10.,10.,.01]]*2,device='cuda'),torch.tensor([.999,.99],device='cuda'),torch.tensor([[.2,.4,.6],[1.,0.,0.]],device='cuda'),torch.tensor([0,1],device='cuda'))
    base,_,_=native(s.colors);dc=torch.cat([torch.tensor([[0.,0,0],[-1.,1,0]],device='cuda')]*2,-1);pred,_,_=delta_color_statistics(s,K,E,dc,16,16);changed,_,_=native(s.colors+dc[:,:3]);early=float((base[0]+pred[...,:3]-changed[0]).abs().max());assert early<2e-5,early
    result=dict(passed=True,delta_native_max_errors=errors,earlystop_delta_error=early,top8_count=float(stat[6,0]),top8_match_count=float(stat[7,0]),top8_close_depth_wrong_identity_count=float(stat[11,0]),matched_gaussian_mass=gmass.tolist())
    path=ROOT/'outputs/route3-recolor-2026-11-13-01/kernel-verification.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
