"""Official Dr.Bokeh forward pipeline with read-only input/intermediate instrumentation."""
import importlib.util
import os
import sys
import time

import cv2
import numpy as np
import torch


def array_stats(x):
    assert isinstance(x,np.ndarray) and np.isfinite(x).all()
    return dict(shape=list(x.shape),dtype=str(x.dtype),min=float(x.min()),max=float(x.max()))


def map_geometry(d,focus,K):
    lo,hi=float(d.min()),float(d.max());span=hi-lo
    assert span>0, 'Constant disparity: no fallback'
    dn=(d-lo)/span;fn=(focus-lo)/span;kn=K*span
    expected=K*(d-focus)
    error=float(np.abs(kn*(dn-fn)-expected).max())
    fp32=np.float32(kn)*(dn.astype(np.float32)-np.float32(fn))
    return dn,fn,kn,dict(min=lo,max=hi,span=span,focus=focus,focus_prime=fn,K_manifest=K,K_Dr=kn,
                          signed_CoC_error_float64=error,signed_CoC_error_float32=float(np.abs(fp32-expected).max()),
                          nominal_input_radius_max=float(np.abs(expected).max()))


def tail_mass(radius,lens):
    # Exact public radial softstep/area term on a sufficiently large lattice.
    assert 0<=radius<100
    grid=np.arange(-160,161,dtype=np.float64)
    rho=np.hypot(grid[:,None],grid[None,:])
    weight=.1/(.1+np.exp(np.clip(-3*((radius+1)-rho),-700,700))+1e-6)
    return float(weight[rho>=lens//2].sum()/weight.sum())


class OfficialDrBokeh:
    def __init__(self,repository,protocol):
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        os.chdir(repository)
        sys.path.insert(0,str(repository))
        spec=importlib.util.spec_from_file_location('official_drbokeh_inference',repository/'app/Render/Inference.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        import image_preprocessing as prep
        self.prep=prep;self.protocol=protocol
        self.model=self.module.DrBokeh(protocol['lens'],highlight=protocol['highlight'])
        # Depth_Inference changes cudnn.benchmark while importing; restore declared mode AFTER loading.
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        self.row={};self.arrays={};self.expected_mask=None
        for name,net in [('LDF',prep.salient_inferencer.net),('LaMa',prep.rgb_inpainting_inferencer.model),('MiDaS',prep.depth_inferencer.model)]:
            net.register_forward_hook(self._hook(name))
        self.original_layers=self.module.RGBAD2layers
        self.module.RGBAD2layers=self._layers
        for name in ('process_fg','process_bg','RGB_inpainting','depth_inpainting'):
            original=getattr(prep,name)
            setattr(prep,name,self._wrap(name,original))
        original=self.model.render_model.renderer.forward
        def capture_render(x,K):
            result=original(x,K)
            assert torch.isfinite(result).all()
            index=self.row['calls'].get('layer_renderer',0)
            self.row['calls']['layer_renderer']=index+1
            w=result[0,3].detach().cpu().numpy();visibility=result[0,4].detach().cpu().numpy()
            self.row.setdefault('render_normalization',[]).append(dict(weight_min=float(w.min()),weight_max=float(w.max()),
                weight_below_1e8_fraction=float((w<1e-8).mean()),visibility_range=[float(visibility.min()),float(visibility.max())]))
            self.arrays[f'layer_{index}_weight']=w
            self.arrays[f'layer_{index}_visibility']=visibility
            return result
        self.model.render_model.renderer.forward=capture_render

    def _hook(self,name):
        def record(*unused):
            self.row['calls'][name]=self.row['calls'].get(name,0)+1
        return record

    def _wrap(self,name,original):
        def wrapped(*args,**kwargs):
            self.row['calls'][name]=self.row['calls'].get(name,0)+1
            start=time.monotonic()
            result=original(*args,**kwargs)
            seconds=time.monotonic()-start
            assert result is not None, f'{name} failed'
            if name in ('RGB_inpainting','depth_inpainting'):
                self.row[name]=dict(input=array_stats(args[0]),mask=array_stats(args[1]),output=array_stats(result),
                                    masked_fraction=float((args[1]>0).mean()))
                self.arrays[name+'_output']=result.copy()
                self.arrays[name+'_mask']=args[1].copy()
            else:
                self.row[name]=array_stats(result)
            self.row[name]['seconds']=seconds
            return result
        return wrapped

    def _layers(self,rgb,alpha,disp,params):
        self.row['calls']['RGBAD2layers']=self.row['calls'].get('RGBAD2layers',0)+1
        assert np.array_equal(alpha,self.expected_mask), 'RGBAD2layers did not receive this frame LDF mask'
        layers=self.original_layers(rgb,alpha,disp,params)
        assert np.array_equal(layers['fg_rgbad'][...,3:4],alpha)
        assert np.all(layers['bg_rgbad'][...,3]==1)
        self.row['layers_before_gamma']={k:array_stats(v) for k,v in layers.items()}
        return layers

    @torch.no_grad()
    def predict(self,rgb,dn,fn,kn):
        self.row={'calls':{'MiDaS':0}};self.arrays={}
        alpha=self.module.salient_segmentation(rgb)
        alpha=cv2.resize(alpha,(rgb.shape[1],rgb.shape[0]))[...,None]
        assert np.isfinite(alpha).all() and 0<=alpha.min()<=alpha.max()<=1
        self.expected_mask=alpha.copy()
        hard=alpha>self.protocol['preprocessing']['threshold']
        self.row['LDF_mask']=dict(**array_stats(alpha),mean=float(alpha.mean()),coverage=float(hard.mean()),
                                  all_zero=bool(np.all(alpha==0)),all_one=bool(np.all(alpha==1)),
                                  threshold_empty=bool(not hard.any()),threshold_full=bool(hard.all()))
        params=dict(self.protocol['preprocessing'],focal=fn,lens_effect=kn,gamma=self.protocol['gamma'],offset=self.protocol['offset'])
        ret=self.model.inference(rgb,dn[...,None],alpha,params)
        value=ret['bokeh'];assert value.shape==rgb.shape and np.isfinite(value).all()
        self.arrays.update(ldf_alpha=alpha,input_disparity_normalized=dn)
        self.row['actual_layers']={}
        for name,layer in ret['layers'].items():
            assert layer.shape==(*rgb.shape[:2],5) and np.isfinite(layer).all()
            radius=kn*np.abs(layer[...,4]-fn)
            stats=dict(disparity_range=[float(layer[...,4].min()),float(layer[...,4].max())],
                       actual_radius_max=float(radius.max()),actual_radius_mean=float(radius.mean()),
                       radial_tail_mass_at_max=tail_mass(float(radius.max()),self.protocol['lens']))
            self.row['actual_layers'][name]=stats
            self.arrays[name+'_render']=layer.copy()
        expected={'LDF':1,'LaMa':2,'MiDaS':0,'RGBAD2layers':1,'process_fg':1,'process_bg':1,
                  'RGB_inpainting':1,'depth_inpainting':1,'layer_renderer':2}
        assert self.row['calls']==expected,self.row['calls']
        self.row['output']=dict(**array_stats(value),clipped_fraction=float(((value<0)|(value>1)).mean()))
        self.arrays['final_float_RGB']=value.copy()
        return value,self.row,self.arrays


@torch.no_grad()
def render_saved(renderer,arrays,kn,fn,gamma,offset):
    # Saved RGBAD values are exactly the inputs AFTER official gamma, not reconstructed approximations.
    value=renderer.inference([arrays['fg_rgbad_render'],arrays['bg_rgbad_render']],kn,fn)
    value=value**(1/gamma)-offset
    assert np.isfinite(value).all()
    return value


def lens_differences(a,b,alpha,dn,config):
    diff=np.abs(a-b)
    kernel=np.ones((5,5),np.uint8)
    hard=(alpha[...,0]>.1).astype(np.uint8)
    edge=(cv2.dilate(hard,kernel)!=cv2.erode(hard,kernel))
    edge|=(cv2.dilate(dn,kernel)-cv2.erode(dn,kernel)>1/255)
    av=np.round(np.clip(a,0,1)*255).astype(np.int16);bv=np.round(np.clip(b,0,1)*255).astype(np.int16)
    stats=dict(MAE=float(diff.mean()),max_abs=float(diff.max()),edge_MAE=float(diff[edge].mean()) if edge.any() else 0.,
               edge_max_abs=float(diff[edge].max()) if edge.any() else 0.,edge_fraction=float(edge.mean()),
               PNG8_max_abs=int(np.abs(av-bv).max()),pixels_above_1e4_fraction=float((diff.max(axis=-1)>1e-4).mean()))
    stats['passed']=bool(stats['MAE']<=config['whole_MAE_max'] and stats['edge_MAE']<=config['edge_MAE_max'] and
                         stats['max_abs']<=config['pixel_max_abs_max'] and stats['PNG8_max_abs']<=config['PNG8_max_abs_max'])
    return stats
