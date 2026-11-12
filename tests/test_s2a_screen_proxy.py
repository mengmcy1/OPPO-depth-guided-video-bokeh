"""Screening measures ray-visible colors without needing a DOF render."""
import importlib.util
from pathlib import Path
import numpy as np
import unittest
import tempfile
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('s2a_proxy',ROOT/'scripts/route3_s2a_proxy.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ScreenProxyTest(unittest.TestCase):
    def test_recoverable_ray_lookup_has_correct_psnr_and_area(self):
        with tempfile.TemporaryDirectory() as directory:
            self.check(Path(directory))

    def check(self,tmp_path):
        K=np.array([[2.,0,1],[0,2.,1],[0,0,1.]])
        yy,xx=np.mgrid[:2,:2];uv=np.dstack([xx+.5,yy+.5]);shape=(16,2,2);rec=np.zeros(shape,bool);rec[:,:1]=True
        g=tmp_path/'geometry.npz';np.savez(g,K=K,recoverable=rec,hidden=rec,residual_pixels=np.zeros((2,2),bool),mask_recoverable_total=rec.any(0),reference_uv_center=np.broadcast_to(uv,(16,2,2,2)),depth_axial_m=np.full(shape,2.),object_id=np.ones(shape,int),aperture_xy=np.zeros((16,2)),frame_id=1)
        d=tmp_path/'input.npz';np.savez(d,depth=np.ones((2,2)),background_depth=np.full((2,2),2.))
        f=tmp_path/'fill.npz';np.savez(f,fill_rgb_linear=np.zeros((2,2,3)))
        p=tmp_path/'parent.npz';np.savez(p,uv=uv.reshape(-1,2),depth=np.full(4,2.),object_id=np.ones(4,int),colors_linear=np.ones((4,3)))
        e=dict(input_npz=str(d),fill=str(f),proxy_nominal_focus_m=5.,cases=[dict(nominal_focus_m=5.,focus_m=5.,fstop=1.4)])
        r=m.proxy(g,e,parent_path=p)
        assert r['recoverable_image_fraction']==.5
        assert r['recoverable_rays']==32
        assert r['color_support_fraction']==1.
        assert abs(r['fill_hidden_PSNR_dB'])<1e-6
        np.savez(f,fill_rgb_linear=np.ones((2,2,3)))
        assert m.proxy(g,e,parent_path=p)['fill_hidden_PSNR_dB']==120.

if __name__=="__main__":unittest.main()
