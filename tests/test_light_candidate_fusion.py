"""Differentiability, empty queries and independent opacity are essential here."""
import unittest
import torch
from src.light_candidate_fusion import LightCandidateFusion

class LightFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1);torch.set_num_threads(1);self.net=LightCandidateFusion(width=16)
        images=torch.rand(2,3,8,8);poses=torch.eye(4)[None].repeat(2,1,1);poses[1,0,3]=.2;K=torch.tensor([[4.,0,4],[0,4.,4],[0,0,1.]])
        self.context=dict(images=images,c2w=poses,K=K);self.features=torch.rand(3,8,19);self.valid=torch.ones(3,8,dtype=torch.bool);self.slots=torch.arange(8)[None].repeat(3,1)%2;self.xy=torch.full((3,8,2),3.);self.z=torch.full((3,8),2.);self.condition=torch.zeros(3,6)
    def run_model(self):return self.net(self.features,self.valid,self.slots,self.xy,self.z,self.condition,self.context,torch.eye(4))
    def test_finite_gradients_through_geometry_and_source_rgb(self):
        p=self.run_model();color_grad=torch.autograd.grad(p['colors'].sum(),self.net.geometry.weight,retain_graph=True)[0];self.assertGreater(float(color_grad[2:4].abs().sum()),0);loss=(p['depth']-2.3).abs().mean()+p['colors'].sum()+p['opacities'].sum()+p['q_logits'].sum();loss.backward()
        self.assertEqual(p['means_camera'].shape,(3,2,3));self.assertTrue(all(torch.isfinite(v).all() for v in p.values()));self.assertTrue(all(torch.isfinite(x.grad).all() for x in self.net.parameters() if x.grad is not None));self.assertGreater(float(self.net.geometry.weight.grad[2:4].abs().sum()),0)
    def test_no_candidates_have_zero_weights_and_opacity(self):
        self.valid.zero_();p=self.run_model();self.assertEqual(float(p['weights'].sum()),0);self.assertEqual(float(p['opacities'].sum()),0);self.assertTrue(all(torch.isfinite(v).all() for v in p.values()))
    def test_source_rgb_uses_rotated_camera_and_pixel_centers(self):
        angle=torch.tensor(.5);c,s=torch.cos(angle),torch.sin(angle);R=torch.tensor([[c,0,s],[0,1,0],[-s,0,c]]);self.context['c2w'][1,:3,:3]=R;self.context['c2w'][1,:3,3]=torch.tensor([.1,.2,.3]);camera=torch.tensor([-.25,-.25,2.]);world=camera@R.T+self.context['c2w'][1,:3,3]
        colors,inside=LightCandidateFusion.source_colors(world[None,None],torch.ones((1,1),dtype=torch.long),self.context)
        torch.testing.assert_close(colors[0,0,0],self.context['images'][1,:,3,3],rtol=1e-5,atol=1e-6);self.assertTrue(bool(inside[0,0,0]))
    def test_uniform_confidence_does_not_scale_opacity(self):
        with torch.no_grad():
            self.net.geometry.weight[4].zero_();self.net.geometry.bias[4]=-20;p=self.run_model();self.net.geometry.bias[4]=20;r=self.run_model()
        torch.testing.assert_close(p['opacities'],r['opacities'],rtol=1e-5,atol=1e-6)
        self.assertGreater(float(r['q_logits'].sigmoid().mean()-p['q_logits'].sigmoid().mean()),.99)

if __name__=='__main__':unittest.main()
