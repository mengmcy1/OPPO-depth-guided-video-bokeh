"""Correct content stays positive when a placement label is unsupported."""
import unittest
import numpy as np
import torch
from src.light_candidate_supervision import pack_targets,objectives

class SupervisionTest(unittest.TestCase):
    def test_unsupported_placement_does_not_negate_content(self):
        shape=(1,8);hit=np.zeros(shape,np.float32);hit[0,0]=1/16
        data=dict(candidate_valid=np.ones(shape,bool),target_K=np.eye(3),query_xy=np.zeros((1,2)))
        labels=dict(content_hit_fraction=hit,candidate_depth_target=np.ones(shape),candidate_RGB_target=np.zeros((*shape,3)),candidate_camera_target=np.ones((*shape,3)),placement_support_v2=np.zeros(shape,bool),appearance_valid=np.ones(shape,bool),hidden_any_at_query=np.ones(1,bool),recoverable_at_query=np.ones(1,bool))
        targets=pack_targets(labels,data);self.assertEqual(targets['q_target'][0,0],1);self.assertFalse(targets['valid'].any());q=torch.zeros(shape,requires_grad=True);pred=dict(depth=torch.zeros((1,2),requires_grad=True),colors=torch.zeros((1,2,3),requires_grad=True),weights=torch.full((1,2,8),1/8),selection_logits=torch.zeros((1,2,8),requires_grad=True),q_logits=q,opacity_logits=torch.zeros((1,2),requires_grad=True));t={k:torch.tensor(v) for k,v in targets.items()};loss,terms=objectives(pred,t,torch.ones(shape,dtype=torch.bool));loss.backward();self.assertLess(float(q.grad[0,0]),0);self.assertGreater(float(q.grad[0,1]),0);self.assertEqual(float(terms['relative_depth_L1']),0);self.assertEqual(float(terms['linear_RGB_L1']),0)

if __name__=='__main__':unittest.main()
