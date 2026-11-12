"""Contract checks for newly introduced source references and factual memory."""
import unittest
import numpy as np
from src.candidate_query_data import observation_flat_index,pack_queries,query_features
from src.observation_memory import build_memory


class PilotObservationData(unittest.TestCase):
    def test_int16_coordinates_reach_last_source_pixel(self):
        ids=observation_flat_index(np.array([0,12],np.uint8),np.array([[0,0],[511,511]],np.int16))
        np.testing.assert_array_equal(ids,[0,13*512*512-1])

    def test_unverified_observation_is_kept_raw_but_not_fused(self):
        xyz=np.array([[0.,0,1.],[.001,0,1.],[.002,0,1.]],np.float32);rgb=np.full((3,3),.2,np.float32);refs=np.array([[1,1,1],[6,2,2],[11,3,3]])
        mem,stats=build_memory(xyz,rgb,np.ones(3),refs,verified=np.array([True,False,True]))
        self.assertEqual(mem['observation_entry_id'][1],-1)
        np.testing.assert_array_equal(mem['original_source_references'],refs)
        self.assertNotIn(1,mem['observation_indices']);self.assertEqual(stats['verified_observations'],2)
        self.assertEqual(mem['feature'].shape[1],0)

    def test_queries_keep_empty_locations_and_whole_candidates(self):
        band=np.array([[True,True],[False,True]]);pool=dict(pixel=np.array([0,0,3]),depth=np.array([4.,2.,5.]))
        q,_=pack_queries(pool,band,max_candidates=1,max_queries=10)
        np.testing.assert_array_equal(q['query_pixel'],[0,1,3]);np.testing.assert_array_equal(q['candidate_indices'][:,0],[1,-1,2]);self.assertFalse(q['candidate_valid'][1,0])

    def test_unknown_confidence_evidence_is_finite_and_flagged(self):
        pool=dict(position_world_estimated=np.zeros((1,3)),depth=np.ones(1),colors=np.ones((1,3))*.2,edge_distance=np.ones(1),reprojection=np.array([np.nan]),FB=np.array([np.inf]),angle_radians=np.zeros(1),weight=np.ones(1)*.05,source_xy=np.array([[511,511]],np.int16),source_frame_id=np.array([60]),edge=np.array([False]),rgb_patch=np.zeros((1,3,3,3),np.uint8))
        q,_=query_features(pool,np.array([[0,-1]]),np.array([[True,False]]));self.assertTrue(np.isfinite(q['features']).all());self.assertTrue(q['feature_missing_mask'][0,0,:2].all());self.assertTrue((q['features'][0,1]==0).all())


if __name__=='__main__':unittest.main()
