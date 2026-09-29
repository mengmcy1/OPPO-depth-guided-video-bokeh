"""Native DeepLens DepthNet plus unchanged pixel-focus LensBlur/single-SR graph."""
import importlib
from pathlib import Path
import numpy as np
from .deeplens import load_builders, tf, NATIVE_TF1


class DeepLensNative:
    def __init__(self, repository, checkpoint, threads=4):
        assert NATIVE_TF1, 'DAVIS uses the native TF1 CPU reference'
        lens, sr, changes = load_builders(repository)
        compat = Path(lens.__file__).parents[1]
        for relative in ['network/resnet50_lrelu.py', 'core/build_depth_resnet.py']:
            original = (repository / relative).read_text()
            path = compat / relative
            if path.exists(): assert path.read_text() == original
            else: path.write_text(original)
        depth = importlib.import_module('core.build_depth_resnet')
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/CPU:0'):
            self.low = tf.placeholder(tf.float32, [1,None,None,3])
            self.high = tf.placeholder(tf.float32, [1,None,None,3])
            self.x = tf.placeholder(tf.int32, [])
            self.y = tf.placeholder(tf.int32, [])
            self.aperture = tf.placeholder(tf.float32, [])
            training = tf.constant(False)
            with tf.variable_scope('Network'):
                self.depth, _ = depth.build(self.low, training)
                self.focus = self.depth[0,self.y,self.x,0]
                self.signed = (self.depth-self.focus)*self.aperture
                self.dof_low, _, _ = lens.build(self.low,self.signed,training)
                self.dof_high, _ = sr.build(self.low,self.signed,self.dof_low,self.high,training)
            variables = tf.global_variables()
            reader = tf.train.load_checkpoint(str(checkpoint))
            shapes = reader.get_variable_to_shape_map()
            for v in variables:
                assert v.op.name in shapes and v.shape.as_list() == shapes[v.op.name], v.op.name
                assert np.isfinite(reader.get_tensor(v.op.name)).all()
            saver = tf.train.Saver(variables)
            self.session = tf.Session(graph=self.graph,config=tf.ConfigProto(device_count={'GPU':0},
                intra_op_parallelism_threads=threads,inter_op_parallelism_threads=1,allow_soft_placement=False))
            saver.restore(self.session,str(checkpoint))
            assert not len(self.session.run(tf.report_uninitialized_variables(variables)))
            self.audit = dict(runtime=tf.__version__,device='CPU',restored_tensors=len(variables),
                checkpoint_tensors=len(shapes),DepthNet_built_and_restored=True,
                depth_tensors=sum('/Depth/' in v.op.name for v in variables),
                original_native_graph_operations=True,SR_stages=1,EMA=False,
                original_SR_uint8_intermediate_preserved=True,compatibility_files=changes)

    def predict_depth(self, low):
        d=self.session.run(self.depth,{self.low:low[None]})[0,:,:,0]
        assert d.shape==low.shape[:2] and np.isfinite(d).all() and d.max()>d.min()
        return d

    def predict(self, low, high, x, y, aperture, saved_depth):
        assert 0<=aperture<=1
        value,d,focus,signed=self.session.run([self.dof_high,self.depth,self.focus,self.signed],
            {self.low:low[None],self.high:high[None],self.x:x,self.y:y,self.aperture:aperture})
        assert all(np.isfinite(v).all() for v in [value,d,focus,signed])
        assert value.shape==(1,*high.shape)
        assert np.array_equal(d[0,:,:,0],saved_depth), 'Depth changed between focus selection and native rendering'
        expected=(saved_depth-saved_depth[y,x])*np.float32(aperture)
        error=float(np.max(np.abs(signed[0,:,:,0]-expected)))
        assert error<=1e-6 and focus==saved_depth[y,x]
        return value[0],dict(signed_error=error,focal_depth=float(focus),
            depth_repeat_bitwise_equal=True,raw_range=[float(value.min()),float(value.max())],
            signed_range=[float(signed.min()),float(signed.max())],
            output_clip_fraction=float(((value<0)|(value>1)).mean()))
