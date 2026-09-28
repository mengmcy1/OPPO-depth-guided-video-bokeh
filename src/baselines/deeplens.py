"""Headless external-disparity DeepLens graph, using the original TF operators and weights."""
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
ROOT = Path(__file__).resolve().parents[2]
AUTHOR_COMMIT = '3dbc1b04ba082b3b6b1dc0d944b3d18fa542ae2c'


def load_builders(repository):
    assert subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip() == AUTHOR_COMMIT
    assert not subprocess.check_output(['git', '-C', str(repository), 'diff', '--', 'network', 'core'], text=True)
    target = ROOT / 'logs/deeplens-runtime-20260928/compat-source'
    files = ['network/network.py', 'network/kernel_net.py', 'network/feature_net.py',
             'network/srnet.py', 'core/build_lensblur.py', 'core/build_sr.py']
    changes = []
    for relative in files:
        original = (repository / relative).read_text()
        text = original.replace('import tensorflow as tf', 'import tensorflow.compat.v1 as tf')
        if relative == 'network/network.py':
            # Only syntax/API compatibility. All variables are restored; no initializer is run.
            assert 'tf.contrib.layers.xavier_initializer()' in text
            text = text.replace('tf.contrib.layers.xavier_initializer()', 'tf.glorot_uniform_initializer()')
            text = text.replace('basestring', 'str').replace('.iteritems()', '.items()')
            text = text.replace('c_i / group', 'int(c_i) // group').replace('c_o/group', 'c_o//group')
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_text() == text
        else:
            path.write_text(text)
        changes.append(dict(file=relative, modified=original != text))
    for relative, text in [('network/__init__.py', 'from .network import Network\n'), ('core/__init__.py', '')]:
        path = target / relative
        if path.exists():
            assert path.read_text() == text
        else:
            path.write_text(text)
    sys.path.insert(0, str(target))
    return importlib.import_module('core.build_lensblur'), importlib.import_module('core.build_sr'), changes


class DeepLensExternal:
    def __init__(self, repository, checkpoint, threads=8):
        lens, sr, changes = load_builders(repository)
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/CPU:0'):
            self.low = tf.placeholder(tf.float32, [1, None, None, 3], name='image_320_external')
            self.high = tf.placeholder(tf.float32, [1, None, None, 3], name='image_640_external')
            self.disparity = tf.placeholder(tf.float32, [1, None, None, 1], name='external_disparity')
            self.focus = tf.placeholder(tf.float32, [], name='external_focus')
            self.aperture = tf.placeholder(tf.float32, [], name='aperture')
            # Explicit legacy TF1 resize convention; no half-pixel centers, no depth inversion.
            self.disparity_low = tf.image.resize_bilinear(self.disparity, tf.shape(self.low)[1:3], align_corners=False)
            self.signed = (self.disparity_low - self.focus) * self.aperture
            training = tf.constant(False)
            with tf.variable_scope('Network'):
                self.dof_low, _, _ = lens.build(self.low, self.signed, training)
                self.dof_high, _ = sr.build(self.low, self.signed, self.dof_low, self.high, training)
            variables = tf.global_variables()
            reader = tf.train.load_checkpoint(str(checkpoint))
            checkpoint_shapes = reader.get_variable_to_shape_map()
            for v in variables:
                assert v.op.name in checkpoint_shapes, f'Missing checkpoint variable: {v.op.name}'
                assert v.shape.as_list() == checkpoint_shapes[v.op.name]
                assert np.isfinite(reader.get_tensor(v.op.name)).all()
            assert variables and all('/Depth/' not in v.op.name for v in variables)
            saver = tf.train.Saver(variables)
            uninitialized = tf.report_uninitialized_variables(variables)
            config = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=threads,
                                    inter_op_parallelism_threads=1, allow_soft_placement=False)
            self.session = tf.Session(graph=self.graph, config=config)
            saver.restore(self.session, str(checkpoint))
            assert len(self.session.run(uninitialized)) == 0
            self.audit = dict(source_commit=AUTHOR_COMMIT, runtime=tf.__version__, device='CPU',
                checkpoint_tensors=len(checkpoint_shapes), restored_tensors=len(variables),
                restored_elements=sum(int(np.prod(v.shape.as_list())) for v in variables),
                scopes=sorted(set(v.op.name.split('/')[2] for v in variables)),
                all_restored_finite=True, no_uninitialized_variables=True,
                DepthNet_built_or_executed=False, compatibility_files=changes,
                native_SR_uint8_intermediate_preserved=True,
                limitation='TF2 compat.v1 CPU execution; numerical parity against a native TF1 binary is not established.')

    def predict(self, bgr_low, bgr_high, disparity_full, focus, aperture):
        assert bgr_high.shape[:2] == disparity_full.shape
        assert bgr_high.shape[:2] == (bgr_low.shape[0] * 2, bgr_low.shape[1] * 2)
        assert np.isfinite(disparity_full).all() and 0 <= aperture <= 1
        value, low, disparity, signed = self.session.run(
            [self.dof_high, self.dof_low, self.disparity_low, self.signed],
            {self.low: bgr_low[None], self.high: bgr_high[None], self.disparity: disparity_full[None, ..., None],
             self.focus: focus, self.aperture: aperture})
        assert all(np.isfinite(x).all() for x in [value, low, disparity, signed])
        assert value.shape == (1, *bgr_high.shape) and low.shape == (1, *bgr_low.shape)
        return value[0], dict(low_output_shape=list(low.shape), high_output_shape=list(value.shape),
            low_disparity_range=[float(disparity.min()), float(disparity.max())],
            signed_disparity_range=[float(signed.min()), float(signed.max())],
            raw_output_range=[float(value.min()), float(value.max())],
            output_clip_fraction=float(((value < 0) | (value > 1)).mean()))
