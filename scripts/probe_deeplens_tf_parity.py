"""Execute identical external-geometry graphs in native TF1 and TF2 compat.v1 on CPU."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',choices=['native_tf1','compat_tf2'],required=True)
    p.add_argument('--inputs',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shared-low',type=Path,help='Diagnostic only: feed the exact saved native low tensor into SR')
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('TF_ENABLE_ONEDNN_OPTS')=='0'
    if args.runtime=='native_tf1':
        import tensorflow as tf
        assert tf.__version__=='1.15.0'
    else:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        assert tf.__version__=='2.15.1'
    repo=ROOT.parent/'DeepLens'
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],universal_newlines=True).strip()=='3dbc1b04ba082b3b6b1dc0d944b3d18fa542ae2c'
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--','network','core'])
    source=ROOT/'logs/deeplens-tf-parity-20260929'/args.runtime/'source'
    for rel in ['network/network.py','network/kernel_net.py','network/feature_net.py','network/srnet.py','core/build_lensblur.py','core/build_sr.py']:
        text=(repo/rel).read_text()
        if rel=='network/network.py':
            text=text.replace('basestring','str').replace('.iteritems()','.items()').replace('c_i / group','int(c_i) // group').replace('c_o/group','c_o//group')
        if args.runtime=='compat_tf2':
            text=text.replace('import tensorflow as tf','import tensorflow.compat.v1 as tf').replace('tf.contrib.layers.xavier_initializer()','tf.glorot_uniform_initializer()')
        target=source/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_text()==text
        else:target.write_text(text)
    for rel,text in [('network/__init__.py','from .network import Network\n'),('core/__init__.py','')]:
        target=source/rel
        if target.exists():assert target.read_text()==text
        else:target.write_text(text)
    sys.path.insert(0,str(source));lens=importlib.import_module('core.build_lensblur');sr=importlib.import_module('core.build_sr')
    graph=tf.Graph()
    with graph.as_default(),tf.device('/CPU:0'):
        lo=tf.placeholder(tf.float32,[1,None,None,3],name='image_320_external')
        hi=tf.placeholder(tf.float32,[1,None,None,3],name='image_640_external')
        disp=tf.placeholder(tf.float32,[1,None,None,1],name='external_disparity')
        focus=tf.placeholder(tf.float32,[],name='external_focus');aperture=tf.placeholder(tf.float32,[],name='aperture')
        resized=tf.image.resize_bilinear(disp,tf.shape(lo)[1:3],align_corners=False)
        signed=(resized-focus)*aperture
        with tf.variable_scope('Network'):
            low,_,_=lens.build(lo,signed,tf.constant(False))
            high,_=sr.build(lo,signed,low,hi,tf.constant(False))
        # Inspect the native SR quantization point independently of final PNG encoding.
        sr_uint8=tf.cast(low*255,tf.uint8)
        variables=tf.global_variables();reader=tf.train.load_checkpoint(str(ROOT/'models/DeepLens/model'))
        for v in variables:
            assert reader.get_variable_to_shape_map()[v.op.name]==v.shape.as_list()
            assert np.isfinite(reader.get_tensor(v.op.name)).all()
        assert len(variables)==210 and all('/Depth/' not in v.op.name for v in variables)
        saver=tf.train.Saver(variables);uninitialized=tf.report_uninitialized_variables(variables)
    config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=8,inter_op_parallelism_threads=1,allow_soft_placement=False)
    out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
    rows=[]
    with tf.Session(graph=graph,config=config) as session:
        saver.restore(session,str(ROOT/'models/DeepLens/model'));assert len(session.run(uninitialized))==0
        for path in args.inputs:
            path=ROOT/path
            with np.load(path) as z:inputs={k:z[k] for k in z.files}
            feed={lo:inputs['bgr_low'][None],hi:inputs['bgr_high'][None],disp:inputs['disparity'][None,...,None],focus:float(inputs['focus']),aperture:float(inputs['aperture'])}
            if args.shared_low:
                assert len(args.inputs)==1
                with np.load(ROOT/args.shared_low) as saved:feed[low]=saved['low']
            start=time.monotonic();values=session.run(dict(resized_disparity=resized,signed=signed,low=low,high=high,sr_uint8=sr_uint8),feed)
            assert all(np.isfinite(v).all() for v in values.values())
            with (out/(path.stem+'.npz')).open('xb') as f:np.savez_compressed(f,**values)
            row=dict(input=str(path.relative_to(ROOT)),seconds=time.monotonic()-start,shapes={k:list(v.shape) for k,v in values.items()})
            rows.append(row);print('DONE',args.runtime,path.stem,row['seconds'],flush=True)
    with (out/'result.json').open('x') as f:
        json.dump(dict(runtime=args.runtime,tensorflow=tf.__version__,python=sys.version,numpy=np.__version__,restored_variables=210,rows=rows,scope='CPU external disparity LensBlur + Feature + one guided SR; no DepthNet, training, benchmark metric or aperture mapping choice'),f,indent=2)


if __name__=='__main__':main()
