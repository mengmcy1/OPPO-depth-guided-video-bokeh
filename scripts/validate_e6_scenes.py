"""Check each full E6 scene against its declared sources, geometry and output contract."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synthetic.any2bokeh_input import discover, validate


def inspect(folder, spec, outputs):
    row = validate(folder)
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['source'] == spec, folder
    c = meta['config']
    assert c['K'] == spec['K'] and c['background_depth_m'] == spec['background_depth_m']
    dynamic = 'source_video' in spec
    if dynamic:
        props = []
        for key in ['source_video','source_alpha']:
            cap = cv2.VideoCapture(str(ROOT/spec[key]))
            assert cap.isOpened()
            props.append([cap.get(k) for k in [cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS,cv2.CAP_PROP_FRAME_COUNT]])
            cap.release()
        assert props[0] == props[1] and props[0][2] > 0
        sw, sh, fps, count = props[0]
        assert spec['start_frame']+25 <= count
        row['source_video_properties'] = dict(width=sw,height=sh,fps=fps,frames=count)
    else:
        for file in spec['foreground_files']:
            alpha = np.array(Image.open(ROOT/file.replace('/fg/','/mask/')))
            assert not any(edge.any() for edge in [alpha[0],alpha[-1],alpha[:,0],alpha[:,-1]])
    coverage, soft, overlaps, radii = [], [], [], []
    for frame in meta['frames']:
        i = frame['index']
        assert frame['focus_distance_m'] == spec['focus_distance_m']
        assert frame['foreground_depths_m'] == spec['foreground_depths_m']
        assert frame['source_frame'] == spec.get('start_frame',0)+i
        if dynamic:
            matrix = np.array(frame['transforms'][0])
            corners = np.array([[0,0,1],[sw,sh,1]])@matrix.T
            assert np.all(corners[0]<=1e-5) and np.all(corners[1]>=[c['width']-1e-5,c['height']-1e-5])
        alpha = np.array(Image.open(folder/'alpha'/f'{i:05d}.png'))
        mask = np.array(Image.open(folder/'mask'/f'{i:05d}.png'))
        depth = np.load(folder/'depth'/f'{i:05d}.npy')
        assert alpha.shape == mask.shape == depth.shape
        # Quantized Alpha can land on 128 near a .5 threshold; check mask labels and range,
        # not a second thresholding of rounded Alpha as if it were exact float Alpha.
        assert set(np.unique(mask)) <= {0,255}
        levels = np.array(spec['foreground_depths_m']+[spec['background_depth_m']])
        assert all(np.isclose(z,levels).any() for z in np.unique(depth))
        coverage.append(float((alpha>127).mean()))
        soft.append(float(((alpha>0)&(alpha<255)).mean()))
        layer_alphas = [np.array(Image.open(folder/file))[:,:,3] for file in frame['foreground_files']]
        assert len(layer_alphas) == len(frame['foreground_depths_m'])
        if len(layer_alphas)>1:
            overlaps.append(int(np.logical_and.reduce([a>127 for a in layer_alphas]).sum()))
        radii.extend((c['K']*np.abs(1/levels-1/frame['focus_distance_m'])).tolist())
    assert max(coverage)>0, 'Entire clip has no visible foreground'
    row.update(alpha_coverage_min=min(coverage),alpha_coverage_max=max(coverage),
               soft_alpha_fraction_min=min(soft),soft_alpha_fraction_max=max(soft),
               empty_foreground_frames=sum(v==0 for v in coverage),
               overlap_pixels_min=min(overlaps) if overlaps else None,
               overlap_pixels_max=max(overlaps) if overlaps else None,
               nominal_layer_radii_px=sorted(set(radii)),
               weak_nonzero_radius_present=any(0<r<2 for r in radii),
               source_boundary_rule_pass=True, input_pass=True)
    if outputs:
        for sub in ['cycles_optix_s1024_f1','pred_any2bokeh']:
            files = sorted((folder/sub).glob('*.png'))
            assert [p.name for p in files] == [f'{i:05d}.png' for i in range(25)]
            for file in files:
                x = cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
                assert x is not None and x.shape == (c['height'],c['width'],3) and np.isfinite(x).all()
        timing = json.loads((folder/'cycles_optix_s1024_f1/timing.json').read_text())
        expected_fstop = (c['focal_length_mm']/1000)**2*c['width']/(2*(c['sensor_width_mm']/1000)*c['K'])
        assert abs(timing['aperture_fstop']-expected_fstop)<1e-5
        assert timing['samples']==1024 and timing['compute_device_type']=='OPTIX'
        conditioning = json.loads((folder/'pred_any2bokeh/input_check.json').read_text())
        assert conditioning['K']==c['K'] and conditioning['frames']==25
        assert conditioning['batch_shapes']['coc']==[6,8,3,c['height'],c['width']]
        row['actual_aperture_fstop'] = timing['aperture_fstop']
        row['output_pass'] = True
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--only',default='')
    p.add_argument('--outputs',action='store_true')
    p.add_argument('--report',type=Path,required=True)
    a = p.parse_args()
    config = json.loads((a.input/'config.json').read_text())
    specs = {s['id']:s for s in config['scenes']}
    folders = discover(a.input,a.only)
    if not a.only:
        assert {p.name for p in folders} == set(specs), 'Full manifest is not complete'
    rows = []
    for folder in folders:
        row = inspect(folder,specs[folder.name],a.outputs)
        rows.append(row)
        print(json.dumps(row),flush=True)
    with a.report.open('x') as f:
        json.dump(dict(scope='Per-scene structural checks; visual review is separate',rows=rows),f,indent=2)


if __name__ == '__main__':
    main()
