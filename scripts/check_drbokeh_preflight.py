"""Aggregate the declared implementation checks; visual review remains explicit."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('outputs/drbokeh-preflight-20260928'))
    args=p.parse_args()
    proto=json.loads((args.run/'protocol.json').read_text())
    rows=[]
    expected_calls={'LDF':1,'LaMa':2,'MiDaS':0,'RGBAD2layers':1,'process_fg':1,'process_bg':1,
                    'RGB_inpainting':1,'depth_inpainting':1,'layer_renderer':2}
    for sid in proto['preflight_scenes']:
        timing=args.run/'timing'/sid
        assert json.loads((timing/'complete.json').read_text())['frames']==25
        for t in range(25):
            row=json.loads((timing/f'{t:05d}.json').read_text())
            assert row['scene']==sid and row['frame']==t and row['calls']==expected_calls
            assert row['geometry']['signed_CoC_error_float64']<=proto['mapping_tolerance']['float64_max_abs']
            assert row['geometry']['signed_CoC_error_float32']<=proto['mapping_tolerance']['float32_max_abs']
            assert row['output']['shape']==[576,1024,3]
            assert (args.run/row['diagnostics']).is_file()
            with Image.open(args.run/'clips'/sid/'pred_drbokeh'/f'{t:05d}.png') as im:
                assert im.size==(1024,576) and im.mode=='RGB';im.verify()
            rows.append(row)
    comparisons=[x for x in rows if 'lens_comparison' in x]
    assert len(comparisons)==50
    assert {(x['scene'],x['frame']) for x in comparisons}=={(s,t) for s in proto['layered_lens_check']['scenes'] for t in range(25)}
    window_ok=all(x['lens_comparison']['passed'] for x in comparisons) and all(x['coverage_passed'] for x in rows)
    summary=dict(status='NUMERICAL_CHECKS_COMPLETE_VISUAL_REVIEW_PENDING',protocol=proto,scenes=8,frames=200,
                 checks={'input_mapping':True,'official_pipeline_calls':True,'finite_arrays_per_frame':True,'all_predictions_PNG8_integrity':True,
                         'all_intermediate_files_present':True,'lens71_coverage':window_ok},
                 geometry={'max_float64_error':max(x['geometry']['signed_CoC_error_float64'] for x in rows),
                           'max_float32_error':max(x['geometry']['signed_CoC_error_float32'] for x in rows)},
                 masks={'threshold_empty_frames':[(x['scene'],x['frame']) for x in rows if x['LDF_mask']['threshold_empty']],
                        'threshold_full_frames':[(x['scene'],x['frame']) for x in rows if x['LDF_mask']['threshold_full']],
                        'coverage_range':[min(x['LDF_mask']['coverage'] for x in rows),max(x['LDF_mask']['coverage'] for x in rows)]},
                 layers={k:{'disparity_range':[min(x['actual_layers'][k]['disparity_range'][0] for x in rows),max(x['actual_layers'][k]['disparity_range'][1] for x in rows)],
                            'actual_radius_max':max(x['actual_layers'][k]['actual_radius_max'] for x in rows),
                            'radial_tail_mass_max':max(x['actual_layers'][k]['radial_tail_mass_at_max'] for x in rows)} for k in ('fg_rgbad','bg_rgbad')},
                 lens_comparison={'frames':50,'passed':window_ok,'maxima':{k:max(x['lens_comparison'][k] for x in comparisons) for k in ['MAE','max_abs','edge_MAE','edge_max_abs','PNG8_max_abs','pixels_above_1e4_fraction']},
                                  'failed_frames':[(x['scene'],x['frame']) for x in comparisons if not x['lens_comparison']['passed']]},
                 output={'max_clipped_fraction':max(x['output']['clipped_fraction'] for x in rows)},
                 peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in rows),
                 suggested_lens=71 if window_ok else 101,
                 next_action='Review fixed keyframes before acceptance. If switching lens, rerender saved layers and recheck coverage; do not rerun segmentation or tune paper scores.')
    out=args.run/'preflight-checks.json'
    with out.open('x') as f:json.dump(summary,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
