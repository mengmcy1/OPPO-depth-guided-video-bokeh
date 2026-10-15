"""Final CPU audit of approved MagicBokeh DAVIS results; no model inference."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads((ROOT/path).read_text())


def main():
    c=read('configs/magicbokeh-davis-frozen.json');data=read(c['data_protocol']);out=ROOT/c['output'];r=read(c['record'])
    assert r['frames']==1250 and r['videos']==50 and r['configuration']==c
    old=read('03_实验记录/records/davis-six-method-50-20260929.json')
    for key,value in old['methods'].items():
        assert all(r['comparison'][key][name]==value[name] for name in ['videos','frames','VEPI_public_mp4','VEPI_png','codec_delta'])
        assert r['codec']==value['codec'] and r['ffmpeg']==value['ffmpeg']
    assert len(r['rows'])==50 and len({s['id'] for s in r['rows']})==50
    for row in r['rows']:
        assert row['frames']==25 and row['unmodified_author_entrance_equal']
        assert len(row['frame_mp4'])==len(row['frame_png'])==25
        assert np.isfinite(row['frame_mp4']+row['frame_png']).all()
        assert row['VEPI_public_mp4']==float(np.mean(row['frame_mp4']))
        assert row['VEPI_png']==float(np.mean(row['frame_png']))
    for key in ['VEPI_public_mp4','VEPI_png','codec_delta']:
        assert abs(r[key]-float(np.mean([s[key] for s in r['rows']])))<1e-14
    coordinates=[];selections=[];audits=[]
    names=read(str((out/'protocol.json').relative_to(ROOT)))['scene_order']
    for index,scene in enumerate(names):
        click_ids=[];coverage=[]
        for t in range(25):
            row=read(str((out/'audit'/scene/f'{t:05d}.json').relative_to(ROOT)));audits.append(row)
            labels=np.asarray(Image.open(ROOT/data['source']/'Annotations'/data['resolution_source']/scene/f'{t:05d}.png'))
            height,width=labels.shape;x,y=row['click_xy'];source_x=int(x*width//1024);source_y=int(y*height//576)
            assert int(labels[source_y,source_x])==row['click_label_ID']!=0
            q=np.asarray(Image.open(out/'geometry'/scene/f'{t:05d}.png'))
            assert int(q[y,x])==row['own_depth_at_click_PNG8'] and row['focus']==q[y,x]/255
            assert row['seed']==42+25*index+t and row['condition_tiles_exactly_match_native']
            assert not row['target_mask_supplied_to_generator'] and row['native_tiles']==6 and row['finite']
            coordinates.append(dict(scene=scene,scene_index=index,frame=t,target_label_ID=row['click_label_ID'],
                display_x=x,display_y=y,original_annotation_x=source_x,original_annotation_y=source_y,
                original_width=width,original_height=height,own_depth_at_point_PNG8=int(q[y,x]),focus=row['focus'],K=16,seed=row['seed']))
            click_ids.append(row['click_label_ID']);coverage.append(row['target_focus_coverage'])
        selections.append(dict(scene=scene,click_IDs=sorted(set(click_ids)),click_ID_changes=sum(a!=b for a,b in zip(click_ids,click_ids[1:])),
            target_focused_coverage_range=[min(coverage),max(coverage)],
            note='Input control coverage only; not output sharpness/quality; no target or threshold adjustment'))
    path=out/'coordinate-audit.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,coordinates[0].keys(),lineterminator='\n');w.writeheader();w.writerows(coordinates)
    summary=dict(status='FINAL_1250_FRAME_COORDINATE_CONTROL_AND_VEPI_CHECKS_PASSED',checked_utc=datetime.now(timezone.utc).isoformat(),
        videos=50,frames=1250,original_to_display_nearest_annotation_coordinates_checked=1250,
        own_depth_focus_seed_and_native_conditions_checked=1250,per_frame_author_entry_equal=True,
        aggregate_means_recomputed=True,codec_exactly_matches_existing_six_methods=True,old_six_method_values_unchanged=True,
        zero_missing=True,constant_outputs=[[a['scene'],a['frame']] for a in audits if a['constant_output']],
        target_focus_coverage_range=[min(a['target_focus_coverage'] for a in audits),max(a['target_focus_coverage'] for a in audits)],
        selected_click_ID_changes_videos=sum(a['click_ID_changes']>0 for a in selections),rows=selections,
        coordinate_csv=str(path.relative_to(ROOT)),preflight_frames_reused=200,remaining_frames_new=1050,
        qualitative_status='No new human visual review or quality conclusion; native input/control variation retained',
        heldout_scope='50-video quantitative protocol only; no output/metric-driven held-out failure discovery or tuning')
    record=ROOT/'03_实验记录/records/magicbokeh-davis-final-check-20261015.json'
    record.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(summary['status'],summary['frames'],summary['target_focus_coverage_range'],flush=True)


if __name__=='__main__':main()
