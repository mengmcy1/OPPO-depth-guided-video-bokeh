#!/usr/bin/env python3
"""Match supplied DoF-NeRF geometry and rectify exact497x331 derived images."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from prepare_gs_repro import save

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/DoF-NeRF/DoF_RealWorld_Data'
LOG=ROOT/'logs/dof-gs-repro-2026-11-05'
SCENES=['amiya','camera','plant','kendo','desk','shelf']


def prepare(name,model):
    original=RAW/name
    cameras,images,points=model.read_model(str(original/'sparse/0'),ext='.bin')
    canonical=ROOT/'data/processed/dofnerf-scenes/mix'/name
    manifest=json.loads((canonical/'manifest.json').read_text())
    lookup={Path(v['source_files']['sharp']).name:v for v in manifest['views']}
    assert len(images)==len(lookup)==len(manifest['views']), 'Supplied camera/view count mismatch'
    assert {Path(i.name).name for i in images.values()}==set(lookup), 'Supplied geometry names do not match sharp views'
    bounds=np.load(original/'poses_bounds.npy')
    assert bounds.shape==(len(images),17)
    calibration={};intrinsics={};new_images={};checks=[]
    for camera in cameras.values():
        assert camera.model=='SIMPLE_RADIAL',camera.model
        f,cx,cy,k=camera.params
        sx,sy=497/camera.width,331/camera.height
        matrix=np.array([[f*sx,0,cx*sx],[0,f*sy,cy*sy],[0,0,1]],np.float64)
        distortion=np.array([k,0,0,0,0],np.float64)
        mapx,mapy=cv2.initUndistortRectifyMap(matrix,distortion,None,matrix,(497,331),cv2.CV_32FC1)
        grid=np.stack(np.meshgrid(np.arange(497),np.arange(331)),axis=-1).astype(np.float64)
        inverse=cv2.undistortPoints(grid.reshape(-1,1,2),matrix,distortion,P=matrix).reshape(331,497,2).astype(np.float32)
        calibration[camera.id]=(matrix,distortion,mapx,mapy,inverse)
        intrinsics[camera.id]=camera._replace(model='PINHOLE',width=497,height=331,
                                             params=np.array([matrix[0,0],matrix[1,1],matrix[0,2],matrix[1,2]]))
    for image in images.values():
        view=lookup[Path(image.name).name];camera=cameras[image.camera_id]
        matrix,distortion,_,_,_=calibration[camera.id]
        xys=image.xys*np.array([497/camera.width,331/camera.height])
        xys=cv2.undistortPoints(xys.reshape(-1,1,2),matrix,distortion,P=matrix).reshape(-1,2) if len(xys) else xys
        new_images[image.id]=image._replace(name=view['output_filename'],xys=xys)
        checks.append({'supplied_image':image.name,'output_image':view['output_filename'],'source_mode':'sharp',
                       'training_selected_mode':view['selected_mode'],'extrinsics_preserved':True})
    ordered=sorted(images.values(),key=lambda i:i.name)
    row_index={i.id:n for n,i in enumerate(ordered)}
    result={'status':'prepared','scene':name,'views':len(images),'points':len(points),'provided_geometry_used':True,
            'matching':checks,'triangulation_required':False,'resolution':[497,331],
            'camera_conversion':'SIMPLE_RADIAL -> standard OpenCV rectification -> PINHOLE; exact independent x/y scaling; radial coefficients preserved in records',
            'poses_source':str(original/'sparse/0'),'poses_bounds_source':str(original/'poses_bounds.npy')}
    for layout in ['mix','fg','bg']:
        source=ROOT/'data/processed/dofnerf-scenes'/layout/name
        target=ROOT/'data/processed/dofnerf-scenes-provided'/layout/name
        marker=target/'rectification.json'
        if marker.exists():continue
        (target/'images').mkdir(parents=True,exist_ok=True)
        (target/'rectification-maps').mkdir(exist_ok=True)
        image_models={i.name:i for i in new_images.values()}
        for filename,image_model in image_models.items():
            image=cv2.imread(str(source/'images_4'/filename));assert image.shape[:2]==(331,497)
            _,_,mapx,mapy,_=calibration[image_model.camera_id]
            rectified=cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            assert cv2.imwrite(str(target/'images'/filename),rectified)
        if not (target/'images_4').exists():(target/'images_4').symlink_to(target/'images',target_is_directory=True)
        sparse=target/'sparse/0';sparse.mkdir(parents=True,exist_ok=True)
        model.write_model(intrinsics,new_images,points,str(sparse),ext='.bin')
        converted=[]
        for image in sorted(new_images.values(),key=lambda i:i.name):
            row=bounds[row_index[image.id]].copy();pose=row[:15].reshape(3,5)
            pose[0,4]=331;pose[1,4]=497;pose[2,4]=intrinsics[image.camera_id].params[0]
            converted.append(row)
        np.save(target/'poses_bounds.npy',np.stack(converted))
        shutil.copy2(source/'manifest.json',target/'manifest.json');(target/'hold=9').touch()
        for camera_id,values in calibration.items():
            matrix,distortion,mapx,mapy,inverse=values
            np.savez_compressed(target/'rectification-maps'/f'{camera_id}.npz',to_original_x=inverse[...,0],to_original_y=inverse[...,1])
        save(marker,{'scene':name,'canonical_source':str(source),'native_images':'rectified497x331',
                     'evaluation':'reproject predictions/depth/alpha back to original497x331; canonical GT/masks remain unchanged',
                     'border':'replicate','image_camera_ids':{i.name:i.camera_id for i in new_images.values()},
                     'intrinsics':{str(c.id):{'supplied_model':c.model,'supplied_width':c.width,'supplied_height':c.height,'supplied_params':c.params.tolist(),
                                             'pinhole_params':intrinsics[c.id].params.tolist()} for c in cameras.values()}})
    return result


def colorize_training_sparse(name, model):
    """Extract missing sparse colors from mixed training views, keeping geometry."""
    source=ROOT/'data/processed/dofnerf-dofgaussian-provided/mix'/name
    target=ROOT/'data/processed/dofnerf-dofgaussian-traincolored/mix'/name
    marker=target/'training-point-colors.json'
    if marker.exists():return json.loads(marker.read_text())
    cameras,images,points=model.read_model(str(source/'sparse/0'),ext='.bin')
    assert all(np.all(p.rgb==0) for p in points.values()), 'Only handle the verified uncolored supplied model'
    point_ids=list(points);indices={pid:n for n,pid in enumerate(point_ids)}
    sums=np.zeros((len(points),3),np.float64);counts=np.zeros(len(points),np.int64)
    ordered=sorted(images.values(),key=lambda image:image.name)
    training=[image for n,image in enumerate(ordered) if n%9!=0]
    for image in training:
        rgb=cv2.cvtColor(cv2.imread(str(source/'images'/image.name)),cv2.COLOR_BGR2RGB)
        xy=image.xys;valid=(image.point3D_ids>=0)&(xy[:,0]>=0)&(xy[:,0]<rgb.shape[1])&(xy[:,1]>=0)&(xy[:,1]<rgb.shape[0])
        xy=xy[valid].astype(np.float32)
        colors=cv2.remap(rgb,xy[:,0].reshape(1,-1),xy[:,1].reshape(1,-1),cv2.INTER_LINEAR).reshape(-1,3)
        ids=np.array([indices[int(pid)] for pid in image.point3D_ids[valid]])
        np.add.at(sums,ids,colors);np.add.at(counts,ids,1)
    colors=np.rint(sums/np.maximum(counts[:,None],1)).clip(0,255).astype(np.uint8)
    colored={pid:point._replace(rgb=colors[indices[pid]]) for pid,point in points.items()}
    assert np.any(colors>0), 'No training colors extracted'
    target.mkdir(parents=True,exist_ok=True)
    for filename in ['images','images_4','poses_bounds.npy','manifest.json','rectification.json','rectification-maps','dense']:
        item=source/filename
        if item.exists() and not (target/filename).exists():(target/filename).symlink_to(item.resolve(),target_is_directory=item.is_dir())
    (target/'hold=9').touch()
    shutil.copytree(source/'sparse',target/'sparse',ignore=shutil.ignore_patterns('points3D.ply'))
    model.write_points3D_binary(colored,str(target/'sparse/0/points3D.bin'))
    _,_,check=model.read_model(str(target/'sparse/0'),ext='.bin')
    assert all(np.array_equal(points[pid].xyz,check[pid].xyz) and np.array_equal(points[pid].image_ids,check[pid].image_ids) and np.array_equal(points[pid].point2D_idxs,check[pid].point2D_idxs) for pid in points)
    assert all((source/'sparse/0'/filename).read_bytes()==(target/'sparse/0'/filename).read_bytes() for filename in ['cameras.bin','images.bin'])
    result={'scene':name,'source':str(source),'target':str(target),'points':len(points),
            'colored_from_training_observations':int(np.sum(counts>0)),'no_training_observation':int(np.sum(counts==0)),
            'training_images':[image.name for image in training],'excluded_test_images':[image.name for n,image in enumerate(ordered) if n%9==0],
            'geometry_and_tracks_preserved':True,'cameras_and_poses_byte_identical':True,'raw_data_modified':False,
            'protocol':'Bilinear RGB from rectified mixed training images at supplied feature observations; equal observation mean; unseen points remain black; no sharp/test GT used',
            'reason':'All supplied sparse RGB values are zero; native DoF-Gaussian renders and clamps them to black with no color learning'}
    save(marker,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--colorize-training-sparse',action='store_true')
    args=parser.parse_args()
    assert shutil.disk_usage('/').free>=1024**3
    cv2.setNumThreads(4)
    filename=Path('/root/autodl-tmp/tools/colmap-point-visibility/src/scripts/python/read_write_model.py')
    spec=importlib.util.spec_from_file_location('official_colmap_rw',filename);model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
    if args.colorize_training_sparse:
        results={name:colorize_training_sparse(name,model) for name in SCENES}
        save(ROOT/'03_实验记录/records/dof-gs-repro-2026-11-05-round1-training-point-colors.json',results)
        return
    report={}
    for name in SCENES:
        try:report[name]=prepare(name,model)
        except Exception as exc:report[name]={'status':'blocked','error':repr(exc),'fallback':'record mismatch before fixed-pose triangulation; do not guess image pairing'}
        print(name,report[name]['status'],flush=True)
        save(LOG/'provided-poses.json',report)
        from continue_gs_round1_preparation import update
        row=report[name]
        update('dofnerf/sparse/'+name,{'status':'complete' if row['status']=='prepared' else 'blocked',
                                     'registered_images':row.get('views'),'provided_geometry_used':True,
                                     'geometry_identical_across_layouts':row['status']=='prepared','details':str(LOG/'provided-poses.json')})
    save(LOG/'provided-poses.finished',report)
    save(ROOT/'03_实验记录/records/dof-gs-repro-2026-11-05-round1-provided-poses.json',report)


if __name__=='__main__':main()
