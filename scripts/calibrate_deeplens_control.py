"""Independent edge-spread control measurement; no benchmark score is used to choose aperture."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def edge_response(value,orientation):
    gray=value.mean(axis=-1)
    profile=gray[128:-128].mean(axis=0) if orientation=='vertical' else gray[:,256:-256].mean(axis=1)
    return profile_response(profile)


def profile_response(profile):
    center=len(profile)//2
    profile=profile[center-80:center+80].astype(np.float64)
    left=float(profile[:16].mean());right=float(profile[-16:].mean());contrast=right-left
    assert contrast>.25 and np.isfinite(profile).all()
    normalized=(profile-left)/contrast
    lsf=np.diff(normalized);x=np.arange(len(lsf),dtype=np.float64)-79.5
    mass=float(lsf.sum());mu=float((x*lsf).sum()/mass);variance=float(((x-mu)**2*lsf).sum()/mass)
    negative=float(-lsf[lsf<0].sum()/max(lsf[lsf>0].sum(),1e-12))
    # Record the central transition crossings; do not silently discard ringing.
    crossings=[]
    for threshold in (.1,.9):
        indices=np.flatnonzero((normalized[:-1]<threshold)&(normalized[1:]>=threshold))
        assert len(indices)>0
        choices=[float(i+(threshold-normalized[i])/(normalized[i+1]-normalized[i])) for i in indices]
        crossings.append(min(choices,key=lambda v:abs(v-79.5)))
    return dict(left=left,right=right,contrast=contrast,lsf_mass=mass,centroid_px=mu,variance_px2=variance,
                negative_lsf_mass_ratio=negative,edge_10_90_width_px=crossings[1]-crossings[0],profile=profile.tolist())


def analyze(out):
    """Keep the failed second-moment estimate; assess the already recorded edge width."""
    raw=json.loads((out/'result.json').read_text())
    rows=raw['rows']
    def disk_cdf(x):
        x=np.clip(x,-1,1)
        return .5+(np.arcsin(x)+x*np.sqrt(np.maximum(0,1-x*x)))/np.pi
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2
        if disk_cdf(mid)<.9:lo=mid
        else:hi=mid
    factor=lo+hi  # twice the normalized .9 quantile
    # Independent positive disk reference, sampled at pixel centers with area integration.
    references=[]
    for radius in (2,4,8,16,32):
        x=np.arange(1024)-511.5
        subs=(np.arange(256)+.5)/256-.5
        profile=.25+.5*disk_cdf((x[:,None]+subs)/radius).mean(axis=1)
        measure=profile_response(profile)
        inferred=measure['edge_10_90_width_px']/factor
        assert abs(inferred-radius)<.25
        references.append(dict(radius_px=radius,measured_radius_px=inferred,error_px=inferred-radius))
    condensed=[]
    for row in rows:
        r=row['response']
        condensed.append(dict(orientation=row['orientation'],delta=row['delta'],aperture=row['aperture'],
            width_px=r['edge_10_90_width_px'],disk_edge_equivalent_radius_px=r['edge_10_90_width_px']/factor,
            signed_variance_px2=r['variance_px2'],variance_above_zero=row['variance_above_zero'],
            negative_lsf_mass_ratio=r['negative_lsf_mass_ratio']))
    nonzero=[r for r in condensed if r['aperture']>0]
    x=np.array([abs(r['delta'])*r['aperture'] for r in nonzero])
    y=np.array([r['disk_edge_equivalent_radius_px'] for r in nonzero])
    slope=float(x@y/(x@x))
    maximum=[r['disk_edge_equivalent_radius_px'] for r in condensed if abs(r['delta'])==1 and r['aperture']==1]
    monotonic={str((orientation,delta)):bool(np.all(np.diff([r['width_px'] for r in condensed if r['orientation']==orientation and r['delta']==delta])>0))
        for orientation in ['vertical','horizontal'] for delta in [-1.,-.5,.5,1.]}
    result=dict(status='SINGLE_SR_CONTROL_MEASURED_TARGET_SUPPORT_DECISION_PENDING',
        source=str(out.relative_to(ROOT)/'result.json'),protocol=raw['protocol'],runtime=raw['load'],
        rejected_measurement='Signed LSF second moment is not a nonnegative PSF variance: neural ringing gives negative variance/increments. The original max(variance,0) radius and fitted scale are diagnostic only, not a valid frozen mapping.',
        replacement_measurement='Previously recorded10-90edge width / analytic uniform-disk factor. Reports an equivalent width, not exact physical kernel support. No subtraction of the small aperture0 pixel-transition width.',
        disk_edge_width_per_radius=float(factor),positive_disk_validation=references,
        aperture0_width_range_px=[min(r['width_px'] for r in condensed if r['aperture']==0),max(r['width_px'] for r in condensed if r['aperture']==0)],
        response_monotonic=monotonic,rows=condensed,
        descriptive_zero_intercept_scale_px=slope,fit_RMSE_px=float(np.sqrt(np.mean((y-slope*x)**2))),
        max_control_edge_equivalent_radius_range_px=[min(maximum),max(maximum)],
        target_K_outside_measured_range=[k for k in [4,8,16,32] if k>max(maximum)],
        mapping_frozen=False,benchmark_inference_started=False,
        limits=['Mapping has not been selected from paper or benchmark scores.',
            'The width estimator was retained in the initial protocol; second moment failure is disclosed rather than hidden.',
            'Uniform-plane edge response does not establish textured/occlusion behavior or a strict support bound.',
            'Choosing saturation or altering the native SR scale policy requires an explicit protocol decision.'])
    path=ROOT/'03_实验记录/records/deeplens-independent-control-20260929.json'
    with path.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','disk_edge_width_per_radius','positive_disk_validation','descriptive_zero_intercept_scale_px','fit_RMSE_px','max_control_edge_equivalent_radius_range_px','target_K_outside_measured_range']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/deeplens-control-calibration-20260929'))
    p.add_argument('--analyze',action='store_true',help='Analyze saved controls without running TensorFlow again')
    args=p.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    if args.analyze:
        analyze(out);return
    from baselines.deeplens import DeepLensExternal,NATIVE_TF1
    assert NATIVE_TF1 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert os.environ.get('TF_ENABLE_ONEDNN_OPTS')=='0'
    protocol=dict(resolution=[1024,576],low_resolution=[512,288],levels=[.25,.75],
        orientations=['vertical','horizontal'],signed_disparities=[-1.,-.5,.5,1.],apertures=[0.,.25,.5,1.],
        geometry='One constant-disparity plane. Delta+1: d=1/focus=0; -1: d=0/focus=1; +.5: d=.75/focus=.25; -.5: d=.25/focus=.75.',
        preprocessing='Original BGR float32, INTER_AREA half resolution, original single guided SR to1024x576',
        measurement='Signed edge-derivative second moment in central160px window, averaged along edge; subtract same-orientation aperture0 variance; equivalent uniform-disk radius=2*sqrt(max(var-var0,0)). Also record negative LSF mass and10-90width.',
        planned_mapping='If response is sufficiently coherent, fit one zero-intercept image-space scale R_eff=c*abs(delta)*aperture, using all predeclared nonzero controls. Inspect monotonicity, orientation/sign and delta=.5 residuals before freezing. Report any target K outside measured native support; no silent saturation or parameter sweep.',
        scope='Independent constructed control images only; no AM/VM/benchmark predictions, PSNR or FVD used for calibration.')
    plan=out/'protocol.json'
    if plan.exists():assert json.loads(plan.read_text())==protocol
    else:plan.write_text(json.dumps(protocol,indent=2)+'\n')
    cv2.setNumThreads(2)
    model=DeepLensExternal(ROOT.parent/'DeepLens',ROOT/'models/DeepLens/model',threads=8)
    assert model.audit['runtime']=='1.15.0'
    rows=[]
    for orientation in protocol['orientations']:
        high=np.full((576,1024,3),.25,np.float32)
        if orientation=='vertical':high[:,512:]=.75
        else:high[288:]=.75
        low=cv2.resize(high,(512,288),interpolation=cv2.INTER_AREA)
        for delta in protocol['signed_disparities']:
            d=(1+delta)/2;focus=(1-delta)/2;disp=np.full((576,1024),d,np.float32)
            for aperture in protocol['apertures']:
                name='{}_d{}_a{}'.format(orientation,delta,aperture);record=out/(name+'.json')
                if record.exists():rows.append(json.loads(record.read_text()));continue
                start=time.monotonic();value,audit=model.predict(low,high,disp,focus,aperture)
                result=dict(name=name,orientation=orientation,delta=delta,aperture=aperture,seconds=time.monotonic()-start,
                    audit=audit,response=edge_response(value,orientation))
                # Keep float outputs: final PNG quantization is not part of this measurement.
                with (out/(name+'.npy')).open('xb') as f:np.save(f,value)
                with record.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
                rows.append(result);print('CONTROL',name,'variance',result['response']['variance_px2'],'width',result['response']['edge_10_90_width_px'],flush=True)
    for row in rows:
        zero=next(r for r in rows if r['orientation']==row['orientation'] and r['delta']==row['delta'] and r['aperture']==0)
        row['variance_above_zero']=row['response']['variance_px2']-zero['response']['variance_px2']
        row['equivalent_radius_px']=2*np.sqrt(max(0,row['variance_above_zero']))
    nonzero=[r for r in rows if r['aperture']>0];x=np.array([abs(r['delta'])*r['aperture'] for r in nonzero]);y=np.array([r['equivalent_radius_px'] for r in nonzero]);slope=float(x@y/(x@x))
    result=dict(status='CONTROL_RESPONSE_MEASURED_MAPPING_NOT_YET_FROZEN',completed_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,
                load=model.audit,rows=rows,zero_intercept_scale_px=slope,fit_RMSE_px=float(np.sqrt(np.mean((y-slope*x)**2))),
                max_equivalent_radius_px=float(y.max()),limits=['A learned image operator need not be an ideal disk convolution; negative line-spread lobes/sign asymmetry must be inspected.','The equivalent radius and10-90width are response summaries, not proof of physically exact bokeh.'])
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('CALIBRATION_COMPLETE',slope,result['fit_RMSE_px'],result['max_equivalent_radius_px'],flush=True)


if __name__=='__main__':main()
