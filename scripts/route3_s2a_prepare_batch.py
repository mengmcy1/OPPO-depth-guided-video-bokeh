"""Resume stage2 through official data only; never starts a neural model.

Poll immutable per-scene milestones, stop on the user's specified conditions.
The S2a implementation starts only after this data stage is complete.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_s2a_screen import OUT,PY,CPU,run,write,LOG

def stopped():return (OUT/'STOP.json').exists() or ((OUT/'screening.json').exists() and json.loads((OUT/'screening.json').read_text()).get('stop_condition_triggered',False))
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['formal','derivatives'],required=True);a=p.parse_args();os.environ['ROUTE3_PILOT_CONFIG']=str(ROOT/'configs/route3-network-s2a-2026-11-12.json')
    if a.mode=='formal':
        while True:
            if stopped():return
            state=json.loads((OUT/'screening.json').read_text()) if (OUT/'screening.json').exists() else {}
            if state.get('screen_complete'):break
            time.sleep(30)
        run([PY,'scripts/route3_s2a_formal.py'],'formal-batch')
        if stopped() or not(OUT/'split.json').exists():return
        for mode in ['ldr','c2']:
            run([PY,'scripts/route3_s2a_heldout.py','--mode',mode],'heldout-'+mode)
            if stopped():return
        write(OUT/'formal-data-ready.json',dict(completed=True,split=str(OUT/'split.json'),all_formal_GT_and_LDR_and_C2=True,new_fusion_network_not_implemented_or_trained_here=True))
    else:
        done=set()
        while True:
            if stopped():return
            for folder in sorted(OUT.glob('scene-*')):
                if (folder/'formal-interface.json').exists() and folder.name not in done:
                    run([CPU,'scripts/route3_s2a_data.py','--scene',folder.name,'--size','256'],folder.name+'-S2a-data');done.add(folder.name);print('S2A_DERIVED',folder.name,flush=True)
            if (OUT/'split.json').exists():
                split=json.loads((OUT/'split.json').read_text());names=split['train']+split['heldout']
                if set(names)<=done:
                    first=split['train'][0];run([CPU,'scripts/route3_s2a_data.py','--scene',first,'--size','256','--GT-warmup'],first+'-GTwarmup-data')
                    for name in split['heldout']+['B2-high','pilot-thin','pilot-multilayer']:
                        for size in [256,512]:run([CPU,'scripts/route3_s2a_data.py','--scene',name,'--size',str(size)],name+f'-eval-data-{size}')
                    write(OUT/'derivative-data-ready.json',dict(completed=True,train_context_frames=[1,16,30,45,60],GTwarmup_scene=first,source_geometry_frontend_cached_from13_inputs=True,five_frame_recoverability_recomputed=True));return
            time.sleep(30)

if __name__=='__main__':main()
