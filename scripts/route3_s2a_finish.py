"""Wait for authorized data/split, then fit and evaluate the light model only.

No automatic S2b or Query upgrade. The user-defined stop markers take
precedence; completed image/model artifacts are never removed.
"""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_s2a_screen import OUT,PY,run,write

def main():
    os.environ['ROUTE3_PILOT_CONFIG']=str(ROOT/'configs/route3-network-s2a-2026-11-12.json')
    while True:
        if (OUT/'STOP.json').exists():return
        if (OUT/'formal-data-ready.json').exists() and (OUT/'derivative-data-ready.json').exists():break
        time.sleep(30)
    stages=[('single-scene-overfit',[PY,'scripts/route3_s2a_learn.py','--mode','overfit']),('S2a-train',[PY,'scripts/route3_s2a_learn.py','--mode','S2a']),('S2a-predict',[PY,'scripts/route3_s2a_predict.py']),('S2a-direct',[PY,'scripts/route3_s2a_learn.py','--mode','direct']),('S2a-score-Cycles',[PY,'scripts/route3_s2a_evaluate.py','--reference','Cycles']),('S2a-score-LDR',[PY,'scripts/route3_s2a_evaluate.py','--reference','LDR'])]
    rows=[]
    for name,cmd in stages:
        seconds=run(cmd,name,gpu=name not in ['S2a-score-Cycles','S2a-score-LDR']);rows.append(dict(stage=name,seconds=seconds));write(OUT/'learning-stage-progress.json',dict(rows=rows))
        if (OUT/'STOP.json').exists():return
    write(OUT/'evaluation-complete.json',dict(completed=True,rows=rows,next='stop_and_report;no_S2b_or_Query_upgrade'))

if __name__=='__main__':main()
