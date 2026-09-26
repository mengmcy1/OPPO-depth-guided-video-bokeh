"""One-off official public-file download using curl's native retry/resume."""
import argparse,json,subprocess,concurrent.futures
from pathlib import Path
from PIL import Image
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',type=Path,action='append',required=True)
parser.add_argument('--error-log',type=Path,required=True)
parser.add_argument('--workers',type=int,default=8)
args=parser.parse_args()
files=[]
for manifest in args.manifest:files+=json.loads(manifest.read_text())
files.sort(key=lambda x:(0 if 'AM-2k' in x['local_path'] else 1, 0 if '/fg/' in x['path'] or '/mask/' in x['path'] else 1, x['path']))
def fetch(x):
 p=(root/x['local_path']).resolve()
 if not p.is_relative_to((root/'data/raw').resolve()):raise ValueError('Destination must be under data/raw')
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():return None
 partial=p.with_name(p.name+'.part')
 cmd=['curl','-fLsS','--retry','3','--retry-all-errors','--retry-delay','3','--connect-timeout','20','--max-time','240','-C','-','https://drive.google.com/uc?export=download&id='+x['id'],'-o',str(partial)]
 r=subprocess.run(cmd,capture_output=True,text=True)
 if r.returncode:return (x['path'],r.stderr[-300:])
 try:
  if p.suffix.lower() in ['.jpg','.png','.jpeg']:
   with Image.open(partial) as im:im.verify()
  partial.rename(p)
 except Exception as e:return (x['path'],str(e))
 return None
errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
 for i,e in enumerate(pool.map(fetch,files),1):
  if e:errors.append(e);print('FAILED',e,flush=True)
  if i%100==0:print('Completed',i,'/',len(files),'errors',len(errors),flush=True)
args.error_log.parent.mkdir(parents=True,exist_ok=True)
args.error_log.write_text(json.dumps(errors,indent=2))
print('DONE',len(files),'errors',len(errors),flush=True)
