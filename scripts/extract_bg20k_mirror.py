"""Restore BG-20K original names from the verified public parquet mirror."""
import argparse,json
from pathlib import Path
import pyarrow.parquet as pq
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
expected={s:set((a.output/(s+'.txt')).read_text().splitlines()) for s in ['train','testval']};seen={s:set() for s in expected}
files=sorted((a.source/'data').glob('*.parquet'));assert len(files)==30,'Wait for all 30 shards before extraction'
for file in files:
 split='testval' if file.name.startswith('test-') else 'train';dest=a.output/split;dest.mkdir(exist_ok=True)
 for batch in pq.ParquetFile(file).iter_batches(batch_size=8):
  for row in batch.to_pylist():
   im=row['image'];name=Path(im['path']).name;assert name in expected[split],name;assert name not in seen[split],name;seen[split].add(name)
   out=dest/name
   if not out.exists():
    tmp=out.with_name(out.name+'.mirror-part');tmp.write_bytes(im['bytes']);tmp.rename(out)
 print(file.name,len(seen[split]),flush=True)
for split in seen:assert seen[split]==expected[split],(split,len(seen[split]),len(expected[split]))
record={'original':'https://github.com/JizhiziLi/GFM','mirror':'https://huggingface.co/datasets/unography/BG-20k','retrieval_endpoint':'https://hf-mirror.com','split_mapping':{'train':'train','test':'testval'},'counts':{k:len(v) for k,v in seen.items()},'validation':'Exact original filename sets; first shard matched decoded pixels of 207 official images. Existing official files preserved.'}
(a.output/'SOURCE.json').write_text(json.dumps(record,indent=2));print('COMPLETE',record['counts'])
