"""Verify saved BokehDiff progress; --start resumes the detached inference and evaluation jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/bokehdiff-adapter-recovery-20260928'
LOG = ROOT / 'logs/bokehdiff-recovery-20260928'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='Without this flag, only check saved files on CPU')
    args = parser.parse_args()
    os.chdir(ROOT)
    assert not (ROOT / '03_实验记录/records/bokehdiff-adapter-recovery-200-20260928.json').exists(), 'Already summarized; do not duplicate jobs'
    protocol = json.loads((ROOT / 'configs/bokehdiff-adapter-recovery.json').read_text())
    assert json.loads((RUN / 'protocol.json').read_text()) == protocol
    previous = json.loads((RUN / 'server-jobs.json').read_text())
    for pid in list(previous['inference'].values()) + [previous['evaluation_pid'], previous['summary_pid']]:
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists():
            text = command.read_bytes()
            assert not any(name in text for name in [b'run_bokehdiff_recovery.py', b'evaluate_bokehdiff_recovery.py', b'summarize_bokehdiff_recovery.py']), f'Existing recovery job is still alive: {pid}'
    checked = {}
    for candidate in protocol['candidates']:
        folder = RUN / candidate
        info = json.loads((folder / 'protocol.json').read_text())
        assert info['protocol'] == protocol and info['candidate'] == candidate and info['stage'] == 'full'
        ids = info['scene_ids']
        prefix = 0
        while prefix < len(ids) and (folder / 'timing' / ids[prefix] / 'complete.json').exists():
            timing = folder / 'timing' / ids[prefix]
            assert json.loads((timing / 'complete.json').read_text())['frames'] == 25
            assert len(list(timing.glob('[0-9][0-9][0-9][0-9][0-9].json'))) == 25
            assert len(list((folder / 'clips' / ids[prefix] / 'pred_bokehdiff').glob('*.png'))) == 25
            assert (timing / 'rng-after.pt').is_file()
            prefix += 1
        assert prefix > 0 and not any((folder / 'timing' / sid / 'complete.json').exists() for sid in ids[prefix:])
        checkpoint = folder / 'timing' / ids[prefix - 1] / 'rng-after.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['next_scene_index'] == prefix
        assert state['cpu'].dtype == state['cuda'].dtype == torch.uint8
        assert state['cpu'].numel() > 0 and state['cuda'].numel() > 0
        for path in (folder / 'clips' / ids[prefix - 1] / 'pred_bokehdiff').glob('*.png'):
            with Image.open(path) as im:
                assert im.size == (1024, 576) and im.mode == 'RGB'
                im.verify()
        reports = sorted((folder / 'metrics').glob('batch_*.json'))
        for path in reports:
            value = json.loads(path.read_text())
            batch = int(path.stem.split('_')[1])
            assert [row['scene'] for row in value['rows']] == ids[batch * 10:(batch + 1) * 10]
            assert value['prediction_root'] == str((folder / 'clips').relative_to(ROOT))
            assert value['prediction_dir'] == 'pred_bokehdiff'
        checked[candidate] = dict(completed_scenes=prefix, completed_frames=prefix * 25,
            next_scene=ids[prefix] if prefix < len(ids) else None, metric_batches=len(reports),
            checkpoint=str(checkpoint.relative_to(ROOT)))
    print(json.dumps(checked, indent=2), flush=True)
    if not args.start:
        print('CHECK_ONLY_PASSED; no job started and no GPU used', flush=True)
        return
    subprocess.run(['nvidia-smi'], check=True)
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    assert free >= 56 * 1024, 'Reserve measured model/evaluator memory before restarting both candidates'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = LOG / f'resume-{stamp}'
    logs.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               CUDA_HOME='/usr/local/cuda-12.8', PATH='/usr/local/cuda-12.8/bin:' + os.environ['PATH'])

    def spawn(python, command, path):
        with path.open('x') as log:
            proc = subprocess.Popen([python, '-u', *command], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return proc.pid

    jobs = dict(started_utc=datetime.now(timezone.utc).isoformat(), resumed_from=checked, inference={})
    for candidate in protocol['candidates']:
        jobs['inference'][candidate] = spawn('/root/autodl-tmp/envs/bokehdiff/bin/python',
            ['scripts/run_bokehdiff_recovery.py', '--stage', 'full', '--candidate', candidate,
             '--output', str(RUN.relative_to(ROOT))], logs / f'inference-{candidate}.log')
    temp = RUN / 'server-jobs.next.json'
    temp.write_text(json.dumps(jobs, indent=2) + '\n')
    os.replace(temp, RUN / 'server-jobs.json')
    evaluation_log = logs / 'evaluation.log'
    jobs['evaluation_pid'] = spawn('/root/miniconda3/envs/magicbokeh/bin/python',
                                   ['scripts/evaluate_bokehdiff_recovery.py'], evaluation_log)
    jobs['summary_pid'] = spawn('/root/miniconda3/envs/magicbokeh/bin/python',
        ['scripts/summarize_bokehdiff_recovery.py', '--wait', '--evaluation-pid', str(jobs['evaluation_pid']),
         '--completion-log', str(evaluation_log.relative_to(ROOT))], logs / 'summary.log')
    jobs['logs'] = str(logs.relative_to(ROOT))
    temp.write_text(json.dumps(jobs, indent=2) + '\n')
    os.replace(temp, RUN / 'server-jobs.json')
    print('RESUMED', json.dumps(jobs), flush=True)


if __name__ == '__main__':
    main()
