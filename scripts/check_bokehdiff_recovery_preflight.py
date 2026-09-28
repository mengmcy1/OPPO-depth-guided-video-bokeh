"""Check both complete BokehDiff preflight sequences and their matched RNG streams."""
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/bokehdiff-recovery-preflight-20260928'


def main():
    protocol = json.loads((ROOT / 'configs/bokehdiff-adapter-recovery.json').read_text())
    ids = [s['id'] for s in json.loads((ROOT / protocol['input'] / 'config.json').read_text())['scenes']
           if s['id'] in protocol['preflight_scenes']]
    results, states = {}, {}
    for candidate in protocol['candidates']:
        folder = RUN / candidate
        complete = json.loads((folder / 'complete.json').read_text())
        assert complete['frames'] == 200 and complete['stage'] == 'preflight'
        rp = json.loads((folder / 'protocol.json').read_text())
        assert rp['protocol'] == protocol and rp['scene_ids'] == ids
        loading = json.loads((folder / 'model-loading.json').read_text())
        assert loading['LoRA_parameter_count'] > 0 and len(loading['PISA_processors']) > 0 and not loading['vae_unexpected']
        rows = []
        states[candidate] = []
        for sid in ids:
            paths = sorted((folder / 'clips' / sid / 'pred_bokehdiff').glob('*.png'))
            assert [x.name for x in paths] == [f'{t:05d}.png' for t in range(25)]
            for t, path in enumerate(paths):
                with Image.open(path) as im:
                    assert im.size == (1024, 576) and im.mode == 'RGB'
                    im.verify()
                audit = json.loads((folder / 'timing' / sid / f'{t:05d}.json').read_text())
                assert audit['scene'] == sid and audit['frame'] == t and audit['seed'] == 42
                assert not audit['reset_global_seed'] and audit['quantization'] == 'round_png8'
                assert audit['geometry']['candidate'] == candidate
                assert audit['output_shape'] == [1, 3, 576, 1024]
                assert np.isfinite(audit['decoded_range']).all() and audit['output_range'][0] >= 0 and audit['output_range'][1] <= 1
                assert audit['geometry']['mapping_error_float64'] < 1e-10
                assert abs(audit['defocus_condition_range'][1] - audit['geometry']['max_defocus_condition']) < 1e-5
                rows.append(audit)
            states[candidate].append(torch.load(folder / 'timing' / sid / 'rng-after.pt', weights_only=True))
        results[candidate] = dict(frames=len(rows), peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in rows),
            median_seconds_concurrent=statistics.median(x['seconds'] for x in rows),
            max_mapping_error_float64=max(x['geometry']['mapping_error_float64'] for x in rows),
            input_d1_frames=sum(x['disparity_equal_one_fraction'] > 0 for x in rows),
            focus_outside_01_frames=sum(not 0 <= x['focus'] <= 1 for x in rows),
            by_K_max_defocus_condition={str(k): max(x['defocus_condition_range'][1] for x in rows if x['K'] == k) for k in (4, 8, 16, 32)},
            PISA_processors=loading['PISA_processors'], LoRA_parameter_count=loading['LoRA_parameter_count'],
            PNG_counts_dimensions_modes_integrity=True, all_output_ranges_finite=True)
    for a, b in zip(states['shared_geometry'], states['normalized']):
        assert a['next_scene_index'] == b['next_scene_index']
        assert torch.equal(a['cpu'], b['cpu']) and torch.equal(a['cuda'], b['cuda']), 'Candidate RNG streams diverged'
    report = dict(status='NUMERICAL_PREFLIGHT_PASSED_VISUAL_REVIEW_PENDING', protocol=protocol,
        candidates=results, matched_CPU_CUDA_RNG_after_all_8_scenes=True,
        stochastic_evidence='03_实验记录/records/bokehdiff-stochastic-review-20260928.json',
        scope='No GT metrics or paper-score selection used for preflight; PNGs cannot be reused in full scene order.')
    with (RUN / 'checks.json').open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
