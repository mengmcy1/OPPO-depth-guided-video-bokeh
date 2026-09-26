"""Build/check an E1–E6 exclusion list and reserved official test source pool.

Identity is the author's raw source, never a crop name or a video frame range.
This checks registered source IDs, not image hashes or baseline training corpora.
"""
import argparse
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('AM-2k', 'BG-20k', 'VideoMatte240K')


def source_id(path):
    path = str(path).replace('\\', '/')
    match = re.search(r'data/raw/AM-2k/(train|validation)/(fg|mask|original|bg|trimap)/([^/]+)\.(png|jpg|jpeg)$', path)
    if match:
        return 'AM-2k:' + match[3]
    match = re.search(r'data/raw/BG-20k/(train|testval|test)/([^/]+)\.(png|jpg|jpeg)$', path)
    if match:
        return 'BG-20k:' + match[2]
    match = re.search(r'data/raw/VideoMatte240K_HEVC/VideoMatte240K/(train|test)/(fgr|pha)/([^/]+)\.mp4$', path)
    if match:
        # The author reuses 0000.mp4 etc. across train and test; keep official namespace.
        return f'VideoMatte240K:{match[1]}:{match[3]}'
    raise ValueError('Missing recognized original-source provenance: ' + path)


def raw_references(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from raw_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from raw_references(item)
    elif isinstance(value, str):
        value = value.replace('\\', '/')
        if any('data/raw/' + d + '/' in value for d in ['AM-2k', 'BG-20k', 'VideoMatte240K_HEVC']):
            try:
                yield source_id(value), value[value.index('data/raw/'):]
            except ValueError:
                if re.search(r'\.(png|jpe?g|mp4|tiff?|bmp)$', value, re.IGNORECASE):
                    raise  # An unrecognized source file must not silently disappear.
                # Directory templates and explanatory text are not used source files.


def collect_development():
    documents = [ROOT/'configs/renderer-pilot.json', ROOT/'configs/renderer-e6.json']
    for pattern in ['renderer-pilot-*', 'renderer-e6-*']:
        for folder in sorted((ROOT/'outputs').glob(pattern)):
            for name in ['metadata.json', 'config.json']:
                documents.extend(sorted(folder.rglob(name)))
    sources = {}
    for document in documents:
        for sid, path in raw_references(json.loads(document.read_text())):
            entry = sources.setdefault(sid, {'source_id':sid, 'raw_paths':set(), 'observed_in':set()})
            entry['raw_paths'].add(path)
            # Preserve concrete provenance, including rejected/replaced input snapshots.
            entry['observed_in'].add(str(document.relative_to(ROOT)))
    entries = [{k:sorted(v) if isinstance(v,set) else v for k,v in e.items()} for _,e in sorted(sources.items())]
    return entries, len(documents)


def build():
    import numpy as np
    import cv2
    from PIL import Image
    entries, count = collect_development()
    excluded = {e['source_id'] for e in entries}
    validation = ROOT/'data/raw/AM-2k/validation'
    originals = {p.stem:p for p in (validation/'original').glob('*.jpg')}
    eligible, rejected = [], []
    for mask in sorted((validation/'mask').glob('*.png')):
        alpha = np.array(Image.open(mask).convert('L'))
        assert mask.stem in originals
        assert Image.open(originals[mask.stem]).size == alpha.shape[::-1]
        if source_id(mask) in excluded:
            reason = 'development_source'
        elif not alpha.any():
            reason = 'empty_foreground'
        elif any(e.any() for e in [alpha[0],alpha[-1],alpha[:,0],alpha[:,-1]]):
            reason = 'source_border_alpha_nonzero'
        else:
            eligible.append(mask.stem)
            continue
        rejected.append({'source_id':'AM-2k:'+mask.stem,'reason':reason})
    video = ROOT/'data/raw/VideoMatte240K_HEVC/VideoMatte240K/test/fgr'
    videos = [p.stem for p in sorted(video.glob('*.mp4')) if source_id(p) not in excluded]
    video_properties = {}
    for name in videos:
        properties = []
        for role in ['fgr','pha']:
            cap = cv2.VideoCapture(str(video.parent/role/(name+'.mp4')))
            assert cap.isOpened()
            properties.append([cap.get(k) for k in [cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS,cv2.CAP_PROP_FRAME_COUNT]])
            cap.release()
        assert properties[0] == properties[1] and properties[0][3] >= 25
        video_properties[name] = dict(zip(['width','height','fps','frames'],properties[0]))
    background = ROOT/'data/raw/BG-20k/testval'
    backgrounds = [p.stem for p in sorted(background.glob('*.jpg')) if source_id(p) not in excluded]
    return {
        'status':'RESERVED_OFFICIAL_TEST_SOURCE_POOL_NOT_FINAL_SCENE_FREEZE',
        'policy':'Prefer official validation/test splits; select by existing source-boundary rule, not model scores',
        'identity_rules':{
            'AM-2k':'Original stem, independent of fg/mask/original/trimap role or extension',
            'BG-20k':'Original author image stem, independent of split or mirror container',
            'VideoMatte240K':'Official split + video stem; fgr/pha share identity; every frame of a source video is excluded together',
            'derived_assets':'Must retain original source_id and raw provenance; a crop name or another frame range never creates a new source',
        },
        'development_metadata_documents':count,
        'development_exclusions':entries,
        'test_pool':{
            'AM-2k':{'split':'validation','source_names':eligible,
                     'original_dir':'data/raw/AM-2k/validation/original','mask_dir':'data/raw/AM-2k/validation/mask',
                     'foreground_rgb_status':'Not supplied in official validation; E7 must estimate foreground colors using original RGB and Alpha. Do not use original RGB as already-separated foreground.'},
            'BG-20k':{'split':'testval','source_names':backgrounds},
            'VideoMatte240K':{'split':'test','source_names':videos,
                             'paired_video_properties':video_properties,
                             'grouping':'Only five official source videos; multiple clips from one video remain the same statistical source group'},
        },
        'AM_validation_ineligible':rejected,
        'remaining_E7_work':['Derive and validate AM foreground RGB without tuning on baseline test scores',
                             'Choose final scene combinations, source-video frame ranges, geometry/focus/K and scene count',
                             'Freeze final scene manifest and scoring configuration; do not treat source-pool reservation as benchmark freeze'],
        'limits':['ID/declared-provenance isolation, not content-deduplication proof',
                  'No claim that pretrained baselines never saw these datasets',
                  'Official source-video namespace trusted; not a person-identity or original-recording-session split'],
    }


def pool_paths(dataset, group, name):
    split = group['split']
    if dataset == 'AM-2k':
        return [f'data/raw/AM-2k/{split}/original/{name}.jpg',f'data/raw/AM-2k/{split}/mask/{name}.png']
    if dataset == 'BG-20k':
        return [f'data/raw/BG-20k/{split}/{name}.jpg']
    return [f'data/raw/VideoMatte240K_HEVC/VideoMatte240K/{split}/{role}/{name}.mp4' for role in ['fgr','pha']]


def check(manifest, local_files=True):
    development = {e['source_id'] for e in manifest['development_exclusions']}
    assert len(development) == len(manifest['development_exclusions']), 'Duplicate development ID'
    observed, documents = collect_development()
    missing = {entry['source_id'] for entry in observed} - development
    assert not missing, 'Development exclusions missing observed sources: '+str(sorted(missing))
    test_ids, counts = set(), {}
    for dataset in DATASETS:
        group = manifest['test_pool'][dataset]
        assert len(group['source_names']) == len(set(group['source_names']))
        counts[dataset] = len(group['source_names'])
        for name in group['source_names']:
            paths = pool_paths(dataset,group,name)
            sid = source_id(paths[0])
            assert all(source_id(path)==sid for path in paths), 'Pair provenance mismatch'
            assert sid not in development, 'Development/test overlap: '+sid
            assert sid not in test_ids
            test_ids.add(sid)
            if local_files:
                assert all((ROOT/path).is_file() for path in paths), 'Missing source pair: '+sid
    return {'development_counts':{d:sum(s.startswith(d+':') for s in development) for d in DATASETS},
            'test_pool_counts':counts,'intersection_count':len(development & test_ids),
            'development_documents_rechecked':documents,'missing_observed_development_sources':[],
            'local_paths_checked':local_files,'source_id_overlap_pass':True}


def regression_checks(manifest):
    # A whole-video overlap must fail even if a different segment would be selected later.
    changed = copy.deepcopy(manifest)
    changed['test_pool']['VideoMatte240K'].update(split='train',source_names=['0000'],frame_range=[100,124])
    try:
        check(changed,False)
    except AssertionError as error:
        assert 'overlap' in str(error)
    else:
        raise AssertionError('A development video was accepted as test')
    for dataset, split, name in [('AM-2k','train','m_00b28969'),('BG-20k','testval','h_0006a281')]:
        changed = copy.deepcopy(manifest)
        changed['test_pool'][dataset].update(split=split,source_names=[name])
        try:
            check(changed,False)
        except AssertionError as error:
            assert 'overlap' in str(error)
        else:
            raise AssertionError('A development image was accepted as test')
    changed = copy.deepcopy(manifest)
    changed['development_exclusions'] = [e for e in changed['development_exclusions'] if e['source_id']!='AM-2k:m_00b28969']
    try:
        check(changed,False)
    except AssertionError as error:
        assert 'missing observed sources' in str(error)
    else:
        raise AssertionError('Omission of a replaced development source was accepted')
    assert source_id('data/raw/AM-2k/train/fg/m_00b28969.png') == source_id('data/raw/AM-2k/train/original/m_00b28969.jpg')
    assert source_id('data/raw/VideoMatte240K_HEVC/VideoMatte240K/train/fgr/0000.mp4') == source_id('data/raw/VideoMatte240K_HEVC/VideoMatte240K/train/pha/0000.mp4')
    assert source_id('data/raw/VideoMatte240K_HEVC/VideoMatte240K/train/fgr/0000.mp4') != source_id('data/raw/VideoMatte240K_HEVC/VideoMatte240K/test/fgr/0000.mp4')
    try:
        source_id('outputs/renamed_crop/new_name.png')
    except ValueError:
        pass
    else:
        raise AssertionError('Unproven crop identity accepted')
    return {'development_video_rejected':True,'AM_roles_share_identity':True,
            'different_segment_same_video_rejected':True,'development_AM_and_BG_images_rejected':True,
            'omitted_replaced_source_rejected':True,
            'foreground_alpha_video_share_identity':True,'official_split_name_collision_handled':True,
            'unknown_derived_path_rejected':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=ROOT/'configs/benchmark-source-isolation.json')
    p.add_argument('--build',action='store_true')
    p.add_argument('--ids-only',action='store_true')
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    manifest=build() if a.build else json.loads(a.manifest.read_text())
    report=check(manifest,not a.ids_only)
    report['regression_checks']=regression_checks(manifest)
    if a.build:
        with a.manifest.open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    with a.report.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
