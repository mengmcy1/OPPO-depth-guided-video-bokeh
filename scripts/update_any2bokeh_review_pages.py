"""Upgrade existing review HTML only; preserve media, user exports and old HTML.

The same URLs and B1 localStorage key remain in use. No model/metric code imported.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from review.any2bokeh_pages import ASSETS, enhance_scene_html, enhance_index_html, render_review_form


def media_state(output, items):
    files = set()
    for item in items:
        folder = ROOT / item['index']; folder = folder.parent
        files.update(p for p in folder.iterdir() if p.suffix in ['.mp4', '.png'])
        files.update((folder / 'originals').rglob('*.png'))
    files.update((output / 'display-input').rglob('*.png'))
    # Metadata checks only; no image interpretation or expensive content hashing.
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino, str(p.resolve())) for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/any2bokeh-failure-analysis-20261009.json')
    parser.add_argument('--date', default='20261013')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    manifest_path = config['record_prefix'] + '-materials.json'
    manifest = json.loads((ROOT / manifest_path).read_text())
    samples = json.loads((ROOT / (config['record_prefix'] + '-B-samples.json')).read_text())['rows']
    assert all(s['split'] == 'exploration' for s in samples)
    output = ROOT / config['output']
    previous_media = media_state(output, manifest['items'])
    history = output / 'page-history' / args.date
    changed, preserved = [], []

    def write(path, content):
        data = content.encode('utf-8') if isinstance(content, str) else content
        if path.exists() and path.read_bytes() == data:
            return
        if path.exists():
            backup = history / path.relative_to(output)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                # Preserve both the initial page and any later revision during this task.
                if backup.read_bytes() != path.read_bytes():
                    stamp = datetime.now(timezone.utc).strftime('%H%M%S%f')
                    backup = backup.with_name(backup.name + '.prior-' + stamp)
                    backup.write_bytes(path.read_bytes())
            else:
                backup.write_bytes(path.read_bytes())
            preserved.append(str(backup.relative_to(ROOT)))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.ui-update.tmp')
        assert not tmp.exists(), f'Inspect incomplete HTML update: {tmp}'
        tmp.write_bytes(data); tmp.replace(path)
        changed.append(str(path.relative_to(ROOT)))

    for source in ASSETS.iterdir():
        write(output / 'review-tools' / source.name, source.read_bytes())
    for item in manifest['items']:
        assert item['split'] == 'exploration'
        path = ROOT / item['index']
        write(path, enhance_scene_html(path.read_text(), item))
    for package in ['B1', 'B2']:
        path = output / package / 'index.html'
        write(path, enhance_index_html(path.read_text(), package))
        write(output / package / 'review.html', render_review_form(samples, package))
    assert previous_media == media_state(output, manifest['items']), 'Media metadata changed during HTML update'
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), status='PAGES_UPDATED_FUNCTION_CHECK_PENDING',
        source_material_manifest=manifest_path, starting_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scene_pages=len(manifest['items']), packages=['B1','B2'], changed_files=changed, preserved_HTML=preserved,
        media_metadata_unchanged=True, media_entries_checked=len(previous_media),
        B1_storage_key='atb-failure-B1-user-20261009', B2_storage_key='atb-failure-B2-user-20261009',
        previous_exports_touched=False, user_visual_fields_filled=False,
        features=['playbackRate1/0.5/0.25 only','PNG frame0–24, keyboard and native-size view',
                  '25 thumbnail entries and exact adjacent0→1 through23→24','manual shared pixel-coordinate ROI',
                  'append location metadata to form on explicit user action, timestamped exports'],
        B2_status='Still waiting for user B1 marks; no visual review or held-out access')
    record = ROOT / '03_实验记录/records' / f'any2bokeh-review-ui-update-{args.date}.json'
    if record.exists():
        prior = json.loads(record.read_text())
        if changed:
            result['changed_files'] = sorted(set(prior['changed_files'] + changed))
            result['preserved_HTML'] = prior['preserved_HTML'] + preserved
            result['initial_update_utc'] = prior.get('initial_update_utc', prior['updated_utc'])
            with record.open('w') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
            print('UI_HTML_UPDATED',len(changed),'MEDIA_UNCHANGED',len(previous_media))
        else:print('UI_PAGES_ALREADY_CURRENT', record)
    else:
        with record.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
        print('UI_HTML_UPDATED', len(changed), 'MEDIA_UNCHANGED', len(previous_media))


if __name__ == '__main__': main()
