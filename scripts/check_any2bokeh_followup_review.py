"""Check follow-up tools at the user's HTTP root, using isolated test storage.

No model execution, perceptual judgments, or access to the user's profile.
The HTTP server is supplied by the caller and is never stopped here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

from check_any2bokeh_review_pages import (
    ROOT, VISUAL, fixture_record, form_preservation, inspect_layout,
    roi_select, select_frame,
)


def load(path):
    return json.loads((ROOT / path).read_text())


def page_config(path):
    text = path.read_text()
    match = re.search(r'<script id="atb-viewer-config" type="application/json">(.*?)</script>', text, re.S)
    assert match, path
    return json.loads(match[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='Existing material-root HTTP URL')
    parser.add_argument('--browser-executable', required=True, help='H.264-capable Chrome')
    args = parser.parse_args()
    base = args.base_url.rstrip('/') + '/'
    c = load('configs/any2bokeh-followup-review-20261014.json')
    output = ROOT / c['output']
    prefix = c['record_prefix']
    materials = load(prefix + '-batch2-materials.json')
    samples = load(prefix + '-batch2-samples.json')['rows']
    b19 = load(prefix + '-B19-input.json')
    log_dir = ROOT / 'logs/any2bokeh-followup-20261014/browser-check'
    log_dir.mkdir(parents=True, exist_ok=True)

    def url(path):
        return base + (ROOT / path).relative_to(output).as_posix()

    def head(address):
        with urlopen(Request(address, method='HEAD'), timeout=30) as response:
            assert response.status == 200, address

    http_checks = 0
    for relative in ['review-tools/viewer.js', 'review-tools/viewer.css', 'review-tools/review-form.js',
                     c['B19']['directory'] + '/', c['B19']['directory'] + '/review.html',
                     c['B19']['directory'] + '/ROI-statistics.csv',
                     c['B19']['directory'] + '/method-mapping.json',
                     c['batch2']['directory'] + '/', c['batch2']['directory'] + '/review.html',
                     'B1/B19/', 'B1/review.html', 'B2/B19/', 'B2/review.html']:
        head(base + relative); http_checks += 1
    scenes = [(ROOT / item['index'], url(item['index'])) for item in materials['items']]
    scenes += [(output / c['B19']['directory'] / kind / 'index.html',
                base + c['B19']['directory'] + '/' + kind + '/') for kind in ['input', 'methods']]
    for path, address in scenes:
        head(address); http_checks += 1
        config = page_config(path)
        assert config['frames'] == 25 and config['width'] == 1024 and config['height'] == 576
        if path.parent.name == 'methods':
            assert [p['label'] for p in config['panels'][1:]] == [
                next(k for k,v in b19['original_anonymous_mapping'].items() if v == method)
                for method in c['B19']['comparison_order']]
        for panel in config['panels']:
            for t in range(25):
                head(urljoin(address, panel['prefix'] + f'/{t:05d}.png')); http_checks += 1
    print('HTTP_ACTUAL_ROOT_PASS', http_checks, flush=True)

    errors = []; checks = []
    old_samples = load('03_实验记录/records/any2bokeh-failure-analysis-20261009-B-samples.json')['rows']
    old_sample = next(s for s in old_samples if s['sample_id'] == 'B19')
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=args.browser_executable, args=['--disable-gpu'])
        version = browser.version
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
        page = context.new_page(); page.on('pageerror', lambda error: errors.append(str(error)))
        for sample_id in ['B25', 'B36']:
            item = next(i for i in materials['items'] if i['sample_id'] == sample_id)
            item = {**item, 'labels': ['Input RGB', 'Frozen ATB']}
            checks.append(inspect_layout(page, item, url(item['index']), errors))
            print('BATCH2_LAYOUT_PASS', sample_id, flush=True)
        old_item = dict(package='B1', dataset='DAVIS', sample_id='B19', labels=['Input RGB', 'Frozen ATB'])
        checks.append(inspect_layout(page, old_item, base + 'B1/B19/', errors))
        preservation = form_preservation(page, old_sample, base + 'B1/B19/', log_dir)
        old_keys = ['atb-failure-B1-user-20261009', 'atb-failure-B2-user-20261009']
        # Seed synthetic B2 data as well; never read the user's real profile.
        page.evaluate('([k,v])=>localStorage.setItem(k,JSON.stringify(v))',
                      [old_keys[1], [{'test_only': True, 'notes': '__OLD_B2_FIXTURE__', 'unknown': [1, 2]}]])
        old_storage = {key: page.evaluate('(k)=>localStorage.getItem(k)', key) for key in old_keys}
        custom_key = c['batch2']['review_storage_key']
        sample = samples[0]
        existing = {**fixture_record(sample), 'batch': sample['batch'], 'sampling_channel': sample['sampling_channel']}
        page.evaluate('([k,v])=>localStorage.setItem(k,JSON.stringify(v))', [custom_key, [existing]])
        item = next(i for i in materials['items'] if i['sample_id'] == sample['sample_id'])
        page.goto(url(item['index'])); page.wait_for_selector('#atb-viewer[data-frame="0"]')
        select_frame(page, 12)
        page.locator('[data-mode="adjacent"]').click(); page.wait_for_selector('#atb-viewer[data-mode="adjacent"]')
        roi_select(page); page.locator('[data-rate="0.25"]').click(); page.locator('#png-note').click()
        page.wait_for_selector('#context-banner:not([hidden])')
        assert page.evaluate('rows()') == [existing]
        page.locator('#import-context').click()
        added = page.evaluate('rows()[1]')
        assert added['frame_id_or_transition_id'] == '11→12' and added['ROI_xywh'] == '100,120,200,120'
        assert added['ROI_selection_origin'] == 'user_manual_coordinates_semantics_unconfirmed'
        assert added['diagnostic_ROI_id'] == ''
        assert added['actual_playback_rate'] == '' and added['video_playback_rate_setting'] == .25
        assert added['batch'] == sample['batch'] and added['sampling_channel'] == sample['sampling_channel']
        assert all(added[k] == '待用户复核' for k in VISUAL) and added['user_observation'] == ''
        names = []
        for i in range(2):
            with page.expect_download() as event: page.locator('#export-records').click()
            download = event.value; target = log_dir / f'batch2-isolated-fixture-{i}.json'
            download.save_as(target); names.append(download.suggested_filename)
            assert json.loads(target.read_text())[0] == existing
        assert names[0] != names[1]
        page.reload(); assert page.evaluate('rows()') == [existing, added]
        assert all(page.evaluate('(k)=>localStorage.getItem(k)', key) == value for key, value in old_storage.items())

        # Diagnostic preset provenance must remain distinct from a user-selected ROI.
        page.goto(base + c['B19']['directory'] + '/input/')
        page.wait_for_selector('#atb-viewer[data-frame="0"]')
        assert page.locator('#atb-viewer').get_attribute('data-roi') == 'null'
        page.locator('#diagnostic-roi-select').select_option('2')
        assert json.loads(page.locator('#atb-viewer').get_attribute('data-roi')) == c['B19']['diagnostic_rois'][2]['xywh']
        page.locator('#png-note').click(); page.wait_for_selector('#context-banner:not([hidden])')
        assert '执行方诊断框' in page.locator('#context-description').inner_text()
        page.locator('#import-context').click()
        last = page.evaluate('rows().at(-1)')
        assert last['ROI_selection_origin'] == 'executor_diagnostic_preset_not_user_confirmed'
        assert last['diagnostic_ROI_id'] == c['B19']['diagnostic_rois'][2]['id']
        assert all(last[k] == '待用户复核' for k in VISUAL)
        assert all(page.evaluate('(k)=>localStorage.getItem(k)', key) == value for key, value in old_storage.items())
        page.goto(base + c['B19']['directory'] + '/input/')
        page.wait_for_selector('#atb-viewer[data-frame="0"]')
        page.locator('#diagnostic-roi-select').select_option('2'); roi_select(page)
        assert page.locator('#diagnostic-roi-select').input_value() == ''
        page.locator('#clear-roi').click()
        assert page.locator('#atb-viewer').get_attribute('data-roi') == 'null'
        assert page.locator('#diagnostic-roi-select').input_value() == ''
        assert not errors, errors
        context.close(); browser.close()

    report = load(prefix + '-verification.json')
    report.update(status='材料已准备，等待用户B19 B2及第二批B1反馈',
                  browser_checked_utc=datetime.now(timezone.utc).isoformat(),
                  browser_function_checks=dict(actual_HTTP_root=base, material_directory=c['output'],
                    HTTP_200_checks=http_checks, scene_configs_checked=len(scenes), browser='Chrome ' + version,
                    Playwright='1.55.0', layouts=checks, first_batch_storage_checks=preservation,
                    batch2_separate_key=custom_key, old_B1_B2_keys_unchanged=True,
                    batch2_append_context_reload_and_distinct_exports=True,
                    batch_and_sampling_channel_preserved=True, manual_ROI_provenance=True,
                    B19_preset_provenance_and_default_empty=True, all_visual_fields_pending=True,
                    B19_manual_override_and_clear_reset_preset=True,
                    javascript_errors=errors, isolated_fixture_profile_only=True,
                    user_client_proxy_path_reproduced=False),
                  B19_prior_HTTP_check=b19.get('HTTP_check'),
                  limitations=['Input proxies do not establish semantic coverage; user confirmation pending',
                    'No paired DAVIS bokeh GT or metric-depth truth; no causal or perceptual conclusion',
                    'Historical malformed GET requests were not reproduced by standard local requests',
                    'No automatic merging of chat records and browser exports; no held-out review'],
                  completed=['程序检查完成', 'CPU输入核查与抽样完成', '材料生成完成'],
                  pending=['用户B19 B2视觉复核', '用户第二批B1独立观察及语义标签确认'])
    (ROOT / (prefix + '-verification.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('FOLLOWUP_HTTP_BROWSER_CHECKS_PASSED', flush=True)


if __name__ == '__main__': main()
