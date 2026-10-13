"""Functional browser checks only, in a fresh profile; no perceptual judgments.

Run with isolated Playwright+Chromium. Test storage/downloads are synthetic
fixtures, never the user's browser profile or exported review files.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
from threading import Thread
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from review.any2bokeh_pages import FIELDS, VISUAL


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args): pass


def select_frame(page, frame):
    page.locator('#frame-slider').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}', frame)
    page.wait_for_selector(f'#atb-viewer[data-frame="{frame}"]')
    page.wait_for_function('!document.getElementById("png-note").disabled')


def roi_select(page):
    page.locator('#roi-enabled').check()
    page.locator('.atb-surface').first.scroll_into_view_if_needed()
    bounds = page.locator('.atb-surface img').first.bounding_box()
    page.mouse.move(bounds['x']+100*bounds['width']/1024, bounds['y']+120*bounds['height']/576)
    page.mouse.down()
    page.mouse.move(bounds['x']+300*bounds['width']/1024, bounds['y']+240*bounds['height']/576,steps=6)
    page.mouse.up()
    roi = json.loads(page.locator('#atb-viewer').get_attribute('data-roi'))
    assert roi == [100,120,200,120], roi
    assert all(json.loads(r)==roi for r in page.locator('.atb-crop canvas').evaluate_all('(els)=>els.map(e=>e.dataset.roi)'))
    return roi


def inspect_layout(page, item, url, errors):
    page.goto(url)
    page.wait_for_selector('#atb-viewer[data-frame="0"]')
    count = len(item['labels'])
    assert page.locator('#all-frames .atb-thumbnail').count()==25
    assert page.locator('.atb-frame-image').count()==count
    assert '24→0' in page.locator('#loop-warning').inner_text()
    for speed in [1,.5,.25]:
        page.locator(f'[data-rate="{speed:g}"]').click()
        assert page.locator('video').evaluate('(v)=>v.playbackRate')==speed
        assert f'{speed:g}×' in page.locator('#rate-label').inner_text()
    # Actual native playback advances; no new frames or encodings are generated.
    page.locator('video').evaluate('(v)=>v.play()')
    page.wait_for_function('document.querySelector("video").currentTime>0')
    page.locator('video').evaluate('(v)=>v.pause()')
    video_time = page.locator('video').evaluate('(v)=>v.currentTime')
    assert abs(page.locator('video').evaluate('(v)=>v.duration')-1.25)<.001
    select_frame(page,12)
    assert page.locator('.atb-frame-image').evaluate_all('(els)=>els.every(e=>e.dataset.frame==="12"&&e.src.endsWith("00012.png")&&e.naturalWidth===1024&&e.naturalHeight===576)')
    assert page.locator('video').evaluate('(v)=>v.currentTime')==video_time
    page.locator('#frame-status').click()
    page.keyboard.press('ArrowRight');page.wait_for_selector('#atb-viewer[data-frame="13"]')
    page.keyboard.press('ArrowLeft');page.wait_for_selector('#atb-viewer[data-frame="12"]')
    select_frame(page,24);page.locator('#frame-status').click();page.keyboard.press('ArrowRight')
    assert page.locator('#atb-viewer').get_attribute('data-frame')=='24'
    select_frame(page,0);page.locator('#frame-status').click();page.keyboard.press('ArrowLeft')
    assert page.locator('#atb-viewer').get_attribute('data-frame')=='0'
    select_frame(page,12);page.locator('[data-mode="adjacent"]').click()
    page.wait_for_selector('#atb-viewer[data-mode="adjacent"]')
    assert page.locator('#transition-label').inner_text()=='11→12'
    assert page.locator('.atb-frame-image').count()==count*2
    assert page.locator('.atb-frame-image').evaluate_all('(els)=>els.every((e,i)=>Number(e.dataset.frame)===[11,12][i%2]&&e.src.endsWith(String([11,12][i%2]).padStart(5,"0")+".png"))')
    page.locator('[data-size="native"]').click()
    assert page.locator('.atb-frame-image').first.bounding_box()['width']==1024
    page.locator('[data-size="double"]').click()
    assert page.locator('.atb-frame-image').first.bounding_box()['width']==2048
    page.locator('[data-size="fit"]').click();roi_select(page)
    mismatches=page.locator('.atb-surface').evaluate_all('''els=>els.map(surface=>{
      const img=surface.querySelector('img'), crop=surface.parentElement.querySelector('canvas');
      const ref=document.createElement('canvas');ref.width=1024;ref.height=576;
      const ctx=ref.getContext('2d');ctx.drawImage(img,0,0);
      const a=ctx.getImageData(100,120,200,120).data,b=crop.getContext('2d').getImageData(0,0,200,120).data;
      let different=0;for(let i=0;i<a.length;i++)different+=a[i]!==b[i];return different;
    })''')
    assert all(v==0 for v in mismatches), mismatches
    page.locator('#clear-roi').click();assert page.locator('#atb-viewer').get_attribute('data-roi')=='null'
    page.locator('[data-mode="frame"]').click();page.wait_for_selector('#atb-viewer[data-mode="frame"]')
    page.locator('.atb-thumbnail[data-frame="7"] button').click();page.wait_for_selector('#atb-viewer[data-frame="7"]')
    assert page.locator('.atb-thumbnail[data-frame="7"] a').count()==count
    # Fast slider changes cannot commit an old asynchronous PNG request.
    page.locator('#frame-slider').evaluate('(e)=>{for(const v of [22,4]){e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}}')
    page.wait_for_selector('#atb-viewer[data-frame="4"]');page.wait_for_function('!document.getElementById("png-note").disabled')
    assert page.locator('.atb-frame-image').evaluate_all('(els)=>els.every(e=>e.dataset.frame==="4")')
    page.evaluate('()=>{let e=document.createElement("textarea");e.id="keyboard-edit-fixture";document.body.append(e);e.focus()}')
    page.keyboard.press('ArrowRight');assert page.locator('#atb-viewer').get_attribute('data-frame')=='4'
    assert not errors, errors
    return dict(package=item['package'],dataset=item['dataset'],sample_id=item['sample_id'],panels=count,
        playback_rates_checked=[1,.5,.25],actual_native_playback=True,PNG_sync_keyboard_boundaries=True,
        adjacent_11_12=True,ROI_xywh=[100,120,200,120],ROI_native_pixel_mismatches=mismatches,
        native100_200percent=True,thumbnails25=True,rapid_navigation_race_checked=True,MP4_not_seeked_by_PNG_viewer=True)


def fixture_record(sample):
    pending=set(VISUAL+['severity','confidence','normal_motion_or_reasonable_defocus','similar_in_input_or_GT'])
    base={k:sample[k] for k in ['sample_id','dataset','scene_id','source_group','split']}
    base.update(package='B1',**{k:'待用户复核' if k in pending else '' for k in FIELDS if k!='actual_playback_rate'})
    base.update(frame_id_or_transition_id='7',user_observation='__UI_TEST_EXISTING_TEXT__',notes='__UI_TEST_EXISTING_NOTES__',extra_user_field={'preserve':True})
    return base


def form_preservation(page, sample, scene_url, log_dir):
    key='atb-failure-B1-user-20261009';original=fixture_record(sample)
    page.goto(urljoin(scene_url,'../review.html'));page.wait_for_selector('[data-kind="review"]')
    page.evaluate('([k,r])=>localStorage.setItem(k,JSON.stringify([r]))',[key,original])
    page.goto(scene_url);page.wait_for_selector('#atb-viewer[data-frame="0"]')
    select_frame(page,12);page.locator('[data-mode="adjacent"]').click();page.wait_for_selector('#atb-viewer[data-mode="adjacent"]')
    roi_select(page);page.locator('[data-rate="0.25"]').click();page.locator('#png-note').click()
    page.wait_for_selector('#context-banner:not([hidden])')
    assert page.evaluate('(k)=>JSON.parse(localStorage.getItem(k))',key)==[original]
    assert page.evaluate('rows()')==[original]
    page.locator('#import-context').click()
    data=page.evaluate('(k)=>JSON.parse(localStorage.getItem(k))',key)
    assert data[0]==original and len(data)==2
    added=data[1];assert added['frame_id_or_transition_id']=='11→12' and added['ROI_xywh']=='100,120,200,120'
    assert added['actual_view_mode']=='相邻帧PNG' and added['actual_playback_rate']=='' and added['video_playback_rate_setting']==.25
    assert all(added[k]=='待用户复核' for k in VISUAL) and added['user_observation']==''
    page.locator('[data-kind="review"]').nth(1).locator('[data-key="user_observation"]').fill('__UI_TEST_NEW_TEXT__')
    downloads=[]
    for i in range(2):
        with page.expect_download() as event:page.locator('#export-records').click()
        download=event.value;path=log_dir/f'isolated-fixture-export-{i}.json';download.save_as(path)
        assert json.loads(path.read_text())[0]==original;downloads.append(download.suggested_filename)
    assert downloads[0]!=downloads[1]
    page.reload();assert page.evaluate('rows()[0]')==original
    assert page.locator('[data-kind="review"]').nth(1).locator('[data-key="user_observation"]').input_value()=='__UI_TEST_NEW_TEXT__'
    page.goto(scene_url);page.wait_for_selector('#atb-viewer[data-frame="0"]')
    page.locator('[data-rate="0.5"]').click();page.locator('#video-note').click();page.wait_for_selector('#context-banner:not([hidden])')
    page.locator('#import-context').click();data=page.evaluate('rows()')
    assert data[0]==original and data[-1]['frame_id_or_transition_id']=='' and data[-1]['actual_playback_rate']==.5 and data[-1]['actual_view_mode']=='视频慢放'
    return dict(B1_same_key=True,old_text_and_unknown_fields_preserved=True,append_only_context=True,
        exact_transition_and_shared_ROI_transferred=True,PNG_rate_not_misrepresented=True,actual_video_rate_recorded=True,
        reload_persistence=True,timestamped_exports_distinct=True,test_fixture_profile_only=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--date',default='20261013')
    parser.add_argument('--browser-executable',help='Optional H.264-capable Chrome executable for actual playback checks')
    args=parser.parse_args()
    c=json.loads((ROOT/'configs/any2bokeh-failure-analysis-20261009.json').read_text())
    m=json.loads((ROOT/(c['record_prefix']+'-materials.json')).read_text())
    samples=json.loads((ROOT/(c['record_prefix']+'-B-samples.json')).read_text())['rows']
    for item in m['items']:
        path=ROOT/item['index'];text=path.read_text()
        config=json.loads(re.search(r'<script id="atb-viewer-config" type="application/json">(.*?)</script>',text,re.S).group(1))
        assert [p['label'] for p in config['panels']]==item['labels']
        assert config['frames']==25 and config['width']==1024 and config['height']==576
        for panel in config['panels']:
            assert all((path.parent/panel['prefix']/f'{t:05d}.png').is_file() for t in range(25))
        if item['package']=='B1':assert 'Proxy GT' not in text and 'PSNR' not in text and 'LPIPS' not in text
    log_dir=ROOT/'logs'/f'any2bokeh-review-ui-{args.date}';log_dir.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(ROOT)))
    thread=Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}/'
    errors=[];checks=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,executable_path=args.browser_executable,args=['--disable-gpu']);version=browser.version
            context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
            page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
            for package,dataset in [('B1','Synthetic'),('B2','Synthetic'),('B1','DAVIS'),('B2','DAVIS')]:
                item=next(i for i in m['items'] if i['package']==package and i['dataset']==dataset)
                checks.append(inspect_layout(page,item,base+item['index'],errors));print('BROWSER_LAYOUT_PASS',package,dataset,flush=True)
            scene=ROOT/c['output']/'B1/B01/index.html'
            preservation=form_preservation(page,samples[0],base+str(scene.relative_to(ROOT)),log_dir)
            # A new B2 form has a distinct key and still displays the B1-first notice.
            page.goto(base+c['output']+'/B2/review.html')
            page.wait_for_selector('[data-kind="review"]');assert 'B2仅在用户完成B1记录后使用' in page.locator('body').inner_text()
            assert page.evaluate('JSON.parse(localStorage.getItem("atb-failure-B1-user-20261009"))[0]')==fixture_record(samples[0])
            assert page.evaluate('localStorage.getItem("atb-failure-B2-user-20261009")') is None
            page.goto(scene.as_uri());page.wait_for_selector('#atb-viewer[data-frame="0"]');select_frame(page,12)
            page.locator('[data-mode="adjacent"]').click();page.wait_for_selector('#atb-viewer[data-mode="adjacent"]');roi_select(page)
            assert page.locator('.atb-frame-image').count()==4
            file_log=log_dir/'file-fixtures';file_log.mkdir(exist_ok=True)
            file_preservation=form_preservation(page,samples[0],scene.as_uri(),file_log)
            assert not errors,errors
            context.close();browser.close()
    finally:server.shutdown();server.server_close()
    report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),browser=f'Chrome {version}',Playwright='1.55.0',
        static_scene_pages_checked=len(m['items']),HTTP_layout_checks=checks,storage_and_export_checks=preservation,
        direct_file_storage_and_export_checks=file_preservation,
        direct_file_PNG_navigation_and_ROI=True,B2_separate_key_and_pending_notice=True,javascript_errors=errors,
        scope='Tool functionality only; isolated synthetic test storage. No visual quality conclusions or actual user profile access.')
    path=ROOT/'03_实验记录/records'/f'any2bokeh-review-ui-update-{args.date}.json';record=json.loads(path.read_text())
    record['status']='TOOL_FUNCTION_CHECKS_PASSED_USER_VISUAL_REVIEW_PENDING';record['browser_function_checks']=report
    with path.open('w') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    print('ALL_UI_FUNCTION_CHECKS_PASSED',flush=True)


if __name__=='__main__':main()
