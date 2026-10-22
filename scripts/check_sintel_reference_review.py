"""Fresh-profile functional checks of generated review pages; no visual grading."""
import json
import argparse
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--browser-executable',required=True);parser.add_argument('--base-url',default='http://127.0.0.1:8765/ours-sintel-reference-atb-20261023/');args=parser.parse_args()
BASE=args.base_url
LOG=ROOT/'logs/ours-sintel-dual-input-v2'
errors=[];checked=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=args.browser_executable,headless=True,args=['--disable-gpu'])
    ctx=browser.new_context(accept_downloads=True,viewport={'width':1440,'height':1000});page=ctx.new_page()
    page.on('pageerror',lambda e:errors.append(str(e)))
    for name in ['ambush_4','shaman_2']:
        page.goto(BASE+'view/'+name+'/',wait_until='networkidle')
        page.wait_for_selector('#atb-viewer[data-frame="0"]')
        assert page.locator('#all-frames .atb-thumbnail').count()==25
        for rate in [1,.5,.25]:
            page.locator(f'[data-rate="{rate:g}"]').click()
            assert page.locator('video').evaluate('(v)=>v.playbackRate')==rate
        page.locator('video').evaluate('(v)=>v.play()')
        page.wait_for_function('document.querySelector("video").currentTime>0')
        page.locator('video').evaluate('(v)=>v.pause()')
        duration=page.locator('video').evaluate('(v)=>v.duration');assert abs(duration-1.25)<.001
        page.locator('#frame-slider').evaluate('(e)=>{e.value=24;e.dispatchEvent(new Event("input",{bubbles:true}))}')
        page.wait_for_selector('#atb-viewer[data-frame="24"]');page.wait_for_function('!document.getElementById("png-note").disabled')
        assert page.locator('.atb-frame-image').evaluate_all('(els)=>els.length===3&&els.every(e=>e.naturalWidth===1024&&e.naturalHeight===436&&e.dataset.frame==="24")')
        page.locator('[data-mode="adjacent"]').click();page.wait_for_selector('#atb-viewer[data-mode="adjacent"]')
        assert page.locator('.atb-frame-image').count()==6
        page.locator('[data-mode="frame"]').click();page.locator('#roi-enabled').check()
        surf=page.locator('.atb-surface').first;surf.scroll_into_view_if_needed();b=surf.locator('img').bounding_box()
        page.mouse.move(b['x']+100*b['width']/1024,b['y']+100*b['height']/436);page.mouse.down()
        page.mouse.move(b['x']+300*b['width']/1024,b['y']+220*b['height']/436,steps=6);page.mouse.up()
        roi=json.loads(page.locator('#atb-viewer').get_attribute('data-roi'));assert roi==[100,100,200,120],roi
        assert all(json.loads(x)==roi for x in page.locator('.atb-crop canvas').evaluate_all('(els)=>els.map(e=>e.dataset.roi)'))
        page.locator('#png-note').click();page.wait_for_selector('#context-banner:not([hidden])')
        page.locator('#import-context').click();page.locator('[data-key="user_observation"]').last.fill('__FUNCTION_TEST_NOT_USER_FEEDBACK__')
        page.reload();assert page.locator('[data-key="user_observation"]').last.input_value()=='__FUNCTION_TEST_NOT_USER_FEEDBACK__'
        with page.expect_download() as ev:page.locator('#export-records').click()
        download=ev.value;file=LOG/(name+'-isolated-ui-fixture.json');download.save_as(file)
        rows=json.loads(file.read_text());assert rows[-1]['ROI_xywh']=='100,100,200,120'
        checked.append(dict(scene=name,video_decodes=True,duration=duration,PNG_1024x436=True,adjacent=True,ROI=roi,storage_export=True))
    assert not errors,errors
    result=dict(status='FUNCTIONAL_REVIEW_CHECKS_PASSED',checks=checked,errors=errors,chrome=browser.version,isolated_profile=True,
                user_visual_feedback_written=False,base_url=BASE)
    browser.close()
with (ROOT/'03_实验记录/records/ours-sintel-reference-atb-20261023-review-check.json').open('x') as f:
    json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
