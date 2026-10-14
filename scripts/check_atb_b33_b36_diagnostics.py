"""Check the existing HTTP root and moving ROI tools in a private test profile."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin,urlparse
from urllib.request import Request,urlopen

from playwright.sync_api import sync_playwright
from check_any2bokeh_review_pages import ROOT,VISUAL,inspect_layout,form_preservation,select_frame,roi_select


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and v and not v.startswith(('data:','#')):self.links.append(v)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base-url',required=True);parser.add_argument('--browser-executable',required=True)
    args=parser.parse_args();base=args.base_url.rstrip('/')+'/'
    c=json.loads((ROOT/'configs/any2bokeh-B33-B36-diagnostics-20261014.json').read_text());out=ROOT/c['output'];root=out.parent
    prefix=c['record_prefix'];materials=json.loads((ROOT/(prefix+'-materials.json')).read_text())
    def url(path):return base+(ROOT/path).relative_to(root).as_posix()
    addresses=set()
    for p in [out/'index.html',out/'B33/index.html',out/'B36/index.html',out/'review-B33.html',out/'review-B36.html']:
        address=url(p);addresses.add(address);links=Links();links.feed(p.read_text())
        addresses.update(urljoin(address,v) for v in links.links)
    for code,item in materials['items'].items():
        folder=(ROOT/item['index']).parent
        for panel in item['panels']:
            for t in range(25):addresses.add(url(folder/panel['prefix']/f'{t:05d}.png'))
    addresses.add(url(out/'method-mapping.json'))
    def head(address):
        assert urlparse(address).hostname==urlparse(base).hostname
        with urlopen(Request(address,method='HEAD'),timeout=30) as response:assert response.status==200,address
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(head,sorted(addresses)))
    print('HTTP_ROOT_PASS',len(addresses),flush=True)
    errors=[];logs=ROOT/'logs/atb-b33-b36-browser-20261014';logs.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        b=p.chromium.launch(headless=True,executable_path=args.browser_executable,args=['--disable-gpu']);version=b.version
        context=b.new_context(viewport={'width':1440,'height':1000},accept_downloads=True);page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        item=materials['items']['B36'];report36=inspect_layout(page,dict(item,package='B2',dataset='DAVIS',sample_id='B36',labels=[p['label'] for p in item['panels']]),url(item['index']),errors)
        print('B36_STANDARD_LAYOUT_PASS',flush=True)
        # Check moving positions, missing frames, and native crop coordinates.
        for code in ['B33','B36']:
            item=materials['items'][code];page.goto(url(item['index']));initial=1 if code=='B33' else 0
            page.wait_for_selector(f'#atb-viewer[data-frame="{initial}"]')
            if code=='B33':assert page.locator('[data-jump="1"]').count()==page.locator('[data-jump="22"]').count()==1
            selection='R2' if code=='B33' else 'head';roi=next(r for r in item['tracked_rois'] if r['id']==selection)
            page.locator('#tracked-roi-select').select_option(selection)
            for t in [1,22] if code=='B33' else [0,6,12,18,24]:
                select_frame(page,t);box=roi['xywh_by_frame'][str(t)]
                assert json.loads(page.locator('#atb-viewer').get_attribute('data-roi'))==box
                assert page.locator('.atb-crop canvas').evaluate_all('(els)=>els.map(e=>JSON.parse(e.dataset.roi))')==[box]*len(item['panels'])
            page.locator('[data-mode="adjacent"]').click();page.wait_for_selector('#atb-viewer[data-mode="adjacent"]')
            target=22 if code=='B33' else 24
            select_frame(page,target)
            values=page.locator('.atb-surface').evaluate_all('''els=>els.map(surface=>{
                const img=surface.querySelector('img'),crop=surface.parentElement.querySelector('.atb-crop'),canvas=crop.querySelector('canvas');
                const box=canvas.dataset.roi?JSON.parse(canvas.dataset.roi):null;let different=0;
                if(!crop.hidden){const ref=document.createElement('canvas');ref.width=1024;ref.height=576;
                  const ctx=ref.getContext('2d');ctx.drawImage(img,0,0);const a=ctx.getImageData(...box).data;
                  const b=canvas.getContext('2d').getImageData(0,0,box[2],box[3]).data;
                  for(let i=0;i<a.length;i++)different+=a[i]!==b[i];}
                return {frame:Number(img.dataset.frame),roi:box,hidden:crop.hidden,different};})''')
            for value in values:
                expected=roi['xywh_by_frame'][str(value['frame'])]
                assert value['hidden']==(expected is None) and value['different']==0
                if expected is not None:assert value['roi']==expected
            page.locator('[data-mode="frame"]').click();page.wait_for_selector('#atb-viewer[data-mode="frame"]')
            if code=='B33':
                page.locator('#tracked-roi-select').select_option('R3');select_frame(page,22)
                assert page.locator('#atb-viewer').get_attribute('data-roi')=='null'
                assert page.locator('.atb-crop').evaluate_all('(els)=>els.every(e=>e.hidden)')
            else:
                select_frame(page,2);assert page.locator('#atb-viewer').get_attribute('data-roi')=='null'
            print('MOVING_ROI_AND_MISSING_PASS',code,flush=True)

        old_samples=json.loads((ROOT/'03_实验记录/records/any2bokeh-failure-analysis-20261009-B-samples.json').read_text())['rows']
        sample=next(r for r in old_samples if r['sample_id']=='B19')
        old_check=form_preservation(page,sample,base+'B1/B19/',logs)
        page.evaluate('()=>localStorage.setItem("atb-failure-B2-user-20261009",JSON.stringify([{test_only:true,unknown:[1,2]}]))')
        old_keys=['atb-failure-B1-user-20261009','atb-failure-B2-user-20261009','atb-failure-B1-batch2-user-20261014']
        before={k:page.evaluate('(k)=>localStorage.getItem(k)',k) for k in old_keys}
        for code in ['B33','B36']:
            item=materials['items'][code];page.goto(url(item['index']));page.wait_for_selector('#atb-viewer[data-frame]')
            select_frame(page,6);roi_select(page);page.locator('[data-rate="0.5"]').click();page.locator('#png-note').click()
            page.wait_for_selector('#context-banner:not([hidden])');page.locator('#import-context').click()
            last=page.evaluate('rows().at(-1)');assert last['ROI_xywh']=='100,120,200,120' and last['frame_id_or_transition_id']=='6'
            assert last['ROI_selection_origin']=='user_manual_coordinates_semantics_unconfirmed' and last['actual_playback_rate']==''
            assert all(last[k]=='待用户复核' for k in VISUAL) and last['user_observation']==''
            key=c[code]['review_storage_key'];assert page.evaluate('(k)=>JSON.parse(localStorage.getItem(k)).at(-1)',key)==last
            downloads=[]
            for i in range(2):
                with page.expect_download() as event:page.locator('#export-records').click()
                download=event.value;download.save_as(logs/f'{code}-isolated-fixture-{i}.json');downloads.append(download.suggested_filename)
            assert downloads[0]!=downloads[1];page.reload();assert page.evaluate('rows().at(-1)')==last
            assert all(page.evaluate('(k)=>localStorage.getItem(k)',k)==v for k,v in before.items())
        assert not errors,errors;context.close();b.close()
    path=ROOT/(prefix+'-verification.json');report=json.loads(path.read_text())
    report.update(status='诊断材料已准备，等待用户视觉复核',browser_checks=dict(checked_utc=datetime.now(timezone.utc).isoformat(),
        actual_HTTP_base=base,material_root=str(root.relative_to(ROOT)),HTTP_200_links=len(addresses),browser='Chrome '+version,
        B36_standard_layout=report36,B33_initial1_and_endpoint_buttons=True,moving_ROI_coordinates_same_for_all_methods_per_frame=True,
        adjacent_frames_use_own_coordinates=True,missing_presets_never_carried_or_interpolated=True,ROI_canvas_pixel_mismatches=0,
        old_form_preservation=old_check,independent_keys_append_export_reload=True,old_B1_B2_batch2_keys_unchanged=True,
        javascript_errors=errors,isolated_fixture_profile_only=True,user_actual_client_proxy_path_reproduced=False))
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('ALL_DIAGNOSTIC_TOOL_CHECKS_PASS',flush=True)


if __name__=='__main__':main()
