"""Bounded HTTP, coordinate/pixel and form checks in a private CPU browser.

Simulated form content is test data, never a user observation. Existing servers,
browser profiles, predictions and feedback keys are not changed.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import build_opener, ProxyHandler, Request

import websocket

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/any2bokeh-failure-analysis-20261009/V2-local-input-output-check-20261022'
BASE = 'http://127.0.0.1:8765/'+OUT.name+'/'


def main():
    local=build_opener(ProxyHandler({})).open
    items=json.loads((OUT/'viewer-items.json').read_text())
    urls={BASE,BASE+'statistics.csv',BASE+'ROI-candidates.json'}
    for r in items:
        urls.update(BASE+p for p in r['paths'].values())
        urls.update(BASE+r['paths']['RGB'].replace('RGB.png',n) for n in ['five-panel.png','native.png'])
        urls.add('http://127.0.0.1:8765/subject-coc-clamp-v2-20261021/'+r['sample_id']+'/')
    def head(url):
        with local(Request(url,method='HEAD'),timeout=20) as f:
            assert f.status==200,url
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(head,sorted(urls)))
    private=OUT/'browser-check';private.mkdir(exist_ok=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']='/tmp/atb-review-browser-20261013/system-libs/extracted/usr/lib/x86_64-linux-gnu'
    browser=subprocess.Popen(['/root/.cache/ms-playwright/chromium-1187/chrome-linux/chrome',
        '--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-first-run',
        '--remote-debugging-address=127.0.0.1',f'--remote-debugging-port={port}','--remote-allow-origins=*',
        '--user-data-dir='+str(private/'profile'),'about:blank'],env=env,stdout=(private/'chrome.log').open('w'),stderr=subprocess.STDOUT)
    ws=None
    try:
        for _ in range(100):
            try:
                targets=json.load(local(f'http://127.0.0.1:{port}/json',timeout=1));break
            except Exception:
                time.sleep(.1)
        else:
            raise RuntimeError('Private Chrome unavailable')
        target=next(t for t in targets if t['type']=='page')
        ws=websocket.create_connection(target['webSocketDebuggerUrl'],timeout=45,http_no_proxy=['127.0.0.1','localhost'])
        counter=0;errors=[]
        def call(method,params=None):
            nonlocal counter
            counter+=1;ident=counter;ws.send(json.dumps(dict(id=ident,method=method,params=params or {})))
            while True:
                r=json.loads(ws.recv())
                if r.get('method')=='Runtime.exceptionThrown':errors.append(r['params'])
                if r.get('id')==ident:
                    assert 'error' not in r,r
                    return r.get('result',{})
        def js(expression):
            r=call('Runtime.evaluate',dict(expression=expression,awaitPromise=True,returnByValue=True))
            assert 'exceptionDetails' not in r,r
            return r.get('result',{}).get('value')
        def wait(expr):
            return js('(async()=>{for(let n=0;n<400;n++){if('+expr+')return true;await new Promise(r=>setTimeout(r,25))}throw Error("timeout")})()')
        call('Page.enable');call('Runtime.enable')
        call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1000,deviceScaleFactor=1,mobile=False))
        call('Page.navigate',dict(url=BASE))
        wait("document.querySelectorAll('#actual canvas').length===5 && [...document.querySelectorAll('canvas')].every(c=>c.dataset.source)")
        oldkeys=['atb-failure-B1-user-20261009','atb-failure-B2-user-20261009','atb-subject-coc-clamp-v2-user-20261021']
        for k in oldkeys:js('localStorage.setItem('+json.dumps(k)+',"PRIVATE_SENTINEL")')
        checks=[]
        for r in items:
            js("$('sample').value="+json.dumps(r['sample_id'])+";frames();$('frame').value="+str(r['frame_id'])+";rois();$('roi').value="+json.dumps(r['roi_id'])+";show()")
            expected=json.dumps(r['paths']['RGB'])
            wait("document.querySelector('#actual canvas').dataset.source==="+expected+" && [...document.querySelectorAll('canvas')].every(c=>JSON.stringify(JSON.parse(c.dataset.box))===JSON.stringify(c.closest('#actual')?current.display_xywh:current.native_xywh))")
            assert js("$('confirmed').value")== '待用户确认'
            assert js("$('note').value")==''
            differences=js('''(async()=>{return await Promise.all([...document.querySelectorAll('canvas')].map(async c=>{const im=await load(c.dataset.source),box=JSON.parse(c.dataset.box),ref=document.createElement('canvas');ref.width=c.width;ref.height=c.height;const ctx=ref.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(im,...box,0,0,c.width,c.height);const a=ctx.getImageData(0,0,c.width,c.height).data,b=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<a.length;i++)n+=a[i]!==b[i];return n}));})()''')
            assert differences==[0]*8,differences
            assert js("[...document.querySelectorAll('#actual canvas')].every(c=>JSON.stringify(JSON.parse(c.dataset.box))===JSON.stringify(current.display_xywh))")
            checks.append(dict(sample_id=r['sample_id'],frame_id=r['frame_id'],roi_id=r['roi_id'],canvas_pixel_differences=differences))
        # Manual common ROI changes viewing only; statistical row stays fixed.
        js("$('location').scrollIntoView({block:'center'})")
        bounds=js("(()=>{const b=$('location').getBoundingClientRect();return {x:b.x,y:b.y,w:b.width,h:b.height}})()")
        for kind,fx,fy,buttons in [('mousePressed',.2,.3,1),('mouseMoved',.3,.4,1),('mouseReleased',.3,.4,0)]:
            call('Input.dispatchMouseEvent',dict(type=kind,x=bounds['x']+fx*bounds['w'],y=bounds['y']+fy*bounds['h'],button='left',buttons=buttons,clickCount=1))
        wait("JSON.stringify(viewBox)!==JSON.stringify(current.display_xywh) && [...document.querySelectorAll('#actual canvas')].every(c=>JSON.stringify(JSON.parse(c.dataset.box))===JSON.stringify(viewBox))")
        statbox=js('current.display_xywh');assert js('current.statistics.display_xywh')==statbox
        js("$('note').value='PRIVATE FUNCTIONAL TEST ONLY';$('save').click()")
        assert js('stored()[id()].view_box')!=statbox
        assert js('stored()[id()].statistical_box')==statbox
        for k in oldkeys:assert js('localStorage.getItem('+json.dumps(k)+')')=='PRIVATE_SENTINEL'
        downloads=private/'downloads';downloads.mkdir(exist_ok=True)
        call('Browser.setDownloadBehavior',dict(behavior='allow',downloadPath=str(downloads)));js("$('export').click()")
        for _ in range(100):
            files=list(downloads.glob('*.json'))
            if files:break
            time.sleep(.1)
        assert len(files)==1
        exported=json.loads(files[0].read_text());assert len(exported['records'])==1
        row=next(iter(exported['records'].values()));assert row['note']=='PRIVATE FUNCTIONAL TEST ONLY' and row['visual_verdict']=='待用户复核'
        call('Page.reload');wait("document.querySelector('#actual canvas')?.dataset.source")
        last=items[-1]
        js("choose("+json.dumps(last['sample_id'])+");$('frame').value="+str(last['frame_id'])+";rois();$('roi').value="+json.dumps(last['roi_id'])+";show()")
        assert js("$('note').value")=='PRIVATE FUNCTIONAL TEST ONLY'
        assert not errors,errors
        result=dict(HTTP_base=BASE,HTTP_resources=len(urls),all_HTTP_200=True,items=checks,
                    displayed_panel_coordinate_and_pixel_checks=True,manual_ROI_view_only=True,
                    save_reload_export=True,old_keys_preserved=True,JS_exceptions=0,
                    browser=call('Browser.getVersion')['product'],private_profile=True,GPU_disabled=True,
                    simulated_feedback_is_not_user_feedback=True)
        p=ROOT/'03_实验记录/records/any2bokeh-V2-local-input-output-20261022-verification.json'
        assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(HTTP_resources=len(urls),ROIs=len(checks),eight_panel_pixels_match=True,forms_pass=True,JS_exceptions=0)),flush=True)
    finally:
        if ws is not None:ws.close()
        browser.terminate()  # only the private subprocess created above
        browser.wait(timeout=20)


if __name__=='__main__':main()
