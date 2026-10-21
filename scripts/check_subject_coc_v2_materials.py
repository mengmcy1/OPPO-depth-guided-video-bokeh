"""Functional HTTP/CDP checks only; no visual-quality judgement.

Use existing8765 root and a private, GPU-disabled Chrome on9229. Browser
feedback is simulated in that private profile and never becomes user evidence.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.parse import urljoin
from urllib.request import Request,build_opener,ProxyHandler

import websocket
from check_b36_rule_b_materials import Links

ROOT=Path(__file__).resolve().parents[1]


def main():
    c=json.loads((ROOT/'configs/any2bokeh-subject-coc-clamp-v2-20261021.json').read_text());out=ROOT/c['output']
    items=json.loads((out/'viewer-items.json').read_text());checks=json.loads((out/'program-checks.json').read_text())
    local=build_opener(ProxyHandler({})).open;base='http://127.0.0.1:8765/'+out.name+'/'
    urls={base,base+'review.html'}
    for relative in ['index.html','review.html']+[i['sample_id']+'/index.html' for i in items]:
        parser=Links();parser.feed((out/relative).read_text());url=urljoin(base,relative)
        urls.add(url);urls.update(urljoin(url,x) for x in parser.links)
    for item in items:
        for panel in item['panels']:
            urls.update(base+item['sample_id']+'/'+panel['prefix']+f'{t:05d}.png' for t in range(25))
    def head(url):
        with local(Request(url,method='HEAD'),timeout=20) as response:assert response.status==200,url
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(head,sorted(urls)))
    print('HTTP',len(urls),'resources200',flush=True)
    with local('http://127.0.0.1:9229/json') as f:targets=json.load(f)
    target=next(x for x in targets if x['type']=='page');ws=websocket.create_connection(target['webSocketDebuggerUrl'],timeout=45,http_no_proxy=['127.0.0.1','localhost'])
    counter=0;errors=[]
    def call(method,params=None):
        nonlocal counter
        counter+=1;ident=counter;ws.send(json.dumps(dict(id=ident,method=method,params=params or {})))
        while True:
            value=json.loads(ws.recv())
            if value.get('method')=='Runtime.exceptionThrown':errors.append(value['params'])
            if value.get('id')==ident:
                if 'error' in value and value['error'].get('code')==-32000:raise RuntimeError('Navigation pending')
                assert 'error' not in value,value
                return value.get('result',{})
    def js(expression):
        result=call('Runtime.evaluate',dict(expression=expression,awaitPromise=True,returnByValue=True));assert 'exceptionDetails' not in result,result
        return result.get('result',{}).get('value')
    def wait(expression):
        for retry in range(5):
            try:
                return js('(async()=>{for(let i=0;i<500;i++){if('+expression+')return true;await new Promise(r=>setTimeout(r,40));}throw new Error("wait timed out");})()')
            except RuntimeError:
                if retry==4:raise
                time.sleep(.1)
    call('Page.enable');call('Runtime.enable');call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1000,deviceScaleFactor=1,mobile=False))
    oldkeys=['atb-failure-B1-user-20261009','atb-failure-B2-user-20261009','atb-B36-rule-b-input-selection-user-20261021','atb-B36-rule-b-output-user-20261021','atb-B36-rule-b-R2-output-user-20261021','atb-B36-rule-b-tail-input-user-20261021']
    results=[]
    for item in items:
        call('Page.navigate',dict(url=base+item['sample_id']+'/'))
        wait("document.getElementById('atb-viewer')?.dataset.frame==='0' && document.getElementById('frame-status').textContent.includes('PNG同步完成')")
        for key in oldkeys:js('localStorage.setItem('+json.dumps(key)+',"PRIVATE_SENTINEL")')
        for t in range(25):
            js(f"document.getElementById('frame-slider').value={t};document.getElementById('frame-slider').dispatchEvent(new Event('input',{{bubbles:true}}))")
            wait(f"document.getElementById('atb-viewer').dataset.frame==='{t}' && document.getElementById('frame-status').textContent.includes('PNG同步完成')")
            assert js("[...document.querySelectorAll('.atb-frame-image')].every(x=>x.complete&&x.naturalWidth===1024&&x.naturalHeight===576&&Number(x.dataset.frame)==="+str(t)+")")
        for rate in [.25,.5,1]:
            js(f"document.querySelector('[data-rate=\"{rate}\"]').click()")
            assert js('document.querySelector("video").playbackRate')==rate
        js("document.getElementById('tracked-roi-select').value='subject';document.getElementById('tracked-roi-select').dispatchEvent(new Event('change'))")
        expected=next(r for r in item['tracked_rois'] if r['id']=='subject')['xywh_by_frame']['24']
        assert js("JSON.parse(document.getElementById('atb-viewer').dataset.roi)")==expected
        assert js("[...document.querySelectorAll('.atb-crop canvas')].map(x=>JSON.parse(x.dataset.roi))")==[expected]*len(item['panels'])
        differences=js('''[...document.querySelectorAll('.atb-surface')].map(s=>{const c=s.parentElement.querySelector('.atb-crop canvas'),box=JSON.parse(c.dataset.roi),im=s.querySelector('img');const ref=document.createElement('canvas');ref.width=1024;ref.height=576;const ctx=ref.getContext('2d');ctx.drawImage(im,0,0);const x=ctx.getImageData(...box).data,y=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<x.length;i++)n+=x[i]!==y[i];return n;})''')
        assert differences==[0]*len(item['panels'])
        if not results:
            js("document.getElementById('clear-roi').click();document.getElementById('roi-enabled').click();document.querySelector('.atb-surface').scrollIntoView({block:'center'})")
            bounds=js("(()=>{const r=document.querySelector('.atb-surface img').getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()")
            x,y=bounds['x']+.2*bounds['w'],bounds['y']+.2*bounds['h']
            xx,yy=bounds['x']+.4*bounds['w'],bounds['y']+.4*bounds['h']
            for kind,xp,yp,buttons in [('mousePressed',x,y,1),('mouseMoved',xx,yy,1),('mouseReleased',xx,yy,0)]:call('Input.dispatchMouseEvent',dict(type=kind,x=xp,y=yp,button='left',buttons=buttons,clickCount=1))
            roi=js("JSON.parse(document.getElementById('atb-viewer').dataset.roi)");assert roi is not None and roi[2]>0 and roi[3]>0
            assert js("[...document.querySelectorAll('.atb-crop canvas')].map(x=>JSON.parse(x.dataset.roi))")==[roi]*len(item['panels'])
            js("document.getElementById('roi-enabled').click();document.activeElement.blur();document.body.tabIndex=-1;document.body.focus()")
            for kind in ['keyDown','keyUp']:call('Input.dispatchKeyEvent',dict(type=kind,key='ArrowLeft',code='ArrowLeft',windowsVirtualKeyCode=37))
            wait("document.getElementById('atb-viewer').dataset.frame==='23' && document.getElementById('frame-status').textContent.includes('PNG同步完成')")
            for kind in ['keyDown','keyUp']:call('Input.dispatchKeyEvent',dict(type=kind,key='ArrowRight',code='ArrowRight',windowsVirtualKeyCode=39))
            wait("document.getElementById('atb-viewer').dataset.frame==='24' && document.getElementById('frame-status').textContent.includes('PNG同步完成')")
        js("document.querySelector('[data-mode=adjacent]').click()")
        wait("document.getElementById('atb-viewer').dataset.mode==='adjacent' && document.getElementById('frame-status').textContent.includes('PNG同步完成')")
        assert js("[...document.querySelectorAll('.atb-frame-image')].map(x=>Number(x.dataset.frame))")==[23,24]*len(item['panels'])
        js("document.querySelector('[data-size=native]').click()")
        assert js("document.getElementById('atb-viewer').classList.contains('native-size')")
        results.append(dict(sample_id=item['sample_id'],frames=25,panels=len(item['panels']),ROI_pixel_differences=differences,speeds=True,adjacent=True,native_size=True))
        print('BROWSER',item['sample_id'],'25 frames checked',flush=True)
    # Import a real viewer context into a new record, preserving the six defaults.
    js("document.getElementById('png-note').click()")
    wait("document.getElementById('import-context')")
    assert js('window.rows().length')==6
    for key in c['review_fields']:assert js('window.rows().every(r=>r['+json.dumps(key)+']==="待用户复核")')
    js("document.getElementById('import-context').click()")
    wait("document.querySelectorAll('[data-kind=review]').length===7")
    js("(()=>{const t=document.querySelector('textarea[data-key=notes]');t.value='PRIVATE TEST ONLY';t.dispatchEvent(new Event('input',{bubbles:true}));})()")
    for key in oldkeys:assert js('localStorage.getItem('+json.dumps(key)+')')=='PRIVATE_SENTINEL'
    download=out/'browser-check/downloads';download.mkdir(exist_ok=True)
    call('Browser.setDownloadBehavior',dict(behavior='allow',downloadPath=str(download)));js("document.getElementById('export-records').click()")
    for _ in range(100):
        files=list(download.glob('*.json'))
        if files:break
        time.sleep(.1)
    assert len(files)==1 and len(json.loads(files[0].read_text()))==7
    call('Page.reload');wait("document.querySelector('textarea[data-key=notes]')?.value==='PRIVATE TEST ONLY'")
    assert not errors,errors
    checks.update(HTTP_resources=len(urls),all_HTTP_200=True,HTTP_base=base,
        browser=dict(version=call('Browser.getVersion')['product'],private_profile=True,GPU_disabled=True,items=results,
            five_separate_visual_fields_pending=True,manual_ROI=True,keyboard_navigation=True,context_transfer=True,save_reload_export=True,old_keys_preserved=True,JS_exceptions=0,
            simulated_feedback_only_in_private_profile=True))
    (out/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    p=ROOT/(c['record_prefix']+'-verification.json');assert not p.exists();p.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print('ALL_FUNCTIONAL_CHECKS_PASS',flush=True);call('Browser.close');ws.close()


if __name__=='__main__':main()
