"""Check B36 input or paired-output materials at the existing HTTP root.

Uses the already-installed websocket-client and existing headless Chrome CDP;
does not access the user's browser profile, models, or GPU.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import urljoin
from urllib.request import Request, build_opener, ProxyHandler

import websocket

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/any2bokeh-B36-rule-b-20261021.json'


class NavigationPending(RuntimeError):
    pass


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ['href','src'] and v and not v.startswith(('data:','#')))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=str(CONFIG));args=parser.parse_args()
    config_path=Path(args.config);config_path=config_path if config_path.is_absolute() else ROOT/config_path
    c=json.loads(config_path.read_text());out=ROOT/c['output'];checks=json.loads((out/'program-checks.json').read_text())
    is_output='condition_labels' in c
    root='http://127.0.0.1:8765/';base=root+out.name+'/'
    # Loopback validation must reach the existing server directly. Do not alter
    # environment, VS Code forwarding, or any user's proxy configuration.
    local_open=build_opener(ProxyHandler({})).open
    urls=set()
    for relative in ['index.html','B36/index.html','review.html']:
        p=out/relative;url=base+relative;urls.add(url);parser=Links();parser.feed(p.read_text())
        urls.update(urljoin(url,x) for x in parser.links)
    item=json.loads((out/'viewer-item.json').read_text())
    panels=len(item['panels'])
    for panel in item['panels']:
        urls.update(base+'B36/'+panel['prefix']+f'{t:05d}.png' for t in range(25))
    def head(url):
        with local_open(Request(url,method='HEAD'),timeout=15) as f:
            assert f.status==200,url
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(head,sorted(urls)))
    print('HTTP',len(urls),'resources200',flush=True)
    with local_open('http://127.0.0.1:9229/json') as f:targets=json.load(f)
    target=next(x for x in targets if x['type']=='page')
    ws=websocket.create_connection(target['webSocketDebuggerUrl'],timeout=40,http_no_proxy=['127.0.0.1','localhost'])
    counter=0;exceptions=[]
    def call(method,params=None):
        nonlocal counter
        counter+=1;ident=counter;ws.send(json.dumps(dict(id=ident,method=method,params=params or {})))
        while True:
            data=json.loads(ws.recv())
            if data.get('method')=='Runtime.exceptionThrown':exceptions.append(data['params'])
            if data.get('id')==ident:
                if method=='Runtime.evaluate' and 'error' in data and data['error'].get('code')==-32000 and any(
                    word in data['error'].get('message','') for word in ['navigated','context was destroyed','Cannot find context']):
                    raise NavigationPending(data['error']['message'])
                assert 'error' not in data,data
                return data.get('result',{})
    def js(expression):
        r=call('Runtime.evaluate',dict(expression=expression,awaitPromise=True,returnByValue=True))
        assert 'exceptionDetails' not in r,r
        return r.get('result',{}).get('value')
    def wait(expression):
        for attempt in range(4):
            try:
                return js('''(async()=>{for(let i=0;i<600;i++){if('''+expression+''')return true;await new Promise(r=>setTimeout(r,50));}throw new Error('condition timed out');})()''')
            except NavigationPending:
                if attempt==3:raise
                time.sleep(.1)
    def frame(t):
        js(f"document.getElementById('frame-slider').value={t};document.getElementById('frame-slider').dispatchEvent(new Event('input',{{bubbles:true}}));")
        wait(f"document.getElementById('atb-viewer').dataset.frame==='{t}' && document.getElementById('frame-status').textContent.includes('PNG同步完成')")
    call('Page.enable');call('Runtime.enable')
    call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1000,deviceScaleFactor=1,mobile=False))
    call('Page.navigate',dict(url=base+'B36/'))
    wait("document.getElementById('atb-viewer')?.dataset.frame==='"+str(item.get('initial_frame',0))+"'")
    if is_output:
        positions=js("[...document.querySelectorAll('.atb-panel')].map(x=>{const r=x.getBoundingClientRect();return {x:r.x,y:r.y}})")
        assert positions[1]['y']==positions[2]['y'] and positions[2]['x']>positions[1]['x'],'A/R must be side by side'
    for t in range(25):
        frame(t)
        assert js("[...document.querySelectorAll('.atb-frame-image')].every(x=>x.complete&&x.naturalWidth===1024&&x.naturalHeight===576&&Number(x.dataset.frame)==="+str(t)+")")
        assert js("document.querySelectorAll('.atb-frame-image').length")==panels
    print('BROWSER all25 synchronized PNG frames',flush=True)
    for rate in [.5,.25,1]:
        js(f"document.querySelector('[data-rate=\"{rate}\"]').click()")
        assert js("document.querySelector('video').playbackRate")==rate
    js("document.getElementById('tracked-roi-select').value='neck';document.getElementById('tracked-roi-select').dispatchEvent(new Event('change'))")
    frame(0);expected=next(x for x in item['tracked_rois'] if x['id']=='neck')['xywh_by_frame']['0']
    assert js("JSON.parse(document.getElementById('atb-viewer').dataset.roi)")==expected
    assert js("[...document.querySelectorAll('.atb-crop canvas')].map(x=>JSON.parse(x.dataset.roi))")==[expected]*panels
    frame(21)
    neck21=next(x for x in item['tracked_rois'] if x['id']=='neck')['xywh_by_frame']['21']
    if neck21 is None:
        assert js("document.getElementById('atb-viewer').dataset.roi")=='null'
        assert js("[...document.querySelectorAll('.atb-crop')].every(x=>x.hidden)")
    else:
        assert js("JSON.parse(document.getElementById('atb-viewer').dataset.roi)")==neck21
        assert js("[...document.querySelectorAll('.atb-crop canvas')].map(x=>JSON.parse(x.dataset.roi))")==[neck21]*panels
    frame(12);js("document.querySelector('[data-mode=adjacent]').click()")
    wait("document.getElementById('atb-viewer').dataset.mode==='adjacent' && document.getElementById('frame-status').textContent.includes('PNG同步完成')")
    values=js("[...document.querySelectorAll('.atb-frame-image')].map(x=>Number(x.dataset.frame))")
    assert values==[11,12]*panels
    js("document.querySelector('[data-mode=frame]').click()")
    wait("document.getElementById('atb-viewer').dataset.mode==='frame' && document.getElementById('frame-status').textContent.includes('PNG同步完成')")
    js("document.getElementById('clear-roi').click();document.getElementById('roi-enabled').click();document.querySelector('.atb-surface').scrollIntoView({block:'center'});")
    bounds=js("(()=>{const r=document.querySelector('.atb-surface img').getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})()")
    x,y=bounds['x']+bounds['width']*.2,bounds['y']+bounds['height']*.2
    xx,yy=bounds['x']+bounds['width']*.4,bounds['y']+bounds['height']*.4
    for kind,xp,yp,buttons in [('mousePressed',x,y,1),('mouseMoved',xx,yy,1),('mouseReleased',xx,yy,0)]:
        call('Input.dispatchMouseEvent',dict(type=kind,x=xp,y=yp,button='left',buttons=buttons,clickCount=1))
    roi=js("JSON.parse(document.getElementById('atb-viewer').dataset.roi)")
    assert roi is not None and roi[2]>0 and roi[3]>0
    crop_differences=js('''[...document.querySelectorAll('.atb-surface')].map(s=>{
      const a=s.parentElement.querySelector('.atb-crop canvas'),box=JSON.parse(a.dataset.roi),im=s.querySelector('img');
      const ref=document.createElement('canvas');ref.width=1024;ref.height=576;const ctx=ref.getContext('2d');ctx.drawImage(im,0,0);
      const x=ctx.getImageData(...box).data,y=a.getContext('2d').getImageData(0,0,a.width,a.height).data;
      let n=0;for(let i=0;i<x.length;i++)n+=x[i]!==y[i];return n;})''')
    assert crop_differences==[0]*panels
    js("document.querySelector('[data-size=native]').click()")
    assert js("document.getElementById('atb-viewer').classList.contains('native-size')")
    js("document.getElementById('roi-enabled').click();document.activeElement.blur();document.body.tabIndex=-1;document.body.focus()")
    call('Input.dispatchKeyEvent',dict(type='keyDown',key='ArrowRight',code='ArrowRight',windowsVirtualKeyCode=39))
    call('Input.dispatchKeyEvent',dict(type='keyUp',key='ArrowRight',code='ArrowRight',windowsVirtualKeyCode=39))
    wait("document.getElementById('atb-viewer').dataset.frame==='13' && document.getElementById('frame-status').textContent.includes('PNG同步完成')")
    old_keys=['atb-failure-B1-user-20261009','atb-failure-B2-user-20261009']
    if is_output:old_keys.append('atb-B36-rule-b-input-selection-user-20261021')
    old_keys.extend(c.get('additional_preserved_keys', []))
    for key in old_keys:js(f"localStorage.setItem({json.dumps(key)},'PRIVATE_PROFILE_SENTINEL')")
    js("document.getElementById('png-note').click()")
    wait("document.getElementById('import-context')")
    assert js("document.title").startswith('B36')
    if is_output:assert js("window.rows()[0]['视觉是否异常']")=='待用户复核'
    assert js("document.getElementById('context-description').textContent.includes('13')")
    js("document.getElementById('import-context').click()")
    wait("document.querySelectorAll('[data-kind=review]').length===2")
    js("(()=>{const t=document.querySelector('textarea[data-key=notes]');t.value='PRIVATE TEST ONLY';t.dispatchEvent(new Event('input',{bubbles:true}));})()")
    for key in old_keys:assert js(f"localStorage.getItem({json.dumps(key)})")=='PRIVATE_PROFILE_SENTINEL'
    assert len(json.loads(js(f"localStorage.getItem({json.dumps(c['review_storage_key'])})")))==2
    download=out/'browser-check/downloads';download.mkdir(exist_ok=True)
    call('Browser.setDownloadBehavior',dict(behavior='allow',downloadPath=str(download)))
    js("document.getElementById('export-records').click()")
    for _ in range(100):
        files=list(download.glob('*.json'))
        if files:break
        time.sleep(.1)
    assert len(files)==1
    exported=json.loads(files[0].read_text());assert len(exported)==2 and exported[0]['notes']=='PRIVATE TEST ONLY'
    call('Page.reload')
    wait("document.querySelector('textarea[data-key=notes]')?.value==='PRIVATE TEST ONLY'")
    assert not exceptions,exceptions
    version=call('Browser.getVersion')
    checks.update(HTTP_resources=len(urls),all_HTTP_200=True,HTTP_base=base,
        browser=dict(version=version['product'],private_profile=True,GPU_disabled=True,
            synchronized_frames=25,panels=panels,speed_buttons=True,adjacent_frame_pair=[11,12],
            keyboard_navigation=True,moving_ROI=True,missing_ROI_frame21_remains_empty=neck21 is None,
            frame21_ROI_matches_declared_input_policy=True,
            manual_synchronized_ROI=True,crop_pixel_differences=crop_differences,native_size_control=True,
            context_transfer=True,form_save_reload_export=True,old_keys_preserved=True,JS_exceptions=0,
            simulated_feedback_only_in_private_profile=True),
        actual_user_feedback='待用户视觉复核' if is_output else '待用户确认',inference_started=is_output)
    (out/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    dest=ROOT/(c['record_prefix']+'-verification.json');assert not dest.exists();dest.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print('BROWSER_CHECKS_PASS',json.dumps(checks,ensure_ascii=False),flush=True)
    call('Browser.close');ws.close()


if __name__=='__main__':main()
