(() => {
  'use strict';
  const config = JSON.parse(document.getElementById('atb-viewer-config').textContent);
  const root = document.getElementById('atb-viewer');
  const video = document.querySelector('video');
  const $ = id => document.getElementById(id);
  const cache = new Map();
  let frame = 0, from = 0, mode = 'frame', size = 'fit', roi = null, ticket = 0;
  let rendered = false, previousVideoTime = 0;
  let roiOrigin = 'none', diagnosticROI = '';
  const clamp = (n, low, high) => Math.max(low, Math.min(high, n));
  const png = (panel, t) => `${panel.prefix}${String(t).padStart(5, '0')}.png`;
  const pressed = (selector, key, value) => root.querySelectorAll(selector).forEach(b => b.setAttribute('aria-pressed', String(b.dataset[key] === value)));

  function loadPNG(panel, t) {
    const url = png(panel, t);
    if (!cache.has(url)) {
      cache.set(url, new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          if (img.naturalWidth !== config.width || img.naturalHeight !== config.height) {
            reject(new Error(`PNG尺寸不符：${url}`)); return;
          }
          resolve(img);
        };
        img.onerror = () => reject(new Error(`PNG加载失败：${url}`));
        img.src = url;
      }));
    }
    return cache.get(url);
  }

  function originalLink(panel, t) {
    const a = document.createElement('a');
    a.href = png(panel, t); a.target = '_blank'; a.rel = 'noopener';
    a.textContent = `${panel.label} 帧${t}完整PNG`; return a;
  }

  function updateROI() {
    root.dataset.roi = JSON.stringify(roi);
    $('roi-label').textContent = roi ? `ROI x,y,w,h = ${roi.join(', ')}` : 'ROI未选择';
    root.querySelectorAll('.atb-surface').forEach(surface => {
      const rect = surface.querySelector('rect');
      rect.style.display = roi ? '' : 'none';
      const crop = surface.parentElement.querySelector('.atb-crop');
      crop.hidden = !roi;
      if (!roi) return;
      ['x', 'y', 'width', 'height'].forEach((key, i) => rect.setAttribute(key, roi[i]));
      const canvas = crop.querySelector('canvas');
      canvas.width = roi[2]; canvas.height = roi[3];
      const ctx = canvas.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(surface.querySelector('img'), ...roi, 0, 0, roi[2], roi[3]);
      // Native crop pixels are kept; CSS enlargement is identical nearest-neighbor.
      const factor = size === 'native' ? 1 : 2;
      canvas.style.width = `${roi[2] * factor}px`; canvas.style.height = `${roi[3] * factor}px`;
      canvas.dataset.roi = JSON.stringify(roi);
      crop.querySelector('p').textContent = `同坐标ROI，显示${factor * 100}%（最近邻）`;
    });
  }

  function point(event, surface) {
    const bounds = surface.querySelector('img').getBoundingClientRect();
    return [clamp(Math.round((event.clientX - bounds.left) * config.width / bounds.width), 0, config.width),
            clamp(Math.round((event.clientY - bounds.top) * config.height / bounds.height), 0, config.height)];
  }

  function manualROI(surface) {
    let start = null;
    const move = event => {
      if (!start) return;
      const end = point(event, surface);
      const x = Math.min(start[0], end[0], config.width - 1);
      const y = Math.min(start[1], end[1], config.height - 1);
      roi = [x, y, Math.max(1, Math.min(config.width - x, Math.abs(end[0] - start[0]))),
                   Math.max(1, Math.min(config.height - y, Math.abs(end[1] - start[1])))];
      roiOrigin = 'user_manual_coordinates_semantics_unconfirmed'; diagnosticROI = '';
      if ($('diagnostic-roi-select')) $('diagnostic-roi-select').value = '';
      updateROI();
    };
    surface.addEventListener('pointerdown', event => {
      if (!$('roi-enabled').checked || event.button !== 0) return;
      event.preventDefault(); start = point(event, surface); surface.setPointerCapture(event.pointerId);
    });
    surface.addEventListener('pointermove', move);
    surface.addEventListener('pointerup', event => { move(event); start = null; });
    surface.addEventListener('pointercancel', () => { start = null; });
  }

  function imageCell(panel, t, image) {
    const wrap = document.createElement('div'); wrap.className = 'atb-image-wrap';
    const caption = document.createElement('p'); caption.className = 'atb-frame-caption';
    caption.textContent = `帧${t} · `; caption.append(originalLink(panel, t)); wrap.append(caption);
    const surface = document.createElement('div'); surface.className = 'atb-surface';
    image.className = 'atb-frame-image'; image.dataset.frame = t; image.dataset.panel = panel.label;
    image.alt = `${panel.label} 帧${t}`; image.draggable = false; surface.append(image);
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${config.width} ${config.height}`);
    svg.append(document.createElementNS(ns, 'rect')); surface.append(svg); wrap.append(surface);
    const crop = document.createElement('div'); crop.className = 'atb-crop'; crop.hidden = true;
    crop.append(document.createElement('p'), document.createElement('canvas')); wrap.append(crop);
    manualROI(surface); return wrap;
  }

  async function render() {
    const current = ++ticket;
    const times = mode === 'frame' ? [frame] : [from, from + 1];
    $('frame-slider').value = frame; $('transition-slider').value = from;
    $('frame-label').textContent = `帧${frame}/24`; $('transition-label').textContent = `${from}→${from + 1}`;
    $('previous-frame').disabled = mode === 'frame' ? frame === 0 : from === 0;
    $('next-frame').disabled = mode === 'frame' ? frame === 24 : from === 23;
    root.classList.toggle('adjacent', mode === 'adjacent');
    pressed('[data-mode]', 'mode', mode);
    rendered = false; $('png-note').disabled = true; $('frame-status').textContent = '正在加载全部面板PNG…';
    try {
      const images = await Promise.all(config.panels.map(panel => Promise.all(times.map(t => loadPNG(panel, t)))));
      if (current !== ticket) return;
      const fragment = document.createDocumentFragment();
      config.panels.forEach((panel, i) => {
        const figure = document.createElement('figure'); figure.className = 'atb-panel';
        const title = document.createElement('h3'); title.textContent = panel.label; figure.append(title);
        const pair = document.createElement('div'); pair.className = 'atb-pair';
        times.forEach((t, j) => pair.append(imageCell(panel, t, images[i][j])));
        figure.append(pair); fragment.append(figure);
      });
      $('synchronous-panels').replaceChildren(fragment);
      root.dataset.frame = frame; root.dataset.transition = `${from}→${from + 1}`; root.dataset.mode = mode;
      root.querySelectorAll('.atb-thumbnail').forEach(card => card.setAttribute('aria-current', String(Number(card.dataset.frame) === frame)));
      updateROI(); rendered = true; $('png-note').disabled = false;
      $('frame-status').textContent = `PNG同步完成 · ${mode === 'frame' ? `帧${frame}` : `${from}→${from + 1}`}`;
    } catch (error) {
      if (current === ticket) $('frame-status').textContent = `${error.message}；暂不能把未完成位置带入表单。`;
    }
  }

  function setFrame(t) {
    frame = clamp(Number(t), 0, 24); from = clamp(frame - 1, 0, 23); render();
  }
  function step(amount) {
    if (mode === 'frame') setFrame(frame + amount);
    else { from = clamp(from + amount, 0, 23); frame = from + 1; render(); }
  }

  function rateLabel() {
    const rate = video.playbackRate;
    $('rate-label').textContent = `当前${rate}×${rate === 1 ? '（原速参照）' : '（仅播放器倍速）'}`;
    video.dataset.playbackRate = rate;
    pressed('[data-rate]', 'rate', String(rate));
  }
  root.querySelectorAll('[data-rate]').forEach(button => button.addEventListener('click', () => { video.playbackRate = Number(button.dataset.rate); rateLabel(); }));
  video.addEventListener('ratechange', rateLabel); rateLabel();
  video.addEventListener('timeupdate', () => {
    if (Number.isFinite(video.duration) && previousVideoTime > video.duration * .8 && video.currentTime < video.duration * .2) {
      $('loop-warning').textContent = '播放器回到开头（24→0）；这不是原视频连续transition，不能记成模型闪烁。';
    }
    previousVideoTime = video.currentTime;
  });
  $('frame-slider').addEventListener('input', event => setFrame(event.target.value));
  $('transition-slider').addEventListener('input', event => { from = Number(event.target.value); frame = from + 1; mode = 'adjacent'; render(); });
  $('previous-frame').addEventListener('click', () => step(-1));
  $('next-frame').addEventListener('click', () => step(1));
  document.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key) || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.target.closest('input,textarea,select,[contenteditable=true]')) return;
    event.preventDefault(); step(event.key === 'ArrowLeft' ? -1 : 1);
  });
  root.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => { mode = button.dataset.mode; render(); }));
  root.querySelectorAll('[data-size]').forEach(button => button.addEventListener('click', () => {
    size = button.dataset.size; pressed('[data-size]', 'size', size);
    root.classList.toggle('native-size', size === 'native'); root.classList.toggle('double-size', size === 'double'); updateROI();
  }));
  $('roi-enabled').addEventListener('change', event => root.classList.toggle('roi-enabled', event.target.checked));
  $('clear-roi').addEventListener('click', () => {
    roi = null; roiOrigin = 'none'; diagnosticROI = '';
    if ($('diagnostic-roi-select')) $('diagnostic-roi-select').value = '';
    updateROI();
  });
  const presets = $('diagnostic-roi-select');
  if (presets) {
    (config.diagnostic_rois || []).forEach((r, i) => {
      const option = document.createElement('option'); option.value = i;
      option.textContent = `${r.id} ${r.name} [${r.xywh.join(',')}]`; presets.append(option);
    });
    presets.addEventListener('change', () => {
      roi = presets.value === '' ? null : [...config.diagnostic_rois[Number(presets.value)].xywh];
      roiOrigin = roi ? 'executor_diagnostic_preset_not_user_confirmed' : 'none';
      diagnosticROI = roi ? config.diagnostic_rois[Number(presets.value)].id : '';
      updateROI();
    });
  }

  function reviewContext(videoObservation) {
    if (!videoObservation && !rendered) return;
    const context = {};
    ['sample_id', 'dataset', 'scene_id', 'source_group', 'split', 'package'].forEach(k => { context[k] = config[k]; });
    context.frame_id_or_transition_id = videoObservation ? '' : (mode === 'frame' ? String(frame) : `${from}→${from + 1}`);
    context.ROI_xywh = roi ? roi.join(',') : '';
    context.ROI_selection_origin = roiOrigin; context.diagnostic_ROI_id = diagnosticROI;
    context.actual_view_mode = videoObservation ? (video.playbackRate === 1 ? '视频原速' : '视频慢放') : (mode === 'frame' ? '逐帧PNG' : '相邻帧PNG');
    context.actual_playback_rate = videoObservation ? video.playbackRate : '';
    context.video_playback_rate_setting = video.playbackRate;
    context.context_source = videoObservation ? 'user-selected video observation; frame pending PNG localization' : 'user-selected exact PNG viewer';
    const destination = new URL(config.review_url, location.href);
    destination.hash = `context=${encodeURIComponent(JSON.stringify(context))}`;
    location.assign(destination.href);
  }
  $('png-note').addEventListener('click', () => reviewContext(false));
  $('video-note').addEventListener('click', () => reviewContext(true));

  for (let t = 0; t < 25; t++) {
    const card = document.createElement('div'); card.className = 'atb-thumbnail'; card.dataset.frame = t;
    const button = document.createElement('button'); button.title = `同步定位帧${t}`;
    const preview = new Image(); preview.src = png(config.panels[0], t); preview.alt = `输入RGB帧${t}预览`; preview.loading = 'lazy';
    const label = document.createElement('span'); label.textContent = `帧${t}`; button.append(preview, label);
    button.addEventListener('click', () => { setFrame(t); $('synchronous-panels').scrollIntoView({block:'start'}); });
    const originals = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = '打开完整尺寸PNG'; originals.append(summary);
    config.panels.forEach(panel => originals.append(originalLink(panel, t)));
    card.append(button, originals); $('all-frames').append(card);
  }
  render();
})();
