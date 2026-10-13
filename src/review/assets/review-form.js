(() => {
  'use strict';
  const config = JSON.parse(document.getElementById('atb-review-config').textContent);
  const storeKey = config.store_key; // B1 key stays atb-failure-B1-user-20261009.
  const status = document.getElementById('status');
  const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  const pending = new Set([...config.visual, 'severity', 'confidence', 'normal_motion_or_reasonable_defocus', 'similar_in_input_or_GT']);
  let saved = [], storageHealthy = true, rawSaved = null;
  try {
    rawSaved = localStorage.getItem(storeKey);
    if (rawSaved !== null) {
      saved = JSON.parse(rawSaved);
      if (!Array.isArray(saved) || saved.some(r => !r || typeof r !== 'object' || Array.isArray(r))) throw new Error('存储格式不是记录数组');
    }
  } catch (error) {
    storageHealthy = false;
    status.textContent = '旧存储无法读取；保留原存储，不写回覆盖。当前页面仍可独立导出。';
  }

  const newRecord = metadata => ({...metadata, package:config.package,
    ...Object.fromEntries(config.fields.map(k => [k, pending.has(k) ? '待用户复核' : '']))});

  function add(row) {
    const base = row ? {...row} : newRecord(config.samples[0]);
    const section = document.createElement('article'); section.dataset.kind = 'review'; section.atbBase = base;
    const sel = document.createElement('select'); sel.setAttribute('aria-label', '样本');
    if (!config.samples.some(s => s.sample_id === base.sample_id)) {
      const option = document.createElement('option'); option.value = base.sample_id || '';
      option.textContent = `保留原记录样本：${base.sample_id || '未指定'}`; sel.append(option);
    }
    config.samples.forEach(s => {
      const option = document.createElement('option'); option.value = s.sample_id;
      option.textContent = `${s.sample_id} ${s.dataset} ${s.scene_id}`; sel.append(option);
    });
    sel.value = base.sample_id || ''; section.append(sel);
    config.fields.forEach((k, i) => {
      const label = document.createElement('label'); label.textContent = config.labels[i];
      const textarea = document.createElement('textarea'); textarea.dataset.key = k;
      textarea.value = own(base, k) ? (base[k] ?? '') : (pending.has(k) ? '待用户复核' : '');
      textarea.dataset.initialText = textarea.value;
      label.append(textarea); section.append(label);
    });
    if (own(base, 'video_playback_rate_setting')) {
      const info = document.createElement('p'); info.textContent = `定位时播放器设置：${base.video_playback_rate_setting}×。PNG逐帧自身没有播放倍速。`; section.append(info);
    }
    document.getElementById('forms').append(section);
  }

  function rows() {
    return [...document.querySelectorAll('[data-kind=review]')].map(section => {
      const base = section.atbBase; const row = {...base};
      const selected = section.querySelector('select').value;
      if (selected !== base.sample_id) Object.assign(row, config.samples.find(s => s.sample_id === selected) || {sample_id:selected});
      section.querySelectorAll('textarea').forEach(t => {
        const k = t.dataset.key;
        // Preserve absent fields, unknown keys, numeric values and existing text on load.
        if (t.value === t.dataset.initialText) return;
        row[k] = t.value;
      });
      return row;
    });
  }

  function saveLocal() {
    if (!storageHealthy) return;
    try { localStorage.setItem(storeKey, JSON.stringify(rows())); status.textContent = '已保存在当前浏览器；请另导出文件保存。'; }
    catch (error) { status.textContent = '浏览器未允许存储；请导出JSON保存，原有内容未清理。'; }
  }

  function exportData() {
    saveLocal();
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const blob = new Blob([JSON.stringify(rows(), null, 2)], {type:'application/json'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = `ATB-${config.package}-user-review-${stamp}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  (saved.length ? saved : config.samples.map(s => newRecord(s))).forEach(add);
  document.getElementById('add-record').addEventListener('click', () => { add(); saveLocal(); });
  document.getElementById('export-records').addEventListener('click', exportData);
  document.getElementById('forms').addEventListener('input', saveLocal);
  document.getElementById('forms').addEventListener('change', saveLocal);
  // Existing function names remain available for prior page integrations.
  Object.assign(window, {add, rows, saveLocal, exportData});

  const match = location.hash.match(/^#context=(.*)$/);
  if (!match) return;
  try {
    const raw = JSON.parse(decodeURIComponent(match[1]));
    const sample = config.samples.find(s => s.sample_id === raw.sample_id);
    if (!sample || raw.package !== config.package) throw new Error('定位样本或包不匹配');
    const position = raw.frame_id_or_transition_id || '';
    if (position && !/^(\d|1\d|2[0-4])$/.test(position)) {
      const transition = position.match(/^(\d+)→(\d+)$/);
      if (!transition || Number(transition[1]) > 23 || Number(transition[2]) !== Number(transition[1]) + 1) throw new Error('不是0–24内的连续transition（24→0是循环）');
    }
    const roi = raw.ROI_xywh || '';
    if (roi) {
      const box = roi.split(',').map(Number);
      if (box.length !== 4 || !box.every(Number.isInteger) || box[0] < 0 || box[1] < 0 || box[2] < 1 || box[3] < 1 || box[0]+box[2] > 1024 || box[1]+box[3] > 576) throw new Error('ROI超出原图范围');
    }
    const rate = raw.actual_playback_rate === '' ? '' : Number(raw.actual_playback_rate);
    if (rate !== '' && (!Number.isFinite(rate) || rate <= 0 || rate > 16)) throw new Error('倍速不适用');
    const context = newRecord(sample);
    // Accept location metadata only; no observations or visual conclusions are imported.
    Object.assign(context, {frame_id_or_transition_id:position, ROI_xywh:roi,
      actual_view_mode:String(raw.actual_view_mode || ''), actual_playback_rate:rate,
      video_playback_rate_setting:raw.video_playback_rate_setting, context_source:String(raw.context_source || '')});
    const banner = document.getElementById('context-banner'); banner.hidden = false;
    document.getElementById('context-description').textContent = `${sample.sample_id} ${sample.scene_id}：${position || '帧待逐帧定位'}；ROI ${roi || '未选'}；${context.actual_view_mode}；实际视频倍速 ${rate === '' ? '不适用/待填写' : rate+'×'}。仅在点击下面按钮后新增，不覆盖既有记录。`;
    document.getElementById('import-context').addEventListener('click', () => { add(context); saveLocal(); });
  } catch (error) { status.textContent = `定位信息未导入：${error.message}。已有记录保留。`; }
})();
