/**
 * Vosper — Dark Mode Fixed Frontend
 */

'use strict';

const CONFIG = {
  apiBase:       'http://localhost:8000',
  wsPath:        '/ws/live',
  healthInterval: 30_000,
};

const WS_URL = CONFIG.apiBase.replace(/^http/, 'ws') + CONFIG.wsPath;

const State = {
  currentResult: null,
  selectedFile:  null,
  liveStream:    null,
  liveWs:        null,
  liveCtx:       null,
  liveTimer:     null,
  liveSecs:      0,
};

async function checkHealth() {
  try {
    const r = await fetch(`${CONFIG.apiBase}/health`, {
      signal: AbortSignal.timeout(4000),
    });
    r.ok ? setApiStatus('ok', 'API Online') : setApiStatus('err', `HTTP ${r.status}`);
  } catch {
    setApiStatus('err', 'API Offline');
  }
}

function setApiStatus(state, text) {
  const pill = document.getElementById('api-pill');
  const dot  = document.getElementById('api-dot');
  const txt  = document.getElementById('api-txt');

  pill.className = `status-pill ${state === 'ok' ? 'api-ok' : 'api-err'}`;
  txt.textContent = text;
}

function switchTab(tab) {
  const isUpload = tab === 'upload';

  document.getElementById('sec-upload').classList.toggle('section-active', isUpload);
  document.getElementById('sec-upload').classList.toggle('hidden', !isUpload);
  document.getElementById('sec-live').classList.toggle('section-active', !isUpload);
  document.getElementById('sec-live').classList.toggle('hidden', isUpload);

  const tabUpload = document.getElementById('tab-upload');
  const tabLive   = document.getElementById('tab-live');

  tabUpload.classList.toggle('tab-active', isUpload);
  tabUpload.setAttribute('aria-selected', isUpload);
  tabLive.classList.toggle('tab-active', !isUpload);
  tabLive.setAttribute('aria-selected', !isUpload);
}

function onDrop(event) {
  event.preventDefault();
  document.getElementById('drop-zone').classList.remove('drop-zone-active');
  const file = event.dataTransfer?.files?.[0];
  if (file) loadFile(file);
}

function onFilePick(input) {
  if (input.files?.[0]) loadFile(input.files[0]);
}

function loadFile(file) {
  State.selectedFile = file;

  const objectUrl = URL.createObjectURL(file);
  const vidEl     = document.getElementById('vid-preview');
  vidEl.src       = objectUrl;

  document.getElementById('file-name').textContent = file.name;
  document.getElementById('file-size').textContent =
    `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.type || 'unknown type'}`;

  document.getElementById('drop-zone').classList.add('hidden');
  document.getElementById('file-card').classList.remove('hidden');
  document.getElementById('upload-error').classList.add('hidden');
}

function resetFile() {
  State.selectedFile = null;

  const vid = document.getElementById('vid-preview');
  URL.revokeObjectURL(vid.src);
  vid.src = '';

  document.getElementById('drop-zone').classList.remove('hidden');
  document.getElementById('file-card').classList.add('hidden');
  document.getElementById('upload-progress').classList.add('hidden');
  document.getElementById('file-input').value = '';
}

async function doTranscribe() {
  if (!State.selectedFile) return;

  const btn = document.getElementById('transcribe-btn');
  btn.disabled = true;
  document.getElementById('upload-error').classList.add('hidden');
  document.getElementById('upload-progress').classList.remove('hidden');
  hideDashboard();

  for (let i = 0; i < 4; i++) setPipeStep(i, '');

  setProgress('Uploading file…', 15, 'Sending to server');
  setPipeStep(0, 'active');

  const form = new FormData();
  form.append('file', State.selectedFile);

  try {
    await sleep(200);
    setProgress('FFmpeg → WAV…', 30, 'Extracting audio track');
    setPipeStep(0, 'done');
    setPipeStep(1, 'active');

    await sleep(200);
    setProgress('MarbleNet VAD…', 55, 'Detecting speech segments');

    const response = await fetch(`${CONFIG.apiBase}/transcribe`, {
      method: 'POST',
      body:   form,
    });

    setPipeStep(1, 'done');
    setPipeStep(2, 'active');
    setPipeStep(3, 'active');
    setProgress('Vosk + Whisper (parallel)…', 80, 'Running dual transcription');

    if (!response.ok) {
      const err = await response.json().catch(() => ({ message: `HTTP ${response.status}` }));
      throw new Error(err.message || `HTTP ${response.status}`);
    }

    const data = await response.json();

    for (let i = 0; i < 4; i++) setPipeStep(i, 'done');
    setProgress('Complete!', 100, '');
    await sleep(500);

    document.getElementById('upload-progress').classList.add('hidden');

    State.currentResult = { ...data, filename: State.selectedFile.name, source: 'upload' };
    showDashboard(State.currentResult);

  } catch (err) {
    document.getElementById('upload-progress').classList.add('hidden');
    document.getElementById('upload-error').classList.remove('hidden');
    document.getElementById('upload-error-msg').textContent = err.message;
    for (let i = 0; i < 4; i++) setPipeStep(i, '');
  } finally {
    btn.disabled = false;
  }
}

function setProgress(label, pct, note) {
  document.getElementById('prog-label').textContent = label;
  document.getElementById('prog-pct').textContent   = `${pct}%`;
  document.getElementById('prog-fill').style.width  = `${pct}%`;
  document.getElementById('prog-note').textContent  = note;
}

function setPipeStep(idx, state) {
  const el = document.getElementById(`pipe-${idx}`);
  el.className = ['pipe-step', state === 'active' ? 'pipe-active' : state === 'done' ? 'pipe-done' : '']
    .filter(Boolean).join(' ');
}

async function startLive() {
  document.getElementById('live-error').classList.add('hidden');

  try {
    State.liveStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  } catch {
    try {
      State.liveStream = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
    } catch (err) {
      showLiveError(`Microphone access denied: ${err.message}`);
      return;
    }
  }

  const vid = document.getElementById('lvid');
  vid.srcObject = State.liveStream;
  vid.classList.remove('hidden');
  vid.play();
  document.getElementById('cam-placeholder').classList.add('hidden');
  document.getElementById('cam-hud').classList.remove('hidden');
  document.getElementById('live-partial-box').classList.remove('hidden');
  hideDashboard();

  State.liveCtx = new AudioContext({ sampleRate: 16_000 });
  const source  = State.liveCtx.createMediaStreamSource(State.liveStream);
  await State.liveCtx.audioWorklet.addModule(makePcmProcessorUrl());
  const worklet = new AudioWorkletNode(State.liveCtx, 'pcm-processor');

  State.liveWs = new WebSocket(WS_URL);
  setLivePartial('', false);

  State.liveWs.onopen = () => {
    worklet.port.onmessage = (e) => {
      if (State.liveWs?.readyState === WebSocket.OPEN) {
        State.liveWs.send(e.data);
      }
    };
    source.connect(worklet);
    worklet.connect(State.liveCtx.destination);
  };

  State.liveWs.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    switch (msg.type) {
      case 'partial':
        setLivePartial(msg.text, true);
        break;
      case 'status':
        setLivePartial(msg.msg, false);
        break;
      case 'final':
        stopLive();
        State.currentResult = {
          ...msg,
          source:       'live',
          duration_sec: State.liveSecs,
        };
        showDashboard(State.currentResult);
        break;
      case 'error':
        showLiveError(msg.msg);
        break;
    }
  };

  State.liveWs.onerror = () => showLiveError('WebSocket connection failed');

  State.liveSecs = 0;
  State.liveTimer = setInterval(() => {
    State.liveSecs++;
    const m = String(Math.floor(State.liveSecs / 60)).padStart(2, '0');
    const s = String(State.liveSecs % 60).padStart(2, '0');
    document.getElementById('rec-timer').textContent = `${m}:${s}`;
  }, 1000);

  document.getElementById('live-controls').innerHTML = `
    <button class="btn btn-danger btn-lg w-full" onclick="stopAndProcess()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
      </svg>
      Stop & transcribe
    </button>`;
}

function stopAndProcess() {
  if (State.liveWs?.readyState === WebSocket.OPEN) {
    State.liveWs.send(new TextEncoder().encode('__END__').buffer);
  } else {
    stopLive();
  }
}

function stopLive() {
  clearInterval(State.liveTimer);

  State.liveStream?.getTracks().forEach((t) => t.stop());
  State.liveCtx?.close();
  State.liveWs?.close();

  State.liveStream = null;
  State.liveCtx    = null;
  State.liveWs     = null;

  const vid = document.getElementById('lvid');
  vid.srcObject = null;
  vid.classList.add('hidden');
  document.getElementById('cam-placeholder').classList.remove('hidden');
  document.getElementById('cam-hud').classList.add('hidden');
  document.getElementById('live-partial-box').classList.add('hidden');

  document.getElementById('live-controls').innerHTML = `
    <button class="btn btn-primary btn-lg w-full" onclick="startLive()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/>
      </svg>
      Start recording
    </button>`;
}

function setLivePartial(text, hasContent) {
  const el  = document.getElementById('lpartial');
  el.textContent = hasContent ? text : 'Waiting for speech…';
  el.className   = hasContent
    ? 'text-base leading-relaxed text-slate-100 min-h-[48px]'
    : 'text-base leading-relaxed text-slate-500 italic min-h-[48px]';
}

function showLiveError(msg) {
  document.getElementById('live-error').classList.remove('hidden');
  document.getElementById('live-error-msg').textContent = msg;
}

function makePcmProcessorUrl() {
  const code = `
    class PcmProcessor extends AudioWorkletProcessor {
      process(inputs) {
        const channel = inputs[0]?.[0];
        if (!channel) return true;
        const buf = new Int16Array(channel.length);
        for (let i = 0; i < channel.length; i++) {
          buf[i] = Math.max(-32768, Math.min(32767, channel[i] * 32768));
        }
        this.port.postMessage(buf.buffer, [buf.buffer]);
        return true;
      }
    }
    registerProcessor('pcm-processor', PcmProcessor);
  `;
  return URL.createObjectURL(new Blob([code], { type: 'application/javascript' }));
}

function showDashboard(r) {
  document.getElementById('sec-upload').classList.remove('section-active');
  document.getElementById('sec-upload').classList.add('hidden');
  document.getElementById('sec-live').classList.remove('section-active');
  document.getElementById('sec-live').classList.add('hidden');

  const dash = document.getElementById('dashboard');
  dash.classList.remove('hidden');
  dash.classList.add('flex');

  const srcLabel = r.source === 'live'
    ? '● Live recording'
    : `↑ ${r.filename || 'Upload'}`;

  document.getElementById('dash-meta').innerHTML =
    `${esc(srcLabel)} <span class="badge badge-jade">✓ Complete</span>`;

  animateValue('stat-duration', 0, r.duration_sec || 0, 1000, (v) => v.toFixed(1));
  animateValue('stat-vad', 0, (r.vad_segments || []).length, 800, (v) => Math.round(v));
  animateValue('stat-whisper', 0, r.whisper?.word_count ?? 0, 800, (v) => Math.round(v));
  animateValue('stat-vosk', 0, r.vosk?.word_count ?? 0, 800, (v) => Math.round(v));

  renderTranscript('whisper-text', r.whisper?.text);
  renderWhisperSegments(r.whisper?.segments || []);
  renderTranscript('vosk-text', r.vosk?.text);
  renderWordChips(r.vosk?.words || []);
  renderVad(r.vad_segments || [], r.duration_sec || 1);
  renderTiming(r.timing || {});
}

function hideDashboard() {
  const dash = document.getElementById('dashboard');
  dash.classList.add('hidden');
  dash.classList.remove('flex');
}

function animateValue(id, start, end, duration, formatter) {
  const el = document.getElementById(id);
  const startTime = performance.now();
  
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easeProgress = 1 - Math.pow(1 - progress, 3);
    const current = start + (end - start) * easeProgress;
    
    el.textContent = formatter(current);
    
    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }
  
  requestAnimationFrame(update);
}

function renderTranscript(elId, text) {
  const el    = document.getElementById(elId);
  const clean = text?.trim();
  el.textContent = clean || 'No transcription';
  el.className   = `transcript-box${clean ? '' : ' empty'}`;
}

function renderWhisperSegments(segments) {
  const container = document.getElementById('whisper-segments');
  container.innerHTML = '';

  if (!segments.length) {
    container.innerHTML = '<span class="text-xs text-slate-600 px-2">No segments</span>';
    return;
  }

  segments.forEach((seg, i) => {
    const row       = document.createElement('div');
    row.className   = 'seg-row';
    row.setAttribute('role', 'listitem');
    row.style.opacity = '0';
    row.innerHTML   = `
      <span class="seg-time">${seg.start.toFixed(2)}→${seg.end.toFixed(2)}s</span>
      <span class="seg-text">${esc(seg.text)}</span>`;
    container.appendChild(row);
    
    setTimeout(() => {
      row.style.transition = 'opacity 0.3s ease';
      row.style.opacity = '1';
    }, i * 50);
  });
}

function renderWordChips(words) {
  const container = document.getElementById('vosk-words');
  container.innerHTML = '';

  if (!words.length) {
    container.innerHTML = '<span class="text-xs text-slate-600 px-2">No words detected</span>';
    return;
  }

  const MAX_SHOWN = 120;
  words.slice(0, MAX_SHOWN).forEach((w, i) => {
    const chip       = document.createElement('span');
    chip.className   = 'word-chip';
    chip.textContent = w.word;
    chip.title       = `${w.start}s — conf: ${w.conf}`;
    chip.setAttribute('role', 'listitem');
    chip.style.opacity = '0';
    container.appendChild(chip);
    
    setTimeout(() => {
      chip.style.transition = 'opacity 0.2s ease';
      chip.style.opacity = '1';
    }, i * 15);
  });

  if (words.length > MAX_SHOWN) {
    const more       = document.createElement('span');
    more.className   = 'word-chip';
    more.style.color = 'var(--text-muted)';
    more.style.fontWeight = '600';
    more.textContent = `+${words.length - MAX_SHOWN} more`;
    container.appendChild(more);
  }
}

function renderVad(segments, durationSec) {
  document.getElementById('vad-badge').textContent =
    `${segments.length} segment${segments.length !== 1 ? 's' : ''}`;

  const container = document.getElementById('vad-rows');
  container.innerHTML = '';

  if (!segments.length) {
    container.innerHTML = '<span class="text-xs text-slate-600 px-2">No segments detected</span>';
    return;
  }

  const dur = Math.max(durationSec, segments.at(-1)?.end || 1);

  segments.forEach((seg, i) => {
    const pLeft  = (seg.start / dur * 100).toFixed(1);
    const pWidth = Math.max(0.5, (seg.end - seg.start) / dur * 100).toFixed(1);

    const row       = document.createElement('div');
    row.className   = 'vad-row';
    row.setAttribute('role', 'listitem');
    row.style.opacity = '0';
    row.innerHTML   = `
      <span class="vad-index">#${i + 1}</span>
      <div class="vad-track" aria-label="Segment from ${seg.start.toFixed(2)}s to ${seg.end.toFixed(2)}s">
        <div class="vad-seg" style="left:${pLeft}%; width:0%"></div>
      </div>
      <span class="vad-times">${seg.start.toFixed(2)}→${seg.end.toFixed(2)}s</span>`;
    container.appendChild(row);
    
    setTimeout(() => {
      row.style.transition = 'opacity 0.3s ease';
      row.style.opacity = '1';
      const bar = row.querySelector('.vad-seg');
      bar.style.transition = 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)';
      bar.style.width = `${pWidth}%`;
    }, i * 80);
  });
}

function renderTiming(timing) {
  const container = document.getElementById('timing-rows');
  container.innerHTML = '';

  const steps = [
    { name: 'ffmpeg',         ms: timing.ffmpeg_ms,   color: '#64748b' },
    { name: 'marblenet-vad',  ms: timing.vad_ms,      color: '#f59e0b' },
    { name: 'vosk ‖ whisper', ms: timing.parallel_ms, color: '#06b6d4' },
    { name: 'total',          ms: timing.total_ms,    color: '#10b981' },
  ];

  if (!timing.total_ms) {
    container.innerHTML = '<span class="text-xs text-slate-600 px-2">Not available (live mode)</span>';
    return;
  }

  const maxMs = Math.max(...steps.map((s) => s.ms || 0), 1);

  steps.forEach(({ name, ms, color }, i) => {
    if (!ms) return;
    const row       = document.createElement('div');
    row.className   = 'timing-row';
    row.setAttribute('role', 'listitem');
    row.style.opacity = '0';
    row.innerHTML   = `
      <span class="timing-name">${name}</span>
      <div class="timing-bar">
        <div class="timing-fill" style="width:0%; background:${color}"></div>
      </div>
      <span class="timing-val">${(ms / 1000).toFixed(2)}s</span>`;
    container.appendChild(row);
    
    setTimeout(() => {
      row.style.transition = 'opacity 0.3s ease';
      row.style.opacity = '1';
      const fill = row.querySelector('.timing-fill');
      fill.style.width = `${(ms / maxMs * 100).toFixed(0)}%`;
    }, i * 100);
  });
}

function doExport() {
  if (!State.currentResult) return;
  const blob = new Blob(
    [JSON.stringify(State.currentResult, null, 2)],
    { type: 'application/json' }
  );
  const url = URL.createObjectURL(blob);
  const a   = Object.assign(document.createElement('a'), {
    href:     url,
    download: `vosper-${Date.now()}.json`,
  });
  a.click();
  URL.revokeObjectURL(url);
}

function doReset() {
  hideDashboard();
  State.currentResult = null;
  resetFile();
  stopLive();
  document.getElementById('live-error').classList.add('hidden');

  const onLive = document.getElementById('tab-live').classList.contains('tab-active');
  const active = document.getElementById(onLive ? 'sec-live' : 'sec-upload');
  active.classList.remove('hidden');
  active.classList.add('section-active');
}

function copyText(elId) {
  const text = document.getElementById(elId)?.textContent;
  if (text) {
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.activeElement;
      if (btn) {
        const original = btn.innerHTML;
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
        setTimeout(() => {
          btn.innerHTML = original;
        }, 1500);
      }
    }).catch(() => {});
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

checkHealth();
setInterval(checkHealth, CONFIG.healthInterval);