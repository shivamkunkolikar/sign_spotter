// ── Tab switching ──────────────────────────────────────────────
function switchTab(name) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById('panel-' + name).classList.add('active');
    event.currentTarget.classList.add('active');
}

// ── Webcam ─────────────────────────────────────────────────────
let webcamRunning = false;
let detectionInterval = null;

async function startWebcam() {
    const res = await fetch('/api/webcam/start', { method: 'POST' });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }

    webcamRunning = true;
    const img = document.getElementById('webcamFeed');
    img.src = '/api/webcam/feed?' + Date.now();
    img.style.display = 'block';
    document.getElementById('feedPlaceholder').style.display = 'none';
    document.getElementById('camDot').classList.add('green');
    document.getElementById('camStatus').textContent = 'live ●';
    document.getElementById('btnStart').disabled = true;
    document.getElementById('btnStop').disabled = false;
    document.getElementById('globalStatus').textContent = '● live detection';
    document.getElementById('globalStatus').classList.add('live');

    // Poll for detections to update the log
    detectionInterval = setInterval(updateDetLog, 1500);
}

async function stopWebcam() {
    await fetch('/api/webcam/stop', { method: 'POST' });
    webcamRunning = false;
    clearInterval(detectionInterval);
    const img = document.getElementById('webcamFeed');
    img.style.display = 'none';
    document.getElementById('feedPlaceholder').style.display = 'block';
    document.getElementById('camDot').classList.remove('green');
    document.getElementById('camStatus').textContent = 'camera feed';
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnStop').disabled = true;
    document.getElementById('globalStatus').textContent = '● model ready';
    document.getElementById('globalStatus').classList.remove('live');
}

// Grab a frame and detect (for the sidebar log)
async function updateDetLog() {
    if (!webcamRunning) return;
    try {
        const res = await fetch('/api/webcam/snapshot');
        if (!res.ok) return;
        const data = await res.json();
        renderDetLog(data.detections || [], 'detLog');
    } catch (e) { }
}

// ── Video upload (streaming mode) ─────────────────────────────
let videoStateInterval = null;

function handleVideoDrop(e) {
    e.preventDefault();
    document.getElementById('videoDrop').classList.remove('drag');
    const file = e.dataTransfer.files[0];
    if (file) uploadVideo(file);
}
function handleVideoFile(file) { if (file) uploadVideo(file); }

async function uploadVideo(file) {
    // show uploading spinner
    const ind = document.getElementById('uploadingIndicator');
    ind.style.display = 'flex';

    const form = new FormData();
    form.append('video', file);
    const res = await fetch('/api/video/upload', { method: 'POST', body: form });
    const data = await res.json();
    ind.style.display = 'none';

    if (data.error) { alert(data.error); return; }

    // switch to player view
    document.getElementById('videoUploadSection').style.display = 'none';
    document.getElementById('videoPlayerSection').style.display = 'block';
    document.getElementById('vidStatus').textContent =
        `ready · ${data.total_frames} frames · ${data.fps} fps`;
}

async function videoPlay() {
    await fetch('/api/video/start', { method: 'POST' });

    const feed = document.getElementById('videoFeed');
    feed.src = '/api/video/feed?' + Date.now();
    feed.style.display = 'block';
    document.getElementById('videoPlaceholder').style.display = 'none';
    document.getElementById('vidDot').classList.add('green');
    document.getElementById('vidStatus').textContent = 'playing ●';
    document.getElementById('vidBtnPlay').disabled = true;
    document.getElementById('vidBtnPause').disabled = false;
    document.getElementById('vidBtnStop').disabled = false;

    // poll state for detection log + progress
    videoStateInterval = setInterval(pollVideoState, 800);
}

async function videoPause() {
    const res = await fetch('/api/video/pause', { method: 'POST' });
    const data = await res.json();
    const btn = document.getElementById('vidBtnPause');
    const dot = document.getElementById('vidDot');
    if (data.paused) {
        btn.textContent = 'Resume';
        dot.classList.remove('green');
        dot.classList.add('amber');
        document.getElementById('vidStatus').textContent = 'paused ‖';
    } else {
        btn.textContent = 'Pause';
        dot.classList.add('green');
        dot.classList.remove('amber');
        document.getElementById('vidStatus').textContent = 'playing ●';
    }
}

async function videoStop() {
    clearInterval(videoStateInterval);
    await fetch('/api/video/stop', { method: 'POST' });
    const feed = document.getElementById('videoFeed');
    feed.style.display = 'none';
    document.getElementById('videoPlaceholder').style.display = 'block';
    document.getElementById('vidDot').classList.remove('green', 'amber');
    document.getElementById('vidStatus').textContent = 'stopped';
    document.getElementById('vidBtnPlay').disabled = false;
    document.getElementById('vidBtnPause').disabled = true;
    document.getElementById('vidBtnStop').disabled = true;
    document.getElementById('vidBtnPause').textContent = 'Pause';
    document.getElementById('vidProgress').textContent = '0%';
}

function videoReload() {
    videoStop();
    setTimeout(() => {
        document.getElementById('videoUploadSection').style.display = 'block';
        document.getElementById('videoPlayerSection').style.display = 'none';
    }, 200);
}

async function pollVideoState() {
    const res = await fetch('/api/video/state');
    const data = await res.json();

    // progress
    document.getElementById('vidProgress').textContent = data.progress + '%';

    // detection log (reuse same renderer as webcam)
    renderDetLog(data.detections, 'videoDetLog');

    // cumulative signs seen
    const seen = document.getElementById('videoSignsSeen');
    if (data.seen_signs.length) {
        seen.innerHTML = data.seen_signs
            .map(s => `<span class="sign-tag">${s}</span>`).join('');
    }

    // auto-stop UI when video loops back (progress resets)
}

function renderDetLog(dets, targetId) {
    const log = document.getElementById(targetId || 'detLog');
    if (!dets || !dets.length) {
        log.innerHTML = '<div class="no-det">no signs in frame</div>';
        return;
    }
    log.innerHTML = dets.map(d => `
    <div class="det-item">
      <div class="det-name">${d.label}</div>
      <div class="det-conf">${d.confidence}% confidence</div>
      <div class="conf-bar"><div class="conf-fill" style="width:${d.confidence}%"></div></div>
    </div>
  `).join('');
}

// ── Photo classify ─────────────────────────────────────────────
function handlePhotoDrop(e) {
    e.preventDefault();
    document.getElementById('photoDrop').classList.remove('drag');
    const file = e.dataTransfer.files[0];
    if (file) classifyPhoto(file);
}
function handlePhotoFile(file) { if (file) classifyPhoto(file); }

async function classifyPhoto(file) {
    const form = new FormData();
    form.append('image', file);
    const res = await fetch('/api/classify', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }

    document.getElementById('photoThumb').src = 'data:image/jpeg;base64,' + data.thumbnail;
    const top = data.predictions[0].confidence;
    document.getElementById('predList').innerHTML = data.predictions.map((p, i) => `
    <div class="pred-item ${i === 0 ? 'top1' : ''}">
      <span class="pred-rank">${i + 1}</span>
      <span class="pred-name">${p.class}</span>
      <div class="pred-bar-bg"><div class="pred-bar-fill" style="width:${(p.confidence / top * 100).toFixed(0)}%"></div></div>
      <span class="pred-pct">${p.confidence}%</span>
    </div>
  `).join('');
    document.getElementById('photoResultCard').style.display = 'block';
}

