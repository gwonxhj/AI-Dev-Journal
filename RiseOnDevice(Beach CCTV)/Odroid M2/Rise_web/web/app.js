async function jget(url){
  const r = await fetch(url, { cache: "no-store" });
  return await r.json();
}

function setHint(id, v){ document.getElementById(id).innerText = v; }

let dirty = false;        // 사용자가 UI를 만졌는지
let initialized = false;  // 최초 1회 동기화 여부
let lastApplied = null;

function markDirty(){ 
  dirty = true;
  document.getElementById('apply').classList.add('dirty');
}

async function loadVideos(){
  const vids = await jget('/api/videos');
  const sel = document.getElementById('video');
  sel.innerHTML = '';
  vids.forEach(v=>{
    const o = document.createElement('option');
    o.value = v; o.textContent = v;
    sel.appendChild(o);
  });
}

async function refreshStatus(){
  const s = await jget('/api/status');

  // 상태 텍스트는 항상 갱신
  document.getElementById('status').innerText =
    `model=${s.backend}/${s.size}  rgb=${s.rgb}  conf=${s.conf.toFixed(2)} nms=${s.nms.toFixed(2)}  infer=${s.infer_ms.toFixed(1)}ms  det=${s.det_count}`;

  // ✅ UI 컨트롤은 "초기 1회" 또는 "dirty가 아닐 때만" 서버값으로 동기화
  if (!initialized || !dirty) {
    initialized = true;

    document.getElementById('backend').value = s.backend;
    document.getElementById('size').value = s.size;
    document.getElementById('rgb').checked = s.rgb;

    const conf = document.getElementById('conf');
    conf.value = s.conf;
    setHint('confv', s.conf.toFixed(2));

    const nms = document.getElementById('nms');
    nms.value = s.nms;
    setHint('nmsv', s.nms.toFixed(2));

    const vsel = document.getElementById('video');
    if ([...vsel.options].some(o=>o.value===s.video)) vsel.value = s.video;

    // ✅ JPEG Quality도 서버값으로 동기화
    const jpgq = document.getElementById('jpgq');
    if (s.jpeg_quality !== undefined) {
      jpgq.value = s.jpeg_quality;
      setHint('jpgqv', `${s.jpeg_quality}`);
    }
  }
}

async function apply(){
  const backend = document.getElementById('backend').value;
  const size = document.getElementById('size').value;
  const rgb = document.getElementById('rgb').checked ? '1' : '0';
  const conf = document.getElementById('conf').value;
  const nms  = document.getElementById('nms').value;
  const video = document.getElementById('video').value;
  const jpgq  = document.getElementById('jpgq').value;

  const btn = document.getElementById('apply');
  btn.disabled = true;
  btn.innerText = "Applying...";

  // (선택) skip UI가 있으면 같이 보냄
  const skipEl = document.getElementById('skip');
  const skip = skipEl ? skipEl.value : null;

  const nowKey = { backend, size, rgb, video };
  let needStreamReload = false;
  if (!lastApplied) needStreamReload = true;
  else {
    needStreamReload =
      lastApplied.backend !== nowKey.backend ||
      lastApplied.size    !== nowKey.size    ||
      lastApplied.rgb     !== nowKey.rgb     ||
      lastApplied.video   !== nowKey.video;
  }

  let url =
    `/api/set?backend=${encodeURIComponent(backend)}` +
    `&size=${encodeURIComponent(size)}` +
    `&rgb=${encodeURIComponent(rgb)}` +
    `&conf=${encodeURIComponent(conf)}` +
    `&nms=${encodeURIComponent(nms)}` +
    `&video=${encodeURIComponent(video)}` +
    `&jpgq=${encodeURIComponent(jpgq)}`;

  if (skip !== null) url += `&skip=${encodeURIComponent(skip)}`;

  try {
    await jget(url);

    lastApplied = nowKey;
    dirty = false;
    btn.classList.remove('dirty');
    return needStreamReload;

  } finally {
    btn.innerText = "Apply";
    btn.disabled = false;
  }
}

function reloadStream(){
  const img = document.getElementById('stream');
  if (!img) return;
  // 연결을 강제로 새로 열기 (캐시 방지용 timestamp)
  img.src = '';
  setTimeout(() => {
    img.src = `/stream.mjpg?t=${Date.now()}`;
  }, 100);
}

// 스트림이 깨지면 자동으로 재연결(중요)
function bindStreamAutoReconnect(){
  const img = document.getElementById('stream');
  if (!img) return;

  img.addEventListener('error', ()=>{
    // 잠깐 쉬고 재시도
    setTimeout(()=>reloadStream(), 500);
  });
}

function bind(){
  const backend = document.getElementById('backend');
  const size = document.getElementById('size');
  const rgb = document.getElementById('rgb');
  const video = document.getElementById('video');

  //bindStreamAutoReconnect();
  backend.addEventListener('change', markDirty);
  size.addEventListener('change', markDirty);
  rgb.addEventListener('change', markDirty);

  video.addEventListener('change', ()=>{
    markDirty();
  });


  const conf = document.getElementById('conf');
  conf.addEventListener('input', ()=>{
    markDirty();
    setHint('confv', (+conf.value).toFixed(2));
  });

  const nms = document.getElementById('nms');
  nms.addEventListener('input', ()=>{
    markDirty();
    setHint('nmsv', (+nms.value).toFixed(2));
  });

  const jpgq = document.getElementById('jpgq');
  jpgq.addEventListener('input', ()=>{
    markDirty();
    setHint('jpgqv', `${jpgq.value}`);
  });

  document.getElementById('apply').addEventListener('click', async ()=>{
    const needReload = await apply();
    if (needReload) reloadStream();
    await refreshStatus();
  });
}

(async ()=>{
  await loadVideos();
  bind();
  await refreshStatus();
  setInterval(refreshStatus, 1000);
})();
