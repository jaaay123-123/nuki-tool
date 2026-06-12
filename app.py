#!/usr/bin/env python3
import os, io, base64, tempfile, threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from PIL import Image

app = Flask(__name__)

# CPU 코어 수 기준으로 동시 처리 수 제한 (메모리 스래싱 방지)
_WORKERS = min(os.cpu_count() or 4, 6)
_rembg_sem = threading.Semaphore(_WORKERS)

# u2netp: u2net 경량 버전 (모델 크기 4MB, 추론 3배 빠름)
_rembg_session = None
def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2netp")
    return _rembg_session

# ── 누끼 파이프라인 (roomkey-tool/svgtool.py + library.py 직접 이식) ──────────

def _lumakey(im: Image.Image):
    """단색 평탄 배경 누끼 (양극성).
    밝은 배경+어두운 잉크 / 어두운 배경+밝은 잉크 모두 대응.
    조건 안 맞으면 None."""
    try:
        import numpy as np, cv2
    except ImportError:
        return None
    im = im.convert("RGB")
    chk = np.array(im)
    clum = 0.299*chk[...,0] + 0.587*chk[...,1] + 0.114*chk[...,2]
    corners = np.array([clum[0,0], clum[0,-1], clum[-1,0], clum[-1,-1]])
    if float(corners.max() - corners.min()) > 45:
        return None
    bg_lum = float(np.median(corners))
    dark   = clum < bg_lum - 40
    bright = clum > bg_lum + 40
    fdark, fbright = float(dark.mean()), float(bright.mean())
    if max(fdark, fbright) < 0.003:
        return None
    keep_dark = fdark >= fbright
    subj = dark if keep_dark else bright
    ink = float(subj.mean())
    if not (0.003 < ink < 0.6):
        return None
    subj_mean = float(clum[subj].mean())
    t    = (bg_lum + subj_mean) / 2.0
    half = max(8.0, abs(bg_lum - subj_mean) * 0.22)
    long_side = max(im.size)
    if long_side < 1400:
        sc = 1400.0 / long_side
        im = im.resize((round(im.width*sc), round(im.height*sc)), Image.LANCZOS)
    rgb = np.array(im)
    lum = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
    lo, hi = t - half, t + half
    if keep_dark:
        alpha = np.clip((hi - lum) / max(1.0, hi-lo) * 255.0, 0, 255)
    else:
        alpha = np.clip((lum - lo) / max(1.0, hi-lo) * 255.0, 0, 255)
    alpha = cv2.GaussianBlur(alpha.astype("uint8"), (0,0), 1.2)
    return Image.fromarray(np.dstack([rgb, alpha]).astype("uint8"), "RGBA")


def _flat_bg_rgb(im: Image.Image):
    """4 코너가 균일한 단색이면 RGB 반환, 아니면 None."""
    import numpy as np
    rgb = np.array(im.convert("RGB"), dtype="int16")
    h, w = rgb.shape[:2]
    corners = np.array([rgb[0,0], rgb[0,w-1], rgb[h-1,0], rgb[h-1,w-1]], dtype="int16")
    med = np.median(corners, axis=0)
    if int(np.abs(corners - med).max()) < 24:
        return tuple(int(c) for c in med)
    return None


def _matte_bg(im: Image.Image, bg_rgb: tuple, tol: int = 45) -> Image.Image:
    """단색 배경 픽셀 투명화."""
    import numpy as np
    arr = np.array(im.convert("RGBA"))
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    dist = np.maximum(np.maximum(np.abs(r.astype(int)-bg_rgb[0]),
                                  np.abs(g.astype(int)-bg_rgb[1])),
                       np.abs(b.astype(int)-bg_rgb[2]))
    arr[dist < tol, 3] = 0
    return Image.fromarray(arr, "RGBA")


NUKI_METHODS = ["rembg", "lumakey", "matte"]

def strip_bg(im: Image.Image, method: str = "rembg") -> tuple[Image.Image, str]:
    """지정 방법으로 누끼. method: 'rembg' | 'lumakey' | 'matte'"""
    im = im.convert("RGBA")

    if method == "lumakey":
        result = _lumakey(im)
        if result is not None:
            return result, "lumakey"
        # 조건 불충족 시 코너 밝기 기준으로 강제 적용
        try:
            import numpy as np, cv2
            rgb = np.array(im.convert("RGB"))
            lum = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
            corners = np.array([lum[0,0], lum[0,-1], lum[-1,0], lum[-1,-1]])
            bg_lum = float(np.median(corners))
            half = 35.0
            alpha = np.clip((bg_lum + half - lum) / (half * 2) * 255.0, 0, 255)
            alpha = cv2.GaussianBlur(alpha.astype("uint8"), (0,0), 1.2)
            return Image.fromarray(np.dstack([rgb, alpha]).astype("uint8"), "RGBA"), "lumakey"
        except ImportError:
            pass  # numpy/cv2 없으면 rembg로 폴백

    if method == "matte":
        import numpy as np
        flat = _flat_bg_rgb(im)
        if flat is None:
            rgb = np.array(im.convert("RGB"), dtype="int16")
            h, w = rgb.shape[:2]
            corners = np.array([rgb[0,0], rgb[0,w-1], rgb[h-1,0], rgb[h-1,w-1]], dtype="int16")
            flat = tuple(int(c) for c in np.median(corners, axis=0))
        return _matte_bg(im, flat, tol=40), "matte"

    # rembg — 추론용으로 1024px 리사이즈, 알파만 원본 해상도로 복원
    from rembg import remove
    orig_size = im.size
    proc = im.copy()
    if max(proc.size) > 1024:
        proc.thumbnail((1024, 1024), Image.LANCZOS)
    buf_in = io.BytesIO()
    proc.save(buf_in, "PNG")
    with _rembg_sem:
        raw = remove(buf_in.getvalue(), session=get_rembg_session())
    alpha = Image.open(io.BytesIO(raw)).convert("RGBA").getchannel("A")
    alpha = alpha.resize(orig_size, Image.LANCZOS)
    result = im.copy()
    result.putalpha(alpha)
    return result, "rembg"


# ── gpt-image-1 (프롬프트 모드 전용) ─────────────────────────────────────────

def get_client():
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def preprocess_for_api(im: Image.Image):
    orig_w, orig_h = im.size
    size  = max(orig_w, orig_h)
    pad_x = (size - orig_w) // 2
    pad_y = (size - orig_h) // 2
    padded = Image.new("RGBA", (size, size), (255,255,255,255))
    padded.paste(im, (pad_x, pad_y))
    api_im = padded.resize((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    api_im.save(buf, "PNG")
    return buf.getvalue(), {"orig_w":orig_w, "orig_h":orig_h,
                            "pad_x":pad_x, "pad_y":pad_y, "padded_size":size}

def apply_alpha(original: Image.Image, api_result: Image.Image, meta: dict) -> Image.Image:
    alpha = api_result.getchannel("A").resize(
        (meta["padded_size"], meta["padded_size"]), Image.LANCZOS)
    alpha = alpha.crop((meta["pad_x"], meta["pad_y"],
                        meta["pad_x"]+meta["orig_w"], meta["pad_y"]+meta["orig_h"]))
    result = original.copy()
    result.putalpha(alpha)
    return result

def process_with_gpt(image_bytes: bytes, prompt: str) -> bytes:
    from PIL import ImageOps
    client = get_client()
    original = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGBA")
    png_bytes, meta = preprocess_for_api(original)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes); tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resp = client.images.edit(
                model="gpt-image-1", image=f, prompt=prompt,
                background="transparent", quality="medium",
                n=1, size="1024x1024")
    finally:
        os.unlink(tmp_path)
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        img_bytes = base64.b64decode(item.b64_json)
    else:
        import urllib.request
        with urllib.request.urlopen(item.url) as r: img_bytes = r.read()
    api_result = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    result = apply_alpha(original, api_result, meta)
    out = io.BytesIO(); result.save(out, "PNG")
    return out.getvalue()


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.route("/")
def index(): return PAGE

@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()
    if not data:
        return jsonify({"error": "데이터 없음"}), 400
    try:
        image_bytes = base64.b64decode(data["image"].split(",")[-1])
        mode = data.get("mode", "nuki")

        if mode == "nuki":
            from PIL import ImageOps
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
            nuki_method = data.get("nuki_method", "rembg")
            result_im, method = strip_bg(im, nuki_method)
            out = io.BytesIO(); result_im.save(out, "PNG")
            result_bytes = out.getvalue()
        else:
            prompt = data.get("prompt") or "Remove the background completely."
            result_bytes = process_with_gpt(image_bytes, prompt)
            method = "gpt-image-1"

        import uuid
        fname = f"nuki_{uuid.uuid4().hex[:8]}.png"
        return jsonify({
            "result": "data:image/png;base64," + base64.b64encode(result_bytes).decode(),
            "filename": fname,
            "method": method,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>누끼 툴</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Apple SD Gothic Neo",sans-serif;background:#f5f5f7;color:#1d1d1f;min-height:100vh}
.wrap{max-width:1100px;margin:0 auto;padding:40px 24px}
h1{font-size:26px;font-weight:700;margin-bottom:4px}
.sub{color:#86868b;font-size:14px;margin-bottom:28px}

.upload-zone{border:2px dashed #c7c7cc;border-radius:16px;background:#fff;padding:40px;text-align:center;cursor:pointer;transition:all .2s;margin-bottom:28px}
.upload-zone:hover,.upload-zone.over{border-color:#0066cc;background:#f0f5ff}
.upload-zone .icon{font-size:36px;margin-bottom:10px}
.upload-zone p{color:#555;font-size:15px}
.upload-zone .hint{color:#86868b;font-size:13px;margin-top:4px}

/* 전체선택 바 */
.sel-bar{display:none;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap}
.sel-bar label{display:flex;align-items:center;gap:7px;cursor:pointer;font-size:14px;font-weight:600;color:#1d1d1f;user-select:none}
.sel-bar input[type=checkbox]{width:17px;height:17px;accent-color:#0066cc;cursor:pointer}
.sel-info{font-size:13px;color:#86868b;flex:1}
.btn-dl-all{padding:7px 16px;border-radius:9px;border:none;background:#0066cc;color:#fff;font-size:13px;font-weight:600;cursor:pointer;transition:background .12s;display:none}
.btn-dl-all:hover{background:#0055b3}
.btn-dl-all:disabled{background:#b0b0b6;cursor:not-allowed}

.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:20px;margin-bottom:28px}

/* 카드 */
.card{background:#fff;border-radius:16px;box-shadow:0 2px 8px rgba(0,0,0,.08);overflow:hidden;transition:box-shadow .15s}
.card.checked{box-shadow:0 0 0 2.5px #0066cc, 0 2px 12px rgba(0,102,204,.15)}

.card-img{position:relative;background:#e5e5ea;display:flex;align-items:center;justify-content:center}
.card-img img{display:block;width:100%;height:200px;object-fit:contain}

/* 체크박스 */
.card-chk{position:absolute;top:8px;left:8px;z-index:10;cursor:pointer}
.card-chk input{display:none}
.chk-box{display:flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:7px;border:2px solid rgba(255,255,255,.85);background:rgba(0,0,0,.3);backdrop-filter:blur(3px);transition:all .15s}
.card-chk input:checked + .chk-box{background:#0066cc;border-color:#0066cc}
.card-chk input:checked + .chk-box::after{content:'';display:block;width:5px;height:9px;border:2px solid #fff;border-top:none;border-left:none;transform:rotate(45deg) translateY(-1px)}

.spinner-wrap{position:absolute;inset:0;background:rgba(255,255,255,.82);display:none;align-items:center;justify-content:center;flex-direction:column;gap:8px;font-size:13px;color:#555}
.card.processing .spinner-wrap{display:flex}

.card-body{padding:12px 14px 14px}
.card-filename{font-size:12px;color:#86868b;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card-status{font-size:12px;color:#86868b;min-height:16px;margin-bottom:8px}
.card-status.ok{color:#34c759}
.card-status.err{color:#ff3b30}
.card-status.ing{color:#ff9f0a}

.result-btns{display:flex;gap:6px}
.btn-sm{flex:1;padding:8px;border-radius:8px;border:none;font-size:12px;font-weight:600;cursor:pointer}
.btn-toggle{background:#f5f5f7;color:#1d1d1f;border:1px solid #d2d2d7}
.btn-dl{background:#0066cc;color:#fff}
.btn-retry{width:100%;margin-top:6px;padding:8px;border-radius:8px;border:1.5px solid #d2d2d7;background:#f5f5f7;color:#555;font-size:12px;font-weight:600;cursor:pointer;transition:all .12s}
.btn-retry:hover{background:#e8e8ed;border-color:#aaa}

/* 하단 액션 */
.actions{display:flex;gap:12px;justify-content:center;align-items:center;padding:4px 0 24px;flex-wrap:wrap}
.btn-nuki{background:#1d1d1f;color:#fff;border:none;border-radius:14px;padding:15px 44px;font-size:15px;font-weight:700;cursor:pointer;transition:all .15s}
.btn-nuki:hover:not(:disabled){background:#000;transform:translateY(-1px)}
.btn-nuki:disabled{background:#b0b0b6;cursor:not-allowed}
.btn-nuki.running{background:#ff9f0a}
.btn-prompt{background:#fff;color:#0066cc;border:2px solid #0066cc;border-radius:14px;padding:13px 28px;font-size:15px;font-weight:700;cursor:pointer;transition:all .15s;display:none}
.btn-prompt:hover{background:#f0f5ff}

/* 프롬프트 패널 */
#prompt-panel{background:#fff;border-radius:16px;box-shadow:0 2px 12px rgba(0,0,0,.1);padding:20px;margin-bottom:20px;display:none}
.pp-title{font-size:14px;font-weight:700;margin-bottom:10px;color:#1d1d1f}
.pp-hint{font-size:12px;color:#86868b;margin-bottom:10px}
#custom-prompt{width:100%;padding:12px 14px;border:1.5px solid #d2d2d7;border-radius:12px;font-size:14px;font-family:inherit;resize:vertical;min-height:72px;outline:none;transition:border-color .15s;margin-bottom:12px}
#custom-prompt:focus{border-color:#0066cc}
.pp-actions{display:flex;gap:8px;justify-content:flex-end}
.btn-cancel{padding:10px 20px;border-radius:10px;border:1px solid #d2d2d7;background:#f5f5f7;font-size:14px;font-weight:600;cursor:pointer;color:#555}
.btn-apply{padding:10px 24px;border-radius:10px;border:none;background:#0066cc;color:#fff;font-size:14px;font-weight:700;cursor:pointer}
.btn-apply:disabled{background:#b0b0b6;cursor:not-allowed}
.btn-apply.running{background:#ff9f0a}

.spinner{width:24px;height:24px;border:3px solid #e5e5ea;border-top-color:#0066cc;border-radius:50%;animation:spin .7s linear infinite}
.spinner-sm{display:inline-block;width:13px;height:13px;border:2px solid rgba(255,255,255,.35);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:5px}
@keyframes spin{to{transform:rotate(360deg)}}
.checker{background:repeating-conic-gradient(#d0d0d0 0% 25%,#f8f8f8 0% 50%) 0 0/14px 14px}
</style>
</head>
<body>
<div class="wrap">
  <h1>누끼 툴</h1>
  <p class="sub">배경 제거 후 원하는 이미지를 선택해 GPT로 추가 편집할 수 있습니다</p>

  <div class="upload-zone" id="upload-zone">
    <div class="icon">🖼</div>
    <p>이미지를 여기에 드래그하거나 클릭하여 선택</p>
    <p class="hint">여러 장 동시 업로드 · JPG, PNG, WEBP</p>
    <input type="file" id="file-input" multiple accept="image/*" style="display:none">
  </div>

  <div class="sel-bar" id="sel-bar">
    <label>
      <input type="checkbox" id="chk-all" onchange="selectAll(this.checked)">
      전체 선택
    </label>
    <span class="sel-info" id="sel-info"></span>
    <button class="btn-dl-all" id="btn-dl-all" onclick="dlSelected()">선택 다운로드 ZIP</button>
  </div>

  <div class="grid" id="grid"></div>

  <div id="prompt-panel">
    <div class="pp-title" id="pp-title">선택한 이미지에 적용할 프롬프트</div>
    <div class="pp-hint">GPT가 편집합니다 · 처리 시간 30~60초</div>
    <textarea id="custom-prompt" placeholder="예: 배경을 흰색으로 바꿔줘 / 스튜디오 조명 배경으로 교체 / 배경에 블러 효과"></textarea>
    <div class="pp-actions">
      <button class="btn-cancel" onclick="closePromptPanel()">취소</button>
      <button class="btn-apply" id="btn-apply">GPT 편집 적용</button>
    </div>
  </div>

  <div class="actions">
    <button class="btn-nuki" id="btn-nuki" disabled>배경 제거</button>
    <button class="btn-prompt" id="btn-prompt">선택한 <span id="chk-count">0</span>장 프롬프트 편집 →</button>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
<script>
const state = { cards: [] };

// 업로드
const uploadZone = document.getElementById('upload-zone');
const fileInput  = document.getElementById('file-input');
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('over');
  [...e.dataTransfer.files].filter(f => f.type.startsWith('image/')).forEach(addCard);
});
fileInput.addEventListener('change', () => {
  [...fileInput.files].filter(f => f.type.startsWith('image/')).forEach(addCard);
  fileInput.value = '';
});

function addCard(file) {
  const id = 'c' + Date.now() + Math.random().toString(36).slice(2,6);
  state.cards.push({ id, file, dataUrl: null, resultUrl: null, resultFilename: null, showing: 'original', checked: false, nukiMethod: null });

  const el = document.createElement('div');
  el.className = 'card'; el.id = id;
  el.innerHTML = `
    <div class="card-img">
      <label class="card-chk" title="선택">
        <input type="checkbox" id="chk_${id}" onchange="toggleCheck('${id}', this.checked)">
        <span class="chk-box"></span>
      </label>
      <img id="img_${id}" alt="">
      <div class="spinner-wrap">
        <div class="spinner"></div>
        <span id="spin_lbl_${id}">처리 중...</span>
      </div>
    </div>
    <div class="card-body">
      <div class="card-filename">${esc(file.name)}</div>
      <div class="card-status" id="st_${id}"></div>
      <div class="result-btns" id="btns_${id}" style="display:none">
        <button class="btn-sm btn-toggle" onclick="toggleView('${id}')">원본 보기</button>
        <button class="btn-sm btn-dl" onclick="dl('${id}')">다운로드</button>
      </div>
      <button class="btn-retry" id="retry_${id}" style="display:none" onclick="retryCard('${id}')"></button>
    </div>`;
  document.getElementById('grid').appendChild(el);

  const reader = new FileReader();
  reader.onload = e => {
    const c = state.cards.find(x => x.id === id);
    c.dataUrl = e.target.result;
    document.getElementById(`img_${id}`).src = e.target.result;
    updateBtns();
  };
  reader.readAsDataURL(file);
}

// 체크박스
function toggleCheck(id, checked) {
  const c = state.cards.find(x => x.id === id);
  if (!c) return;
  c.checked = checked;
  document.getElementById(id).classList.toggle('checked', checked);
  updateBtns();
}

function selectAll(checked) {
  state.cards.forEach(c => {
    c.checked = checked;
    document.getElementById(c.id).classList.toggle('checked', checked);
    const chk = document.getElementById(`chk_${c.id}`);
    if (chk) chk.checked = checked;
  });
  updateBtns();
}

// 동시 처리 수 제한 실행기
async function runConcurrent(items, fn, concurrency = 4) {
  let idx = 0;
  async function worker() {
    while (idx < items.length) {
      const item = items[idx++];
      await fn(item);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
}

// 배경 제거 (rembg)
document.getElementById('btn-nuki').addEventListener('click', async () => {
  const btn = document.getElementById('btn-nuki');
  const toProcess = state.cards.filter(c => c.dataUrl && !c.resultUrl);
  if (!toProcess.length) return;
  btn.disabled = true;
  btn.classList.add('running');

  const total = toProcess.length;
  let done = 0;
  await runConcurrent(toProcess, async c => {
    await runProcess(c, 'nuki', '');
    done++;
    btn.innerHTML = `<span class="spinner-sm"></span>${done}/${total} 처리 중...`;
  }, 4);

  btn.innerHTML = '배경 제거';
  btn.classList.remove('running');
  updateBtns();
});

// 프롬프트 패널
document.getElementById('btn-prompt').addEventListener('click', () => {
  const n = state.cards.filter(c => c.checked).length;
  document.getElementById('pp-title').textContent = `선택한 ${n}장에 적용할 프롬프트`;
  document.getElementById('prompt-panel').style.display = 'block';
  document.getElementById('custom-prompt').focus();
  document.getElementById('prompt-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});

function closePromptPanel() {
  document.getElementById('prompt-panel').style.display = 'none';
}

document.getElementById('btn-apply').addEventListener('click', async () => {
  const prompt = document.getElementById('custom-prompt').value.trim();
  if (!prompt) { document.getElementById('custom-prompt').focus(); return; }
  const targets = state.cards.filter(c => c.checked);
  if (!targets.length) return;

  const btn = document.getElementById('btn-apply');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-sm"></span>편집 중...';
  btn.classList.add('running');

  await Promise.all(targets.map(c => runProcess(c, 'prompt', prompt)));

  btn.innerHTML = 'GPT 편집 적용';
  btn.classList.remove('running');
  btn.disabled = false;
  closePromptPanel();
  updateBtns();
});

const NUKI_METHODS = ['rembg', 'lumakey', 'matte'];
const NUKI_LABELS  = { rembg: 'rembg (ML)', lumakey: '루마키', matte: '매트' };

function nextNukiMethod(current) {
  const idx = NUKI_METHODS.indexOf(current || 'rembg');
  return NUKI_METHODS[(idx + 1) % NUKI_METHODS.length];
}

function updateRetryBtn(c) {
  const btn = document.getElementById(`retry_${c.id}`);
  if (!btn) return;
  if (!c.nukiMethod) { btn.style.display = 'none'; return; }
  const next = nextNukiMethod(c.nukiMethod);
  btn.style.display = 'block';
  btn.textContent = `↩ 재시도 · ${NUKI_LABELS[next]}`;
}

async function retryCard(id) {
  const c = state.cards.find(x => x.id === id);
  if (!c) return;
  const next = nextNukiMethod(c.nukiMethod);
  c.resultUrl = null;
  await runProcess(c, 'nuki', '', next);
}

// 공통 처리
async function runProcess(c, mode, prompt, nukiMethod) {
  const el = document.getElementById(c.id);
  el.classList.add('processing');
  document.getElementById(`spin_lbl_${c.id}`).textContent =
    mode === 'nuki' ? '누끼 처리 중...' : 'GPT 편집 중...';
  setSt(c.id, mode === 'nuki' ? '처리 중...' : 'GPT 편집 중...', 'ing');

  const imageData = (mode === 'prompt' && c.resultUrl) ? c.resultUrl : c.dataUrl;
  const body = { image: imageData, mode, prompt };
  if (mode === 'nuki') body.nuki_method = nukiMethod || 'rembg';

  try {
    const resp = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    c.resultUrl = data.result;
    c.resultFilename = data.filename;
    if (mode === 'nuki') c.nukiMethod = data.method;
    el.classList.remove('processing');

    const img = document.getElementById(`img_${c.id}`);
    img.src = c.resultUrl;
    img.classList.add('checker');
    c.showing = 'result';

    setSt(c.id, `완료 · ${data.method}`, 'ok');
    document.getElementById(`btns_${c.id}`).style.display = 'flex';
    updateRetryBtn(c);
  } catch (err) {
    el.classList.remove('processing');
    setSt(c.id, '실패: ' + err.message, 'err');
  }
}

function toggleView(id) {
  const c = state.cards.find(x => x.id === id);
  if (!c?.resultUrl) return;
  const img = document.getElementById(`img_${id}`);
  const btn = document.querySelector(`#btns_${id} .btn-toggle`);
  if (c.showing === 'result') {
    img.src = c.dataUrl; img.classList.remove('checker');
    c.showing = 'original'; btn.textContent = '결과 보기';
  } else {
    img.src = c.resultUrl; img.classList.add('checker');
    c.showing = 'result'; btn.textContent = '원본 보기';
  }
}

function dl(id) {
  const c = state.cards.find(x => x.id === id);
  if (!c?.resultUrl) return;
  const a = document.createElement('a');
  a.href = c.resultUrl; a.download = c.resultFilename || 'nuki.png'; a.click();
}

async function dlSelected() {
  const targets = state.cards.filter(c => c.checked && c.resultUrl);
  if (!targets.length) { alert('완료된 이미지를 먼저 선택해주세요'); return; }
  const btn = document.getElementById('btn-dl-all');
  btn.textContent = '압축 중...'; btn.disabled = true;
  try {
    const zip = new JSZip();
    targets.forEach(c => {
      const b64 = c.resultUrl.split(',')[1];
      zip.file(c.resultFilename || `${c.id}.png`, b64, { base64: true });
    });
    const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `nuki_${targets.length}장_${Date.now()}.zip`;
    a.click();
    URL.revokeObjectURL(a.href);
  } finally {
    btn.textContent = '선택 다운로드 ZIP'; btn.disabled = false;
  }
}

function setSt(id, msg, cls) {
  const el = document.getElementById(`st_${id}`);
  if (el) { el.textContent = msg; el.className = 'card-status ' + (cls||''); }
}

function updateBtns() {
  const total           = state.cards.length;
  const hasUnprocessed  = state.cards.some(c => c.dataUrl && !c.resultUrl);
  const checkedCount    = state.cards.filter(c => c.checked).length;
  const checkedDone     = state.cards.filter(c => c.checked && c.resultUrl).length;

  document.getElementById('btn-nuki').disabled = !hasUnprocessed;

  const btnPrompt = document.getElementById('btn-prompt');
  btnPrompt.style.display = checkedCount > 0 ? 'block' : 'none';
  document.getElementById('chk-count').textContent = checkedCount;

  // 전체선택 바
  const selBar = document.getElementById('sel-bar');
  selBar.style.display = total > 0 ? 'flex' : 'none';
  const chkAll = document.getElementById('chk-all');
  chkAll.checked       = total > 0 && checkedCount === total;
  chkAll.indeterminate = checkedCount > 0 && checkedCount < total;
  document.getElementById('sel-info').textContent =
    checkedCount > 0 ? `${checkedCount}/${total}장 선택됨` : `총 ${total}장`;

  // 일괄 다운로드: 선택 중 완료된 것이 1장 이상일 때
  const btnDlAll = document.getElementById('btn-dl-all');
  btnDlAll.style.display = checkedDone > 0 ? 'block' : 'none';
  if (checkedDone > 0) btnDlAll.textContent = `선택 ${checkedDone}장 ZIP 다운로드`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7788))
    print("→ rembg 모델 사전 로드 중...")
    get_rembg_session()
    print(f"→ 준비 완료. http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
