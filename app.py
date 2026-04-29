#!/usr/bin/env python3
# app.py -- FastAPI + YOLOv8 + EasyOCR para detección de placas
# Requiere: fastapi uvicorn ultralytics easyocr opencv-python-headless pillow numpy python-multipart

import os
import logging
import base64
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

# -------------------------
# Config / Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-plates")

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))
RETURN_IMAGE = True

# -------------------------
# App init
# -------------------------
app = FastAPI(title="YOLOv8 - Detector de Placas (OCR)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Cargar modelo y OCR
# -------------------------
logger.info("🔹 Cargando modelo YOLOv8 desde %s ...", MODEL_PATH)
model = YOLO(MODEL_PATH)
logger.info("✅ Modelo YOLOv8 cargado correctamente.")

logger.info("🔹 Inicializando EasyOCR con idiomas: %s", OCR_LANGS)
reader = easyocr.Reader(OCR_LANGS, gpu=False)
logger.info("✅ EasyOCR listo.")

# -------------------------
# Helpers
# -------------------------
def ocr_read_text_from_roi(roi_bgr: np.ndarray) -> Optional[str]:
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        result = reader.readtext(roi_rgb)
        if not result:
            return None
        best = max(result, key=lambda x: x[2])
        text = best[1]
        text = "".join(ch for ch in text if ch.isalnum())
        return text.upper() if text else None
    except Exception as e:
        logger.exception("OCR error: %s", e)
        return None


def image_to_base64_jpg(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode('utf-8')


# -------------------------
# Frontend Visual
# -------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Detector de Placas · YOLOv8</title>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0a0a0f;
      --surface: #111118;
      --border: #1e1e2e;
      --accent: #00e5b4;
      --accent2: #7b61ff;
      --warn: #ff6b35;
      --text: #e8e8f0;
      --muted: #5a5a7a;
      --card: #13131f;
    }

    body {
      font-family: 'Syne', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Animated grid background */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(var(--border) 1px, transparent 1px),
        linear-gradient(90deg, var(--border) 1px, transparent 1px);
      background-size: 48px 48px;
      opacity: 0.4;
      z-index: 0;
      pointer-events: none;
    }

    body::after {
      content: '';
      position: fixed;
      top: -40%;
      left: -20%;
      width: 80vw;
      height: 80vw;
      background: radial-gradient(circle, rgba(0,229,180,0.06) 0%, transparent 65%);
      z-index: 0;
      pointer-events: none;
      animation: pulse 8s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.1); opacity: 0.7; }
    }

    .wrapper {
      position: relative;
      z-index: 1;
      max-width: 960px;
      margin: 0 auto;
      padding: 48px 24px 80px;
    }

    /* Header */
    header {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 56px;
      animation: fadeDown 0.6s ease both;
    }

    .logo-mark {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      display: grid;
      place-items: center;
      font-size: 22px;
      flex-shrink: 0;
    }

    .header-text h1 {
      font-size: clamp(1.4rem, 4vw, 1.9rem);
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.1;
    }

    .header-text h1 span {
      color: var(--accent);
    }

    .header-text p {
      font-family: 'DM Mono', monospace;
      font-size: 0.75rem;
      color: var(--muted);
      margin-top: 4px;
      letter-spacing: 0.05em;
    }

    .badge {
      margin-left: auto;
      font-family: 'DM Mono', monospace;
      font-size: 0.7rem;
      padding: 5px 12px;
      border-radius: 999px;
      border: 1px solid var(--accent);
      color: var(--accent);
      letter-spacing: 0.1em;
      white-space: nowrap;
      flex-shrink: 0;
    }

    /* Upload zone */
    .upload-zone {
      border: 2px dashed var(--border);
      border-radius: 20px;
      padding: 56px 32px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.25s, background 0.25s;
      background: var(--card);
      animation: fadeUp 0.6s 0.1s ease both;
      position: relative;
      overflow: hidden;
    }

    .upload-zone::before {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(135deg, rgba(0,229,180,0.03), rgba(123,97,255,0.03));
      opacity: 0;
      transition: opacity 0.3s;
    }

    .upload-zone:hover, .upload-zone.drag-over {
      border-color: var(--accent);
    }

    .upload-zone:hover::before, .upload-zone.drag-over::before {
      opacity: 1;
    }

    .upload-icon {
      font-size: 3rem;
      margin-bottom: 16px;
      display: block;
      filter: grayscale(0.3);
    }

    .upload-zone h2 {
      font-size: 1.1rem;
      font-weight: 700;
      margin-bottom: 8px;
    }

    .upload-zone p {
      font-family: 'DM Mono', monospace;
      font-size: 0.78rem;
      color: var(--muted);
    }

    #fileInput { display: none; }

    /* Confidence control */
    .controls {
      display: flex;
      align-items: center;
      gap: 16px;
      margin: 20px 0;
      animation: fadeUp 0.6s 0.15s ease both;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 20px;
    }

    .controls label {
      font-family: 'DM Mono', monospace;
      font-size: 0.78rem;
      color: var(--muted);
      white-space: nowrap;
    }

    .controls input[type=range] {
      flex: 1;
      accent-color: var(--accent);
      height: 4px;
      cursor: pointer;
    }

    #confVal {
      font-family: 'DM Mono', monospace;
      font-size: 0.85rem;
      color: var(--accent);
      font-weight: 500;
      min-width: 40px;
      text-align: right;
    }

    /* Analyze button */
    .btn-analyze {
      width: 100%;
      padding: 18px;
      border-radius: 14px;
      border: none;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      color: #0a0a0f;
      font-family: 'Syne', sans-serif;
      font-size: 1rem;
      font-weight: 800;
      letter-spacing: 0.03em;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.15s;
      margin-top: 4px;
      animation: fadeUp 0.6s 0.2s ease both;
    }

    .btn-analyze:hover { opacity: 0.9; transform: translateY(-1px); }
    .btn-analyze:active { transform: translateY(0); }
    .btn-analyze:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    /* Preview strip */
    #previewWrap {
      display: none;
      margin-top: 20px;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid var(--border);
      animation: fadeUp 0.4s ease both;
    }

    #previewWrap img {
      width: 100%;
      display: block;
      max-height: 340px;
      object-fit: contain;
      background: #000;
    }

    .preview-label {
      font-family: 'DM Mono', monospace;
      font-size: 0.7rem;
      color: var(--muted);
      padding: 8px 14px;
      background: var(--card);
      letter-spacing: 0.08em;
    }

    /* Loader */
    .loader-wrap {
      display: none;
      flex-direction: column;
      align-items: center;
      gap: 16px;
      padding: 40px;
      animation: fadeUp 0.4s ease both;
    }

    .spinner {
      width: 48px;
      height: 48px;
      border: 3px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    .loader-wrap p {
      font-family: 'DM Mono', monospace;
      font-size: 0.8rem;
      color: var(--muted);
      letter-spacing: 0.05em;
    }

    /* Results section */
    #results {
      display: none;
      margin-top: 32px;
      animation: fadeUp 0.5s ease both;
    }

    .results-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 20px;
    }

    .results-header h3 {
      font-size: 0.75rem;
      font-family: 'DM Mono', monospace;
      letter-spacing: 0.12em;
      color: var(--muted);
      text-transform: uppercase;
    }

    .results-count {
      font-family: 'DM Mono', monospace;
      font-size: 0.72rem;
      color: var(--accent);
      background: rgba(0,229,180,0.1);
      padding: 4px 10px;
      border-radius: 999px;
    }

    /* Plate cards */
    .plates-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }

    .plate-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px 20px;
      text-align: center;
      position: relative;
      overflow: hidden;
      transition: border-color 0.2s, transform 0.2s;
    }

    .plate-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
    }

    .plate-card:hover {
      border-color: var(--accent);
      transform: translateY(-2px);
    }

    .plate-card .plate-label {
      font-family: 'DM Mono', monospace;
      font-size: 0.65rem;
      color: var(--muted);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }

    .plate-card .plate-number {
      font-size: 1.9rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      color: var(--accent);
      line-height: 1;
    }

    /* Processed image */
    .result-image-wrap {
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--border);
    }

    .result-image-wrap img {
      width: 100%;
      display: block;
      object-fit: contain;
      background: #000;
      max-height: 420px;
    }

    .result-image-label {
      background: var(--card);
      padding: 10px 16px;
      font-family: 'DM Mono', monospace;
      font-size: 0.7rem;
      color: var(--muted);
      letter-spacing: 0.08em;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--accent);
      display: inline-block;
      animation: blink 1.5s ease infinite;
    }

    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

    /* No detection state */
    .no-detect {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 32px;
      text-align: center;
      margin-bottom: 24px;
    }

    .no-detect p {
      font-family: 'DM Mono', monospace;
      font-size: 0.82rem;
      color: var(--warn);
    }

    /* Error state */
    .error-box {
      background: rgba(255,107,53,0.08);
      border: 1px solid rgba(255,107,53,0.3);
      border-radius: 14px;
      padding: 20px 24px;
      font-family: 'DM Mono', monospace;
      font-size: 0.8rem;
      color: var(--warn);
      margin-top: 20px;
      display: none;
      animation: fadeUp 0.4s ease both;
    }

    /* Animations */
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(16px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeDown {
      from { opacity: 0; transform: translateY(-12px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    /* Footer */
    footer {
      text-align: center;
      margin-top: 64px;
      font-family: 'DM Mono', monospace;
      font-size: 0.7rem;
      color: var(--muted);
      letter-spacing: 0.06em;
    }

    footer span { color: var(--accent2); }
  </style>
</head>
<body>
<div class="wrapper">

  <header>
    <div class="logo-mark">🚗</div>
    <div class="header-text">
      <h1>Detector de <span>Placas</span></h1>
      <p>YOLOv8 · EasyOCR · FastAPI</p>
    </div>
    <div class="badge">v1.0 · ACTIVO</div>
  </header>

  <!-- Upload -->
  <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <span class="upload-icon">📷</span>
    <h2>Arrastra una imagen o haz clic aquí</h2>
    <p>JPG, PNG · Máx. 10MB</p>
    <input type="file" id="fileInput" accept="image/*"/>
  </div>

  <!-- Preview -->
  <div id="previewWrap">
    <div class="preview-label">PREVISUALIZACIÓN</div>
    <img id="previewImg" src="" alt="preview"/>
  </div>

  <!-- Controls -->
  <div class="controls">
    <label>CONFIANZA MÍNIMA</label>
    <input type="range" id="confRange" min="5" max="95" value="25" step="5"
           oninput="document.getElementById('confVal').textContent = this.value + '%'"/>
    <span id="confVal">25%</span>
  </div>

  <!-- Analyze button -->
  <button class="btn-analyze" id="btnAnalyze" onclick="analyze()" disabled>
    ⚡ ANALIZAR IMAGEN
  </button>

  <!-- Loader -->
  <div class="loader-wrap" id="loader">
    <div class="spinner"></div>
    <p>Procesando con YOLOv8 + OCR...</p>
  </div>

  <!-- Error -->
  <div class="error-box" id="errorBox"></div>

  <!-- Results -->
  <div id="results">
    <div class="results-header">
      <h3>RESULTADO DEL ANÁLISIS</h3>
      <span class="results-count" id="countBadge">0 placas</span>
    </div>

    <div class="plates-grid" id="platesGrid"></div>

    <div class="result-image-wrap" id="resultImgWrap" style="display:none">
      <div class="result-image-label">
        <span class="dot"></span> IMAGEN PROCESADA
      </div>
      <img id="resultImg" src="" alt="resultado"/>
    </div>
  </div>

  <footer>
    Desarrollado por <span>Alfredo Díaz</span> · YOLOv8 + FastAPI · AWS EC2
  </footer>

</div>

<script>
  const dropZone  = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const previewWrap = document.getElementById('previewWrap');
  const previewImg  = document.getElementById('previewImg');
  const loader    = document.getElementById('loader');
  const results   = document.getElementById('results');
  const errorBox  = document.getElementById('errorBox');
  const platesGrid = document.getElementById('platesGrid');
  const resultImg  = document.getElementById('resultImg');
  const resultImgWrap = document.getElementById('resultImgWrap');
  const countBadge = document.getElementById('countBadge');

  let selectedFile = null;

  // Drag & drop
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) loadFile(f);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) loadFile(fileInput.files[0]);
  });

  function loadFile(f) {
    selectedFile = f;
    const url = URL.createObjectURL(f);
    previewImg.src = url;
    previewWrap.style.display = 'block';
    btnAnalyze.disabled = false;
    results.style.display = 'none';
    errorBox.style.display = 'none';
  }

  async function analyze() {
    if (!selectedFile) return;

    const conf = parseInt(document.getElementById('confRange').value) / 100;

    // UI: loading
    btnAnalyze.disabled = true;
    loader.style.display = 'flex';
    results.style.display = 'none';
    errorBox.style.display = 'none';

    try {
      const fd = new FormData();
      fd.append('file', selectedFile);

      const resp = await fetch(`/predict/?conf=${conf}`, { method: 'POST', body: fd });
      const data = await resp.json();

      loader.style.display = 'none';

      if (data.error) {
        errorBox.textContent = '❌ ' + data.error;
        errorBox.style.display = 'block';
        btnAnalyze.disabled = false;
        return;
      }

      // Render plates
      platesGrid.innerHTML = '';
      const placas = data.placas || [];
      countBadge.textContent = placas.length + (placas.length === 1 ? ' placa' : ' placas');

      if (placas.length === 0) {
        platesGrid.innerHTML = '<div class="no-detect"><p>⚠️ No se detectaron placas en la imagen. Intenta con una foto más clara o baja la confianza mínima.</p></div>';
      } else {
        placas.forEach(p => {
          const card = document.createElement('div');
          card.className = 'plate-card';
          card.innerHTML = `<div class="plate-label">PLACA DETECTADA</div><div class="plate-number">${p}</div>`;
          platesGrid.appendChild(card);
        });
      }

      // Processed image
      if (data.image) {
        resultImg.src = 'data:image/jpeg;base64,' + data.image;
        resultImgWrap.style.display = 'block';
      } else {
        resultImgWrap.style.display = 'none';
      }

      results.style.display = 'block';

    } catch (err) {
      loader.style.display = 'none';
      errorBox.textContent = '❌ Error de conexión: ' + err.message;
      errorBox.style.display = 'block';
    }

    btnAnalyze.disabled = false;
  }
</script>
</body>
</html>
"""

# -------------------------
# Rutas
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.post("/predict/")
async def predict(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    conf: float = 0.25
):
    try:
        logger.info("📩 Petición recibida en /predict/")

        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
        elif image_base64:
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]
            image_base64 = image_base64.strip()
            try:
                img_data = base64.b64decode(image_base64 + "===")
            except Exception as e:
                logger.error("❌ Base64 inválido: %s", e)
                return {"error": "Base64 inválido o corrupto."}
            nparr = np.frombuffer(img_data, np.uint8)
        else:
            return {"error": "No se recibió ninguna imagen"}

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "No se pudo decodificar la imagen"}

        threshold = conf if conf != 0.25 else CONF_THRESH

        logger.info("🧠 Procesando imagen con YOLOv8 (conf=%.2f)...", threshold)
        results = model.predict(source=frame, conf=threshold, verbose=False)

        if not results:
            return {"placas": [], "image": None, "success": True, "message": "Sin detecciones"}

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        clss  = r.boxes.cls.cpu().numpy()  if len(r.boxes) > 0 else np.array([])

        placas_detectadas: List[str] = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(clss[i]) if len(clss) > i else None
            label  = model.names[cls_id] if cls_id is not None and cls_id < len(model.names) else "objeto"
            c      = confs[i] if len(confs) > i else 0

            h, w = frame.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            roi = frame[y1c:y2c, x1c:x2c].copy()

            if any(k in label.lower() for k in ["placa", "plate", "license"]):
                text_detected = ocr_read_text_from_roi(roi)
                if text_detected:
                    placas_detectadas.append(text_detected)
                    cv2.putText(frame, text_detected, (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {c:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        img_b64 = image_to_base64_jpg(frame) if RETURN_IMAGE else None

        logger.info("✅ Placas detectadas: %s", placas_detectadas)

        return {
            "success": True,
            "placas": placas_detectadas,
            "num_placas": len(placas_detectadas),
            "image": img_b64,
            "message": "OK" if placas_detectadas else "No se detectaron placas"
        }

    except Exception as e:
        logger.exception("Error en /predict/: %s", e)
        return {"error": str(e)}


@app.post("/predict_json/")
async def predict_json(request: Request):
    try:
        body = await request.json()
        image_base64 = body.get("image_base64")
        if not image_base64:
            return {"error": "No se recibió ninguna imagen"}

        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        image_base64 = image_base64.strip()
        img_data = base64.b64decode(image_base64 + "===")
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "No se pudo decodificar la imagen"}

        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)
        if not results:
            return {"placas": [], "image": None, "success": True, "message": "Sin detecciones"}

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        clss  = r.boxes.cls.cpu().numpy()  if len(r.boxes) > 0 else np.array([])
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])

        placas_detectadas = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(clss[i]) if len(clss) > i else None
            label  = model.names[cls_id] if cls_id is not None and cls_id < len(model.names) else "objeto"
            c      = confs[i] if len(confs) > i else 0
            h, w   = frame.shape[:2]
            roi    = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()

            if any(k in label.lower() for k in ["placa", "plate", "license"]):
                text_detected = ocr_read_text_from_roi(roi)
                if text_detected:
                    placas_detectadas.append(text_detected)
                    cv2.putText(frame, text_detected, (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {c:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        img_b64 = image_to_base64_jpg(frame) if RETURN_IMAGE else None

        return {
            "success": True,
            "placas": placas_detectadas,
            "num_placas": len(placas_detectadas),
            "image": img_b64,
            "message": "OK" if placas_detectadas else "No se detectaron placas"
        }

    except Exception as e:
        logger.exception("Error en /predict_json/: %s", e)
        return {"error": str(e)}


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info("🚀 Iniciando servidor en 0.0.0.0:%s", port)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
