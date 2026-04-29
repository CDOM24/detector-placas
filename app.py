#!/usr/bin/env python3

import os
import logging
import base64
import requests
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

# -------------------------
# Descargar modelo (FIX REAL)
# -------------------------
def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000000:
        logger.info("✅ Modelo ya existe y parece válido.")
        return

    logger.info("⬇️ Descargando modelo desde Google Drive...")

    file_id = "13zoDhJcnL8LSzVL9eo5qNbyRysIP4tuW"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    session = requests.Session()
    response = session.get(url, stream=True, timeout=60)

    # 🔥 Manejo de archivos grandes de Drive
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True, timeout=60)
            break

    # 🔥 Validación anti-HTML (tu error original)
    content_start = response.raw.read(10)
    if b"<" in content_start:
        raise RuntimeError("❌ ERROR: Se descargó HTML en lugar del modelo .pt")

    response = session.get(url, stream=True)

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    logger.info("✅ Modelo descargado correctamente.")

download_model()

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
logger.info("🔹 Cargando modelo YOLOv8...")
model = YOLO(MODEL_PATH)
logger.info("✅ Modelo cargado.")

reader = easyocr.Reader(OCR_LANGS, gpu=False)

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
        text = "".join(ch for ch in best[1] if ch.isalnum())
        return text.upper() if text else None
    except Exception:
        return None


def image_to_base64_jpg(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

# -------------------------
# Rutas
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>API funcionando 🚀</h1>"

@app.post("/predict/")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model.predict(source=frame, conf=conf, verbose=False)

        placas = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]

                text = ocr_read_text_from_roi(roi)
                if text:
                    placas.append(text)

        return {"placas": placas}

    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Main (IMPORTANTE PARA RENDER)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
