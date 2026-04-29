#!/usr/bin/env python3

import os
import logging
import base64
from typing import List, Optional

import gdown
import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

# -------------------------
# Config / Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-plates")

MODEL_PATH = "best.pt"

# -------------------------
# Descargar modelo (FIX REAL)
# -------------------------
def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        logger.info("✅ Modelo ya existe.")
        return

    logger.info("⬇️ Descargando modelo con gdown...")

    url = "https://drive.google.com/uc?id=13zoDhJcnL8LSzVL9eo5qNbyRysIP4tuW"
    gdown.download(url, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        raise RuntimeError("❌ El modelo no se descargó correctamente")

    logger.info("✅ Modelo descargado correctamente.")

download_model()

# -------------------------
# Parámetros
# -------------------------
OCR_LANGS = ["en"]
CONF_THRESH = 0.25

# -------------------------
# App init
# -------------------------
app = FastAPI(title="Detector de Placas YOLOv8")

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
logger.info("🔹 Cargando modelo YOLO...")
model = YOLO(MODEL_PATH)
logger.info("✅ Modelo cargado.")

reader = easyocr.Reader(OCR_LANGS, gpu=False)

# -------------------------
# Helpers
# -------------------------
def ocr_read_text_from_roi(roi):
    try:
        if roi is None or roi.size == 0:
            return None
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result = reader.readtext(roi_rgb)
        if not result:
            return None
        best = max(result, key=lambda x: x[2])
        text = "".join(ch for ch in best[1] if ch.isalnum())
        return text.upper() if text else None
    except:
        return None

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

# -------------------------
# Rutas
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>API YOLO funcionando 🚀</h1>"

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

                # dibujar caja
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        img_b64 = image_to_base64(frame)

        return {
            "placas": placas,
            "image": img_b64,
            "success": True
        }

    except Exception as e:
        logger.exception("Error en predict")
        return {"error": str(e)}

# -------------------------
# Main (IMPORTANTE)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)