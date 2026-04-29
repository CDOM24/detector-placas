import os
import gc
import shutil
import gdown
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO

app = FastAPI()

MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?id=13zoDhJcnL8LSzVL9eo5qNbyRysIP4tuW"

model = None
reader = None

# 📥 Descargar modelo
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Descargando modelo...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Modelo descargado")

# 🧠 YOLO (carga diferida)
def get_model():
    global model
    if model is None:
        print("🔹 Cargando YOLO...")
        model = YOLO(MODEL_PATH)
    return model

# 🔤 easyocr (carga diferida)
def get_reader():
    global reader
    if reader is None:
        print("🔹 Cargando EasyOCR...")
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

@app.get("/")
def home():
    return {"status": "API funcionando 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = "temp.jpg"

    # Guardar imagen
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ======================
    # 1️⃣ DETECCIÓN (YOLO)
    # ======================
    model = get_model()
    results = model(temp_path)

    detections = []
    crops = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            detections.append({
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

            # guardar recorte
            import cv2
            img = cv2.imread(temp_path)
            crop = img[y1:y2, x1:x2]
            crop_path = f"crop_{len(crops)}.jpg"
            cv2.imwrite(crop_path, crop)
            crops.append(crop_path)

    # 🔥 liberar YOLO
    del model
    gc.collect()

    # ======================
    # 2️⃣ OCR (easyocr)
    # ======================
    reader = get_reader()

    texts = []
    for crop_path in crops:
        result = reader.readtext(crop_path)
        for (_, text, conf) in result:
            texts.append({
                "text": text,
                "confidence": conf
            })
        os.remove(crop_path)

    return {
        "detections": detections,
        "texts": texts
    }

# Ejecutar descarga al inicio
download_model()
