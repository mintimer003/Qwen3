import base64, json, re, cv2, numpy as np, os
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from llama_cpp import Llama
import llama_cpp.llama_chat_format as chat_format
import uvicorn

app = FastAPI(title="BPMN Final Backend")

# --- УМНАЯ НАСТРОЙКА ПУТЕЙ (ИЗМЕНЕНО) ---
# Имена твоих файлов в папке models
YOLO_NAME = "best.onnx"
VLM_NAME = "Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf"
CLIP_NAME = "mmproj-F16.gguf"

if os.getenv("DOCKER_ENV"):
    # Пути внутри Docker контейнера (мы их туда пробросим в docker-compose)
    print("🐳 Running in Docker mode")
    BASE_MODEL_DIR = "/models"
else:
    # Пути локально: ищем папку models рядом с файлом скрипта
    print("💻 Running in Local mode")
    BASE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

YOLO_PATH = os.path.join(BASE_MODEL_DIR, YOLO_NAME)
VLM_PATH = os.path.join(BASE_MODEL_DIR, VLM_NAME)
CLIP_PATH = os.path.join(BASE_MODEL_DIR, CLIP_NAME)

print(f"📂 Loading YOLO from: {YOLO_PATH}")
print(f"📂 Loading VLM from: {VLM_PATH}")

# --- ЗАГРУЗКА МОДЕЛЕЙ ---
try:
    yolo = YOLO(YOLO_PATH, task='detect')
except Exception as e:
    print(f"❌ Ошибка загрузки YOLO. Проверь, лежит ли файл {YOLO_NAME} в папке models!")
    raise e

try:
    # Пытаемся загрузить с GPU (n_gpu_layers=20), если не выйдет - уменьшим
    chat_handler = chat_format.NanoLlavaChatHandler(clip_model_path=CLIP_PATH)
    vlm = Llama(
        model_path=VLM_PATH,
        chat_handler=chat_handler,
        n_gpu_layers=20, # Ставим побольше для скорости
        n_ctx=2048,
        verbose=False
    )
except Exception as e:
    print(f"⚠️ Ошибка GPU или загрузки VLM. Пробую на CPU/меньше слоев... Ошибка: {e}")
    # Фолбэк для слабых ПК
    chat_handler = chat_format.NanoLlavaChatHandler(clip_model_path=CLIP_PATH)
    vlm = Llama(
        model_path=VLM_PATH,
        chat_handler=chat_handler,
        n_gpu_layers=0, 
        n_ctx=2048,
        verbose=True
    )

CLASS_COLORS = {
    'process': (0, 255, 255),    'decision': (255, 0, 255),
    'terminator': (0, 0, 255),   'arrowhead': (0, 165, 255),
    'text': (180, 180, 180),     'default': (255, 255, 255)
}

def is_near_arrow(point, arrows, threshold=15):
    for a in arrows:
        ax1, ay1, ax2, ay2 = a
        if (ax1-threshold <= point[0] <= ax2+threshold) and (ay1-threshold <= point[1] <= ay2+threshold):
            return True
    return False

def clean_json_output(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()
    match = re.search(r'(\{|\[).*(\}|\])', text, re.DOTALL)
    if match: return match.group(0)
    return text

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    viz_img = img.copy()
    h, w = img.shape[:2]

    # 1. YOLO
    results = yolo(img, conf=0.25)[0]
    yolo_boxes = []
    arrow_boxes = []

    # 2. ПОДГОТОВКА МАСКИ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_for_lsd = gray.copy()

    # 3. ТОЧЕЧНАЯ ЗАЛИВКА (FIXED WHITE)
    for box in results.boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        label = results.names[int(box.cls[0])]
        yolo_boxes.append({'coords': coords, 'label': label})
        
        if label == 'arrowhead':
            arrow_boxes.append(coords)

        # Паддинг 4px и заливка белым (255)
        pad = 4 
        x1 = max(0, coords[0] - pad)
        y1 = max(0, coords[1] - pad)
        x2 = min(w, coords[2] + pad)
        y2 = min(h, coords[3] + pad)
        cv2.rectangle(mask_for_lsd, (x1, y1), (x2, y2), (255), -1)

    mask_for_lsd = cv2.GaussianBlur(mask_for_lsd, (3, 3), 0)

    # 4. LSD
    scale = 3.0 
    img_resized = cv2.resize(mask_for_lsd, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lsd = cv2.createLineSegmentDetector(0)
    lines_scaled, _, _, _ = lsd.detect(img_resized)

    # 5. ФИЛЬТРАЦИЯ
    if lines_scaled is not None:
        for line in lines_scaled:
            p1 = (int(line[0][0] / scale), int(line[0][1] / scale))
            p2 = (int(line[0][2] / scale), int(line[0][3] / scale))
            length = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            angle = np.abs(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]) * 180 / np.pi)
            
            is_straight = (angle < 4 or angle > 176) or (86 < angle < 94)
            is_long = length > (max(h, w) * 0.05)
            near_arrow = is_near_arrow(p1, arrow_boxes) or is_near_arrow(p2, arrow_boxes)
            
            if (is_long and is_straight) or (length > 10 and near_arrow):
                cv2.line(viz_img, p1, p2, (0, 255, 0), 2)

    # 6. YOLO ПОВЕРХ
    for item in yolo_boxes:
        box, label = item['coords'], item['label']
        color = CLASS_COLORS.get(label, CLASS_COLORS['default'])
        cv2.rectangle(viz_img, (box[0], box[1]), (box[2], box[3]), color, 2)
        if label != 'arrowhead':
            cv2.putText(viz_img, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 7. VLM
    final_h, final_w = viz_img.shape[:2]
    scale_factor = 960 / max(final_h, final_w)
    processed_img_small = cv2.resize(viz_img, (int(final_w * scale_factor), int(final_h * scale_factor))) if scale_factor < 1 else viz_img
    
    _, buffer = cv2.imencode('.jpg', processed_img_small)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    prompt = (
        "Analyze this BPMN diagram. "
        "Focus on GREEN LINES and colored boxes. "
        "Return ONLY a JSON array of connections: "
        "[{\"from\": \"Step A\", \"to\": \"Step B\", \"condition\": \"...\"}]"
    )

    response = vlm.create_chat_completion(
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
        ]}],
        temperature=0.01,
        max_tokens=1024
    )

    full_res = response["choices"][0]["message"]["content"]
    try:
        clean_json = clean_json_output(full_res)
        parsed = json.loads(clean_json)
        final_logic = parsed if isinstance(parsed, list) else parsed.get("logic", [])
        return {"image": img_b64, "logic": final_logic, "raw": full_res}
    except:
        return {"image": img_b64, "logic": [], "raw": full_res}

if __name__ == "__main__":
    # Локальный запуск
    uvicorn.run(app, host="0.0.0.0", port=8000)