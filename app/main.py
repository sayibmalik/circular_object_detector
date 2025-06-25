# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2

from app.utils import segment_circles
import shutil, os, json
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import uuid

IMAGE_DIR = "app/data/annotated"
STATIC_DIR = "app/data"
SITE_DIR = "app/site"
RESULTS_DIR = "app/storage"

DATA_DIR = Path("app/data")
OUTPUT_DIR = Path("app/data/annotated")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve static annotated images
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/site", StaticFiles(directory=SITE_DIR), name="index")
# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
model = YOLO("yolov8n.pt")


@app.post("/detect_by_yolo")
async def upload_image(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    filename = f"{file_id}{ext}"
    image_path = DATA_DIR / filename

    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run YOLO inference
    results = model(str(image_path))[0]

    # Draw and save annotated image
    img = cv2.imread(str(image_path))
    annotations = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = str(model.names[int(box.cls[0])])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        annotations.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class": label
        })

    annotated_path = OUTPUT_DIR / f"annotated_{filename}"
    cv2.imwrite(str(annotated_path), img)

    return {
        "image_id": file_id,
        "filename": filename,
        "annotated_image_url": f"/annotated/{annotated_path.name}",
        "detections": annotations
    }


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_id = file.filename.split('.')[0]
    filepath = os.path.join(IMAGE_DIR, file.filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    objects, annotated_path = segment_circles(filepath, output_dir=IMAGE_DIR)
    
    # Save metadata
    with open(f"{RESULTS_DIR}/{file_id}.json", "w") as f:
        json.dump(objects, f)

    return {
        "message": "Image uploaded and processed",
        "image_id": file_id,
        "object_count": len(objects),
        "annotated_image": f"/annotated/{os.path.basename(annotated_path)}"
    }

@app.get("/objects/{image_id}")
def get_objects(image_id: str):
    path = f"{RESULTS_DIR}/{image_id}.json"
    if not os.path.exists(path):
        return {"error": "Image ID not found"}
    return json.load(open(path))

@app.get("/objects/{image_id}/{object_id}")
def get_object_by_id(image_id: str, object_id: str):
    objects = json.load(open(f"{RESULTS_DIR}/{image_id}.json"))
    for obj in objects:
        if obj["id"] == object_id:
            return obj
    return {"error": "Object ID not found"}


# Visual debug helper
# def show_debug_overlay(image_path: str, objects: List[Dict]):
#     image = cv2.imread(image_path)
#     for obj in objects:
#         x, y = obj["centroid"]
#         r = obj["radius"]
#         cv2.circle(image, (x, y), r, (0, 255, 0), 2)
#         cv2.putText(image, obj["id"][:4], (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
#     cv2.imshow("Result", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
DATA_DIR = Path("app/data/coin-dataset")

@app.get("/images/")
def list_uploaded_images():
    if not DATA_DIR.exists():
        return JSONResponse(status_code=404, content={"error": "Data directory not found"})

    image_files = [f.name for f in DATA_DIR.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    for f in DATA_DIR.iterdir():
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            print(f)
            # objects, annotated_path = segment_circles(f, output_dir=IMAGE_DIR)

    return {"images": image_files, "count": len(image_files)}
