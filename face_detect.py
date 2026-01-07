from ultralytics import YOLO
import cv2
import os
import sys


CLASS_DIR = sys.argv[1] # college/class

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "yolov8n-face.pt"
)

INPUT_DIR = os.path.join(CLASS_DIR, "input")
OUTPUT_DIR = os.path.join(CLASS_DIR, "detected")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

face_count = 0
MARGIN = 0.10

for img_file in os.listdir(INPUT_DIR):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img = cv2.imread(os.path.join(INPUT_DIR, img_file))
    if img is None:
        continue

    h, w, _ = img.shape
    results = model(img)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            bw = x2 - x1
            bh = y2 - y1
            mx = int(MARGIN * bw)
            my = int(MARGIN * bh)

            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w, x2 + mx)
            y2 = min(h, y2 + my)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            face_count += 1
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"face_{face_count}.png"),
                crop
            )

print("OK Faces cropped:", face_count)
