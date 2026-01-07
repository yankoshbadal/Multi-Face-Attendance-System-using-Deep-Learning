from ultralytics import YOLO
import cv2
import os
import sys
#augment se pehle crop karna hai
#python pre_face_emb.py col_001 clsid001 yankosh

if len(sys.argv) < 4:
    raise RuntimeError(
        "Usage: python pre_face_emb.py <college_id> <class_id> <person_name>"
    )

COLLEGE_ID = sys.argv[1]
CLASS_ID = sys.argv[2]
PERSON_NAME = sys.argv[3]

BASE_DIR = r"C:\Users\yanko\OneDrive\Desktop\Desktop Files\present-me\colleges"

CLASS_DIR = os.path.join(BASE_DIR, COLLEGE_ID, CLASS_ID)

INPUT_DIR = os.path.join(CLASS_DIR, "raw_faces_imgs", PERSON_NAME)
OUTPUT_DIR = os.path.join(CLASS_DIR, "registered_faces", PERSON_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(INPUT_DIR):
    raise RuntimeError(f"Input folder not found: {INPUT_DIR}")

#--- 
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "yolov8n-face.pt"
)

model = YOLO(MODEL_PATH)

#crop before aug
face_count = 0
MARGIN = 0.25

print(f"Processing faces for: {PERSON_NAME}")
print(f"Input folder : {INPUT_DIR}")
print(f"Output folder: {OUTPUT_DIR}")

for img_file in os.listdir(INPUT_DIR):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, img_file)
    img = cv2.imread(img_path)

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

print(f"\nPre Embd Faces cropped and saved: {face_count}")
