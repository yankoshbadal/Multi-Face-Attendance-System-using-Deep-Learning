import os
import sys
import cv2  # type: ignore
import numpy as np


#python image_augmentation.py col_001 clsid001/ then some student name

if len(sys.argv) < 4:
    raise RuntimeError(
        "Usage: python image_augmentation.py <college_id> <class_id> <person_name>"
    )

COLLEGE_ID = sys.argv[1]
CLASS_ID = sys.argv[2]
PERSON_NAME = sys.argv[3]

BASE_DIR = r"C:\Users\yanko\OneDrive\Desktop\Desktop Files\present-me\colleges"

CLASS_DIR = os.path.join(BASE_DIR, COLLEGE_ID, CLASS_ID)

INPUT_DIR = os.path.join(CLASS_DIR, "registered_faces", PERSON_NAME)
OUTPUT_DIR = os.path.join(CLASS_DIR, "registered_faces_augmented", PERSON_NAME)

if not os.path.exists(INPUT_DIR):
    raise RuntimeError(f"Person folder not found: {INPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def horizontal_flip(img):
    return cv2.flip(img, 1)

def change_brightness(img, value=25):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def change_contrast(img, alpha=1.25):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def slight_rotation(img, angle=7):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


print(f"Augmenting images for: {PERSON_NAME}")
print(f"Input folder : {INPUT_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Augmentation complete for: {PERSON_NAME}")


for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    name, ext = os.path.splitext(file)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_orig{ext}"), img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_flip{ext}"), horizontal_flip(img))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_bright{ext}"), change_brightness(img))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_contrast{ext}"), change_contrast(img))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_rotate{ext}"), slight_rotation(img))

print(f"\nAugmentation complete for: {PERSON_NAME}")

