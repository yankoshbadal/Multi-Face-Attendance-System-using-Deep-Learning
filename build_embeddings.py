import os
import sys

#CPU -
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#warnings
import tensorflow as tf  # type: ignore
tf.get_logger().setLevel("ERROR")

import cv2  # type: ignore
import numpy as np
from keras_facenet import FaceNet  # type: ignore

#python build_embeddings.py col_001 clsid001 person_name (Neeche hai)

if len(sys.argv) < 4:
    raise RuntimeError(
        "Usage: python build_embeddings.py <college_id> <class_id> <person_name>"
    )

COLLEGE_ID = sys.argv[1]
CLASS_ID = sys.argv[2]
PERSON_NAME = sys.argv[3]

#---
BASE_DIR = r"C:\Users\yanko\OneDrive\Desktop\Desktop Files\present-me\colleges"

CLASS_DIR = os.path.join(BASE_DIR, COLLEGE_ID, CLASS_ID)

REGISTERED_DIR = os.path.join(
    CLASS_DIR,
    "registered_faces_augmented",
    PERSON_NAME
)

EMB_DIR = os.path.join(CLASS_DIR, "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

if not os.path.exists(REGISTERED_DIR):
    raise RuntimeError(f"Person folder not found: {REGISTERED_DIR}")

# ---
MIN_IMAGES = 4
IMAGE_SIZE = (160, 160)

#facent
print("Loading FaceNet model...")
embedder = FaceNet()

def get_embedding(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    return embedder.embeddings([img])[0]

# ====
print("\n Building embedding for:")
print(f" College: {COLLEGE_ID}")
print(f" Class  : {CLASS_ID}")
print(f" Person : {PERSON_NAME}\n")

embeddings = []

for file in os.listdir(REGISTERED_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        emb = get_embedding(os.path.join(REGISTERED_DIR, file))
        if emb is not None:
            embeddings.append(emb)

if len(embeddings) >= MIN_IMAGES:
    avg_emb = np.mean(embeddings, axis=0)
    save_path = os.path.join(EMB_DIR, f"{PERSON_NAME}.npy")
    np.save(save_path, avg_emb)
    print(f"OK Saved embedding: {PERSON_NAME}")
else:
    print(
        f"Skipped {PERSON_NAME} "
        f"(need {MIN_IMAGES} images, found {len(embeddings)})"
    )

print("\nOK Embedding build complete")
print(f"OK Saved to: {EMB_DIR}")
