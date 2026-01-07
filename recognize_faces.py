import os
import sys

#CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import cv2
import json
import numpy as np
from datetime import datetime
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity


DEBUG = True  #False for no detailed print

#run_pipeline.py passes CLASS_DIR
CLASS_DIR = sys.argv[1]

EMB_DIR = os.path.join(CLASS_DIR, "embeddings")
DETECTED_DIR = os.path.join(CLASS_DIR, "detected")
OUTPUT_FILE = os.path.join(CLASS_DIR, "present_faces.json")

#settings
THRESHOLD = 0.75
IMAGE_SIZE = (160, 160)

#facenet
embedder = FaceNet()

#load embeddings
registered_embeddings = {}
for f in os.listdir(EMB_DIR):
    if f.endswith(".npy"):
        registered_embeddings[f.replace(".npy", "")] = np.load(
            os.path.join(EMB_DIR, f)
        )

if not registered_embeddings:
    raise RuntimeError("No embeddings found")

#load detected faces
face_images = []
face_files = []

for f in os.listdir(DETECTED_DIR):
    if not f.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img = cv2.imread(os.path.join(DETECTED_DIR, f))
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)

    face_images.append(img)
    face_files.append(f)

if not face_images:
    raise RuntimeError("No detected faces found")

face_images = np.array(face_images)

#group embeddings
face_embeddings = embedder.embeddings(face_images)

present_people = set()

#output matching results hai isko only for debug , upar false DEBUG
if DEBUG:
    print("FACE MATCHING RESULTS")

for idx, face_emb in enumerate(face_embeddings):
    if DEBUG:
        print("Detected face:", face_files[idx])

    for person, ref_emb in registered_embeddings.items():
        similarity = cosine_similarity(
            face_emb.reshape(1, -1),
            ref_emb.reshape(1, -1)
        )[0][0]

        if DEBUG:
            print(" ", person, "similarity =", round(similarity, 3))

        if similarity >= THRESHOLD:
            present_people.add(person)

#jason save
record = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "present_faces": sorted(list(present_people))
}

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"records": []}
else:
    data = {"records": []}

data["records"].append(record)

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=4)

print("Recognition completed")
