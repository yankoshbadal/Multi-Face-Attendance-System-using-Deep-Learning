import subprocess
import sys
import os

PROJECT_DIR = r"C:\Users\yanko\OneDrive\Desktop\Desktop Files\present-me"
COLLEGES_DIR = os.path.join(PROJECT_DIR, "colleges")

#terminal input
COLLEGE_ID = "col_001"
CLASS_ID = "clsid001"
PERSON_NAME = "priyanshu"

#scripts
PRE_FACE_SCRIPT = os.path.join(PROJECT_DIR, "pre_face_emb.py")
IMG_AUG_SCRIPT = os.path.join(PROJECT_DIR, "image_augmentation.py")
BUILD_EMB_SCRIPT = os.path.join(PROJECT_DIR, "build_embeddings.py")

#---
def run_script(cmd):
    print(f"\n=> Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError("XX Pipeline failed")

    print("OK Finished")

#pipeline
try:
    #Crop raw images
    run_script([
        sys.executable,
        PRE_FACE_SCRIPT,
        COLLEGE_ID,
        CLASS_ID,
        PERSON_NAME
    ])

    #Augment cropped images
    run_script([
        sys.executable,
        IMG_AUG_SCRIPT,
        COLLEGE_ID,
        CLASS_ID,
        PERSON_NAME
    ])

    #Build embeddings
    run_script([
        sys.executable,
        BUILD_EMB_SCRIPT,
        COLLEGE_ID,
        CLASS_ID,
        PERSON_NAME
    ])

    print("\nBUILD PIPELINE COMPLETED SUCCESSFULLY")

except Exception as e:
    print(e)
    print("XX Pipeline stopped")
