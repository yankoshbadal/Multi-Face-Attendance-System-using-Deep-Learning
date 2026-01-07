import subprocess
import sys
import os


PROJECT_DIR = r"C:\Users\yanko\OneDrive\Desktop\Desktop Files\present-me"

COLLEGE_ID = "col_001"
CLASS_ID = "clsid001"

CLASS_DIR = os.path.join(
    PROJECT_DIR,
    "colleges",
    COLLEGE_ID,
    CLASS_ID
)

FACE_DETECT_SCRIPT = os.path.join(PROJECT_DIR, "face_detect.py")
RECOGNIZE_SCRIPT = os.path.join(PROJECT_DIR, "recognize_faces.py")

def run_script(cmd):
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
        raise RuntimeError("Pipeline failed")

try:
    run_script([sys.executable, FACE_DETECT_SCRIPT, CLASS_DIR])
    run_script([sys.executable, RECOGNIZE_SCRIPT, CLASS_DIR])
    print("Pipeline completed successfully")
except Exception as e:
    print(e)
    print("Pipeline stopped")
