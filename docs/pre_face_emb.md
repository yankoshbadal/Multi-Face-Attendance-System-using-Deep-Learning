# pre_face_emb.py
This script:
-   Take **raw images of a student**
-   Detect **faces using YOLOv8**
-   **Crop the faces**
-   Save cropped faces images. These cropped faces will later be used for **image augmentation**

1.`from ultralytics import YOLO` Imports the **YOLO model class** from the **Ultralytics library**.

2.`import cv2` Imports **OpenCV**, a library used for image processing.
Here it is used for:
-   Reading images
-   Cropping faces
-   Saving images

3.`import os` 

4.`import sys` The `sys` module allows access to **command-line arguments**.
This script expects inputs like:
**python pre_face_emb.py col_001 clsid001 yankosh**
These values are accessed using: sys.argv

**5. Line 8-11**
```py
if  len(sys.argv) <  4:
raise  RuntimeError("Usage: python pre_face_emb.py <college_id> <class_id> <person_name>")
```
a. Checks if enough arguments were provided. Minimum **4 elements** must exist.
`sys.argv` contains: script name-0, college id-1, class id-2, person name-3
b. If arguments are missing, the script stops and displays the correct usage.

5.`COLLEGE_ID  =  sys.argv[1]`
`CLASS_ID  =  sys.argv[2]`
`PERSON_NAME  =  sys.argv[3]`

Stores the **college id**, **class id** **student name** from the command line.

6.`BASE_DIR = r"C:\Users\yanko\OneDrive\Desktop\Desktop Files\present-me\colleges"` This is the **root directory of the dataset**.

7.`CLASS_DIR = os.path.join(BASE_DIR, COLLEGE_ID, CLASS_ID)` Creates the class directory path. 

8.`INPUT_DIR = os.path.join(CLASS_DIR, "raw_faces_imgs", PERSON_NAME)` Folder containing **original student images**.
colleges/col_001/clsid001/raw_faces_imgs/yankosh

9.`OUTPUT_DIR = os.path.join(CLASS_DIR, "registered_faces", PERSON_NAME)` Folder where **cropped face images** will be stored.
colleges/col_001/clsid001/**registered_faces/yankosh**

10.`os.makedirs(OUTPUT_DIR, exist_ok=True)` Creates the folder if it does not exist. `exist_ok=True` means: do not raise error if folder already exists

**11. Line 26-27**
```py
if  not  os.path.exists(INPUT_DIR):
raise  RuntimeError(f"Input folder not found: {INPUT_DIR}")
```
Stops the program if the input folder does not exist.

## Loading YOLO v8

12.`MODEL_PATH = os.path.join( os.path.dirname(__file__),"yolov8n-face.pt")`
Creates the path for the YOLO face detection model. `__file__` = location of the current script. **present-me/yolov8n-face.pt**

13.`model = YOLO(MODEL_PATH)` Loads the **YOLOv8**

14.`face_count = 0` Counts how many faces have been detected and saved.

15.`MARGIN = 0.25` Add more context around face.

**16. For Loop, Line 45-82---** **for  img_file  in  os.listdir(INPUT_DIR):**
1. Face Detection
```py
	if  not  img_file.lower().endswith((".jpg", ".jpeg", ".png")):
		continue
	img_path  =  os.path.join(INPUT_DIR, img_file)
	img  =  cv2.imread(img_path)
	if  img  is  None:
		continue
```
- Reads every file in the input folder.
- Skips files that are **not images**.
- Creates the full path to the image. raw_faces_imgs/yankosh/img1.jpg
- Reads the image using OpenCV.
- If image fails to load, skip it.
- 
2. Face Detection
`h, w, _ = img.shape` Gets the image dimensions.
`results = model(img)` Runs YOLO face detection on the image.
YOLO returns: bounding boxes, confidence scores, detected objects

3. Crop Results
```py
for  r  in  results:
	for  box  in  r.boxes:
		x1, y1, x2, y2  =  map(int, box.xyxy[0])
		
		bw  =  x2  -  x1
		bh  =  y2  -  y1
		
		mx  =  int(MARGIN  *  bw)
		my  =  int(MARGIN  *  bh)
		
		x1  =  max(0, x1  -  mx)
		y1  =  max(0, y1  -  my)
		x2  =  min(w, x2  +  mx)
		y2  =  min(h, y2  +  my)
		
		crop  =  img[y1:y2, x1:x2]
if  crop.size  ==  0:
	continue
```
**for box in r.boxes:** Each detected face has a **bounding box**.
**x1, y1, x2, y2 = map(int, box.xyxy[0])** Extracts face bounding box coordinates.
x1 = left
y1 = top
x2 = right
y2 = bottom
**bw = x2 - x1** Bounding box width.
**bw = x2 - x1**  Bounding box height.
**mx  =  int(MARGIN  *  bw)** Extra width margin.
**x1 = max(0, x1 - mx)** Prevents coordinates from going outside the image.
If cropping failed, skip.

4. Save the cropped images
```py
face_count  +=  1
cv2.imwrite(os.path.join(OUTPUT_DIR, f"face_{face_count}.png"),crop)
```
17.`print(f"\nPre Embd Faces cropped and saved: {face_count}")` Final output: Pre Embd Faces cropped and saved: 25.
