# image_augmentation.py
## Purpose of this Script
1.  Takes **cropped face images**
2.  Applies different transformations (flip, brightness, rotation, etc.)
3.  Saves multiple versions of each image

1 .
``` py
import  os
import  sys
import  cv2
import  numpy  as  np
```
All imports: We have talked about this a lot.

2 .
``` py 
if  len(sys.argv) <  4:
	raise  RuntimeError("Usage: python image_augmentation.py <college_id> <class_id> <person_name>")
```
a. Checks if enough arguments were provided. Minimum **4 elements** must exist.
`sys.argv` contains: script name-0, college id-1, class id-2, person name-3
b. If arguments are missing, the script stops and displays the correct usage.

3 .
```py
COLLEGE_ID  =  sys.argv[1]
CLASS_ID  =  sys.argv[2]
PERSON_NAME  =  sys.argv[3]
```
Stores the **college id**, **class id** **student name** from the command line.

4 .
```py
BASE_DIR  =  r"C:\Users\yanko\Desktop\Desktop Files\present-me\colleges"
CLASS_DIR  =  os.path.join(BASE_DIR, COLLEGE_ID, CLASS_ID)
INPUT_DIR  =  os.path.join(CLASS_DIR, "registered_faces", PERSON_NAME)
OUTPUT_DIR  =  os.path.join(CLASS_DIR, "registered_faces_augmented", PERSON_NAME)
```
Creates paths for Class Folder, Input Folder, Output Folder.

5 .
```py
if  not  os.path.exists(INPUT_DIR):
raise  RuntimeError(f"Person folder not found: {INPUT_DIR}")
```
Stops the program if the input folder does not exist.

6 .
```py
os.makedirs(OUTPUT_DIR, exist_ok=True)
```
Make Output Folder if it does not exist.

7 .
```py
def  horizontal_flip(img):
return  cv2.flip(img, 1)
```
Function to flip image left ↔ right.

8 .
```py
def  change_brightness(img, value=25):
	hsv  =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v  =  cv2.split(hsv)
	v  =  np.clip(v  +  value, 0, 255)
	hsv  =  cv2.merge((h, s, v))
	return  cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```
a. Increases brightness.
b. Convert image to HSV color space.
c. Split channels.
d. Increase brightness safely.
e. Merge channels.
f. Convert back to BGR.

9 .
```py
def  change_contrast(img, alpha=1.25):
	return  cv2.convertScaleAbs(img, alpha=alpha, beta=0)
```
Adjusts contrast. Using Formula: **new_pixel = alpha * pixel + beta**

10 .
```py
def  slight_rotation(img, angle=7):
	h, w  =  img.shape[:2]
	M  =  cv2.getRotationMatrix2D((w  //  2, h  //  2), angle, 1.0)
	return  cv2.warpAffine(img, M, (w, h),borderMode=cv2.BORDER_REFLECT)
```
a. Rotates image slightly.
b. Gets height and width.
c. Creates rotation matrix.
d. Applies rotation. `BORDER_REFLECT` avoids black edges.

11 .
```py
print(f"Augmenting images for: {PERSON_NAME}")
print(f"Input folder : {INPUT_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
```
Displays processing info.

**12 .** Loop: Through all images.
```py
for  file  in  os.listdir(INPUT_DIR):
	if  not  file.lower().endswith((".jpg", ".jpeg", ".png")):
		continue
```
Reads all files. Skips non-image files.
```py 
img_path = os.path.join(INPUT_DIR, file)
img = cv2.imread(img_path)
```
Loads image after reading.
```py
if img is None:
    continue
```
Skips invalid images.

`name, ext  =  os.path.splitext(file)` Split Name & Extension: face_1.jpg → name=face_1 , ext=.jpg

```py 
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_orig{ext}"), img)
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_flip{ext}"), horizontal_flip(img))
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_bright{ext}"), change_brightness(img))
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_contrast{ext}"), change_contrast(img))
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_rotate{ext}"), slight_rotation(img))
```
Apply all functions on each image and saves output.
