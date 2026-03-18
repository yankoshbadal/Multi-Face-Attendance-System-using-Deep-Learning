# run_build_pipeline.py
## Purpose of this File

This script works as the **automation pipeline** for your **Multi-Face Attendance System**.
It sequentially executes three important steps:
1.  **Face Detection & Cropping** (`pre_face_emb.py`)
2.  **Image Augmentation** (`image_augmentation.py`)
3.  **Embedding Generation** (`build_embeddings.py`)
These scripts are executed automatically using Python’s **subprocess module**.

## Explanation

1.`import subprocess`
This imports Python's **subprocess module**, which allows the program to **run other Python scripts or system commands** from within the current script.

2.`import sys`
This module provides access to **Python interpreter variables and functions**. In this script it is used for: `sys.executable` This ensures that the **same Python interpreter** running this script is used to run the other scripts.

3.`import os` The `os` module allows interaction with the **operating system**.
It is used here for:
-   Managing file paths
-   Creating platform-independent directory paths
-   Joining folder names safely

4.`PROJECT_DIR = r"C:\Users\yanko\OneDrive\Desktop\Desktop Files\present-me"`
This variable stores the **absolute path of the main project directory**.
The prefix `r` means **raw string**, which prevents Python from treating backslashes (`\`) as escape characters.

5.`COLLEGES_DIR = os.path.join(PROJECT_DIR, "colleges")`
This creates the path to the **colleges folder** inside the project.
`os.path.join()` safely combines directory names.

6.`COLLEGE_ID = "col_001"` 
   `CLASS_ID = "clsid001"`
   `PERSON_NAME = "yankosh"`
These should be terminal commands to identify the student folder.
colleges/col_001/clsid001/yankosh/

7.
```py
PRE_FACE_SCRIPT = os.path.join(PROJECT_DIR, "pre_face_emb.py")`
`IMG_AUG_SCRIPT  = os.path.join(PROJECT_DIR, "image_augmentation.py")`
`BUILD_EMB_SCRIPT  =  os.path.join(PROJECT_DIR, "build_embeddings.py")
   ```
These create full paths of all the script which will be run by this pipeline.

### 8. Function `def run_script(cmd):` Line 18-38
To **execute another script** and manage its output.
Parameter:
`cmd`This is the **command list** to be executed.
Example: ["python", "pre_face_emb.py", "col_001", "clsid001", "yankosh"]
**8. Line 19-25**
```py
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace"
)
```
Runs the command using the `subprocess.run()` function.
Parameters:
**cmd** Command to execute.
**capture_output=True** Captures both: standard output (stdout) and error output (stderr), instead of printing them directly.
**text=True** Returns output as **string instead of bytes**.
**encoding="utf-8"** Ensures output uses UTF-8 encoding.
**errors="replace"** If an encoding error occurs, it replaces invalid characters instead of crashing.

**9. Line 31-32**
```py
if result.stdout:
    print(result.stdout)
```
If the script prints anything to **standard output**, it will be displayed in the terminal.

**10. Line 33-36**
```py
if  result.returncode  !=  0:
	if  result.stderr:
		print(result.stderr)
	raise  RuntimeError("XX Pipeline failed")
```
Checks if the script **failed during execution**.  
- a. Code Meaning 0 Success Non-zero Error
- b. If there is an error message, 
- c. It prints the **standard error output**.
- d. Raises an exception to **stop the pipeline immediately**. This prevents the next steps from running if one step fails.

### Calling the above function
**11. Line 46-51**
```py
run_script([
    sys.executable,
    PRE_FACE_SCRIPT,
    COLLEGE_ID,
    CLASS_ID,
    PERSON_NAME
])
```
Runs the **face detection and cropping script**.
the array parameter passed is the **cmd** parameter
Equivalent command in terminal: 
python pre_face_emb.py col_001 clsid001 yankosh

**Similarly next 2 function calls execute the other 2 scripts**
