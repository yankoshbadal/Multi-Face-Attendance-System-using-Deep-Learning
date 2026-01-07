import os

BASE_PATH = r"C:\Users\yanko\OneDrive\Desktop\Desktop Files\present-me\colleges"

#terminal input
college_name = input("Enter college name: ").strip()
class_name = input("Enter class name: ").strip()
student_name = input("Enter student name: ").strip()

#path creation
class_path = os.path.join(BASE_PATH, college_name, class_name)

folders_to_create = [
    os.path.join(class_path, "raw_faces_imgs", student_name),
    os.path.join(class_path, "registered_faces", student_name),
    os.path.join(class_path, "registered_faces_augmented", student_name),
]

#folder creation
for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)

#last print
print("\nAll required folders are ready.")
print("Existing folders were left untouched.")