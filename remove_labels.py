import os
import glob

# 1. Define the path to your label folders (do this for train, val, and test if you have them)
label_dirs = [
    "/home/yangje/Garbage_Classification/TACO_YOLO/train/labels",
    "/home/yangje/Garbage_Classification/TACO_YOLO/val/labels"
]

# 2. Put the IDs of the classes you want to delete here
classes_to_remove = ['1', '3', '6', '14', '15', '16'] # Example: Bottle cap, Cigarette, Pop tab

removed_count = 0

for label_dir in label_dirs:
    # Find all .txt files in the directory
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    for file_path in txt_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Keep only the lines where the first number (Class ID) is NOT in our removal list
        clean_lines = [line for line in lines if line.split()[0] not in classes_to_remove]
        
        # If we removed something, update the count
        removed_count += (len(lines) - len(clean_lines))
        
        # Overwrite the file with the cleaned data
        with open(file_path, 'w') as f:
            f.writelines(clean_lines)

print(f"Done! Successfully removed {removed_count} annotations.")