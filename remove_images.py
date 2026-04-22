import os
import shutil

# Paths to your current training data
labels_dir = "/home/yangje/Garbage_Classification/TACO_YOLO/train/labels"
images_dir = "/home/yangje/Garbage_Classification/TACO_YOLO/train/images"

# Where to put the empty ones so they don't get deleted forever
backup_dir = "/home/yangje/Garbage_Classification/TACO_YOLO/background_backup"
os.makedirs(os.path.join(backup_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(backup_dir, "labels"), exist_ok=True)

removed_count = 0

for label_filename in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_filename)
    
    # Check if the text file is completely empty
    if os.path.getsize(label_path) == 0:
        image_filename = label_filename.replace('.txt', '.jpg') # Change to .png if necessary
        image_path = os.path.join(images_dir, image_filename)
        
        # Move the empty label and the image out of the training pipeline
        shutil.move(label_path, os.path.join(backup_dir, "labels", label_filename))
        if os.path.exists(image_path):
            shutil.move(image_path, os.path.join(backup_dir, "images", image_filename))
            
        removed_count += 1

print(f"Removed {removed_count} pure background images. Dataset is balanced again.")