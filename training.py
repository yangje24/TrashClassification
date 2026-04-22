import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

from ultralytics import YOLO

# 1. Load the model (YOLO26 Nano is best for your Pi)
model = YOLO("yolo26m.pt") 

# 2. Start training
# 'data' points to your dataset's data.yaml file
# 'device=0' tells it to use your first NVIDIA GPU
# model.train(
#    data="/home/yangje/Garbage_Classification/TACO_YOLO/data.yaml", 
#    epochs=300, 
#    imgsz=1024, 
#    device=0,
#    batch=8,
#    optimizer='SGD',
#    lr0=0.01,
#    cos_lr=True,
#    momentum=0.937,
#    weight_decay=0.0005, # Reverted to standard to let the model learn naturally
#    patience=50,         # The only addition: stops training when validation loss degrades
#    mosaic=1.0,          
#    close_mosaic=10      
#)

model.train(
    data="/home/yangje/Garbage_Classification/TACO_YOLO/data.yaml", 
    epochs=100, 
    imgsz=1024, 
    device=0,
    batch=8,
    cache=False
    #amp=True
)