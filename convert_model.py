from ultralytics import YOLO
model = YOLO("runs/detect/train8/weights/best.pt") # Point to your trained model
model.export(format="engine", half=True, dynamic=True)