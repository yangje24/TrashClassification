import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

print("1. Loading AI Brain (NCNN Model)...")

model = YOLO("testmod")

print("2. Initializing Pi Camera...")
picam2 = Picamera2()

picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))

print("3. Starting Live AI Inference... (Press 'q' in the video window to quit)")

try:
   picam2.start()

   while True:
      frame = picam2.capture_array()
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      results = model.predict(source=frame, show=False, conf=0.4)
      annotated_frame = results[0].plot()
      cv2.imshow("Trash Detection - Live AI", annotated_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         print("Quit command received. Wrapping up...")
         break

finally:
   print("Shutting down camera hardware safely...")
   picam2.stop()
   picam2.close()
   cv2.destroyAllWindows()
