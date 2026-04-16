import cv2
import zmq
import time
from picamera2 import Picamera2

PC_IP_ADDRESS = "10.0.0.168"

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(f"tcp://{PC_IP_ADDRESS}:5555")

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (1024, 1024), "format": "BGR888"})
picam2.configure(config)
picam2.start()
time.sleep(2.0)

print(f"Streaming via Picamera2 to {PC_IP_ADDRESS} and waiting for text data...")

try:
	while True:
		frame = picam2.capture_array()
		if frame is None or frame.size == 0:
			print("Warning: Captured empty frame. Skipping")
			continue
		frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		_, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
		socket.send(buffer.tobytes())
		detections = socket.recv_json()
		if detections:
			print(f"Detected: {detections}")

except KeyboardInterrupt:
	print("\nStopping stream...")

finally:
	picam2.stop()
	socket.close()
	context.term()
