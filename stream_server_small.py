import cv2
import zmq
import time
import serial
from picamera2 import Picamera2
from libcamera import controls

#PC_IP_ADDRESS = "172.21.238.113"
PC_IP_ADDRESS = "10.227.24.113"

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 9600

try:
	ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
	print(f"Connected to serial port: {SERIAL_PORT}")
except Exception as e:
	print(f"Warning: Could not open serial port {SERIAL_PORT}. Error {e}")
	ser = None

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(f"tcp://{PC_IP_ADDRESS}:5555")

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "BGR888"})
picam2.configure(config)
picam2.start()
time.sleep(2.0)

picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

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
		response = socket.recv_json()
		if isinstance(response, dict):
			detections = response.get("detections", [])
			command = response.get("command", None)
		else:
			detections = response
			command = None

		if detections:
			print(f"Detected {len(detections)} object(s).")

		if command:
			print(f"Command received: {command.strip()}")
			if ser is not None:
				ser.write(command.encode('utf-8'))

except KeyboardInterrupt:
	print("\nStopping stream...")

finally:
	picam2.stop()
	socket.close()
	context.term()
	if ser is not None:
		ser.close()
