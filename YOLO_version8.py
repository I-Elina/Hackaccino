
import cv2
from ultralytics import YOLO
import pyttsx3  # Offline TTS (no internet needed)
import time
import numpy as np

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")  # Or yolov8s.pt, yolov8m.pt, etc.

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech (words per minute)

# Initialize webcam
cap = cv2.VideoCapture(0)
print("Starting real-time detection. Press 'q' to quit...")

# Variables to control speech frequency
last_spoken_time = 0
speak_cooldown = 1  # Seconds between voice announcements

# Timing control
last_scan_time = time.time()
scan_interval = 10  # Scan every 10 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, verbose=False)

    # Get detected objects
    detections = results[0].boxes.cls.tolist()
    class_names = [model.names[int(cls)] for cls in detections]

    # Initialize detection flags
    wall_detected = False
    glass_detected = False
    
    # Wall detection (edge-based)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
    if lines is not None:
        wall_detected = True
        class_names.append("wall")

    # Glass detection (YOLO segmentation)
    seg_model = YOLO("yolov8n-seg.pt")
    glass_results = seg_model(frame, classes=[39,40,41], verbose=False)
    if any(int(cls) in [39,40,41] for cls in glass_results[0].boxes.cls.tolist()):
        glass_detected = True
        class_names.append("glass")

    # # Print to console
    # if class_names:
    #     print("Detected:", ", ".join(class_names))

    # Speak detections (with cooldown to avoid spam)
    current_time = time.time()
    if class_names and (current_time - last_spoken_time) > speak_cooldown:
        engine.say(f"I see {', '.join(class_names)}")
        engine.runAndWait()
        last_spoken_time = current_time

    # Display annotated frame
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Additional glass/wall detection (runs in parallel)
    seg_model = YOLO("yolov8n-seg.pt")
    glass_results = seg_model(frame, classes=[39, 40, 41], verbose=False)  # Bottle(39), wine glass(40), cup(41)
    
    # Glass detection
    glass_detected = any(int(cls) in [39, 40, 41] for cls in glass_results[0].boxes.cls.tolist())
    
    # Wall detection (simple edge-based)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    wall_detected = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10) is not None
    
    # Add warnings to existing audio feedback
    if (current_time - last_spoken_time) > speak_cooldown:
        if glass_detected:
            class_names.append("glass")  # Add to existing detection list
        if wall_detected:
            class_names.append("wall")  # Add to existing detection list

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
