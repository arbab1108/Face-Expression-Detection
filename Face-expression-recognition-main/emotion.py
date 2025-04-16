import cv2
import threading
import time
import matplotlib.pyplot as plt
from deepface import DeepFace
from collections import defaultdict
from mtcnn import MTCNN
import numpy as np

# Load MTCNN face detector
detector = MTCNN()

# Start capturing video
cap = cv2.VideoCapture(0)

# Emotion data storage
emotion_counts = defaultdict(int)
emotion_history = []

# Function to update and display Matplotlib graph
def update_graph():
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    
    while True:
        if emotion_counts:
            ax.clear()
            ax.bar(emotion_counts.keys(), emotion_counts.values(), color='blue')
            ax.set_xlabel("Emotions")
            ax.set_ylabel("Count")
            ax.set_title("Real-time Emotion Statistics")
            plt.pause(1)  # Refresh every second
    
    plt.ioff()
    plt.show()

# Start a separate thread for the graph
thread = threading.Thread(target=update_graph, daemon=True)
thread.start()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (MTCNN requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame using MTCNN
    faces = detector.detect_faces(rgb_frame)
    
    for face in faces:
        x, y, w, h = face['box']
        
        # Ensures bounding box coordinates are within valid range
        x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
        
        # Extracts the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Performs emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determining the dominant emotion
        emotion = result[0]['dominant_emotion']
        emotion_counts[emotion] += 1  # Update emotion count
        emotion_history.append(emotion)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Displays the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
