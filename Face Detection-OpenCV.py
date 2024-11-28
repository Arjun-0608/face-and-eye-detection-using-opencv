import cv2
import numpy as np

# Load the pre-trained face detection classifier
face_cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default webcam (camera index 0)
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    is_frame_captured, current_frame = video_capture.read()
    
    # Convert the frame to grayscale for face detection
    grayscale_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    detected_faces = face_cascade_classifier.detectMultiScale(grayscale_frame)
    
    # Draw green rectangles around detected faces
    for (face_x, face_y, face_width, face_height) in detected_faces:
        cv2.rectangle(current_frame, (face_x, face_y), (face_x + face_width, face_y + face_height), (0, 255, 0), 3)  # Green color,3  Rectangle line thickness
    
    # Display the frame with detected faces
    cv2.imshow("Face Detection", current_frame)
    
    # Exit the loop if 'q' is pressed (Quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()