import cv2
import numpy as np

# Load pre-trained Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():
    # Capture each frame from the video feed
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to grayscale for better accuracy with the classifiers
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(grayscale_frame, scaleFactor = 1.1, minNeighbors = 4)
    
    # Iterate through each detected face
    for (face_x, face_y, face_width, face_height) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, 
                      pt1 = (face_x, face_y), 
                      pt2 = (face_x + face_width, face_y + face_height), color = (0, 255, 0), thickness = 2)

        # Define regions of interest (ROI) for grayscale and color frames
        face_region_gray = grayscale_frame[face_y : face_y + face_height, face_x : face_x + face_width]
        face_region_color = frame[face_y : face_y + face_height, face_x : face_x + face_width]
        
        # Detect eyes within the detected face region
        eyes = eye_cascade.detectMultiScale(face_region_gray)
        
        # Iterate through each detected eye
        for (eye_x, eye_y, eye_width, eye_height) in eyes:
            # Draw a rectangle around each detected eye
            cv2.rectangle(face_region_color, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), color = (0, 255, 0), thickness = 2)

    # Display the frame with the rectangles
    cv2.imshow("Face and Eye Detection", frame)
    
    # Break the loop if the user presses the 'q' key (Quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
