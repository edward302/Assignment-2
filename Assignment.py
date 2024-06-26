Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

# Load pre-trained face detection and emotion models 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
emotion_model = tf.keras.models.load_model('facial_emotion_recognition.h5')  

# Defining emotion categories 
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def detect_and_classify_emotions (https://github.com/edward302/Assignment-2/blob/main/20240529_153201.jpg):
  """Detects faces in an image and classifies their emotions.

  Args:
      image_path: https://github.com/edward302/Assignment-2/blob/main/20240529_153201.jpg

  Returns:
      A list of dictionaries, where each dictionary contains:
          - x1: Top-left x-coordinate of the bounding box.
          - y1: Top-left y-coordinate of the bounding box.
          - x2: Bottom-right x-coordinate of the bounding box.
          - y2: Bottom-right y-coordinate of the bounding box.
          - emotion: Predicted emotion label.
  """
  img = cv2.imread (https://github.com/edward302/Assignment-2/blob/main/20240529_153201.jpg) 
 # Read the image
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
# Convert to grayscale

  # Detect faces
  faces = face_cascade.detectMultiScale(gray, 1.1, 4)

  # Process each detected face
  face_data = []
  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw bounding box 

    # Extract face region of interest (ROI)
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48, 48))  # Resize to model's input size
    roi = roi.astype("float") / 255.0  # Normalize pixel values
    roi = np.expand_dims(roi, axis=0)  # Add a dimension for batch processing

    # Predict emotion
    prediction = emotion_model.predict(roi)[0]  # Get the probability vector
    emotion_label = emotions[np.argmax(prediction)]  # Get predicted emotion

    # Store face data
    face_data.append({
      "x1": x,
      "y1": y,
      "x2": x + w,
      "y2": y + h,
      "emotion": emotion_label
    })

  return face_data

# usage
image_path = https://github.com/edward302/Assignment-2/blob/main/20240529_153201.jpg

face_data = detect_and_classify_emotions(https://github.com/edward302/Assignment-2/blob/main/20240529_153201.jpg)

if face_data:
  for face in face_data:
    print(f"Face coordinates: ({face['x1']}, {face['y1']}), ({face['x2']}, {face['y2']})")
    print(f"Predicted emotion: {face['emotion']}")
else:
  print("No faces detected in the image.")


# Example 
result_image = detect_and_classify_emotions(image.copy())
# Avoid modifying the original image
cv2.imshow('Emotion Detection', result_image)
cv2.waitKey(0)
