Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import cv2
... from tensorflow.keras.models import load_model
... 
... # Load pre-trained face detection model (Haar cascade classifier)
... face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
... 
... # Load pre-trained emotion classification model (TensorFlow CNN)
... emotion_model = load_model('emotion_classification_model.h5')
... 
... # Define emotion labels (modify based on model's output classes)
... emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
... 
... def detect_and_classify_emotions(image):
...     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
... 
...     # Detect faces
...     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
... 
...     # Process each detected face
...     for (x, y, w, h) in faces:
...         # Extract the face region of interest (ROI)
...         face_roi = gray[y:y+h, x:x+w]
... 
...         # Preprocess the face ROI for emotion classification model (resize, normalize)
...         face_roi = cv2.resize(face_roi, (48, 48))  # Adjust dimensions as needed by your model
...         face_roi = face_roi.astype('float32') / 255.0
...         face_roi = np.expand_dims(face_roi, axis=0)  # Add extra dimension for batch processing
... 
...         # Predict emotion using loaded model
...         predictions = emotion_model.predict(face_roi)
...         predicted_emotion = emotion_labels[np.argmax(predictions)]
... 
...         # Draw bounding box and emotion label on the image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green bounding box
        cv2.putText(image, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image

# Example 
result_image = detect_and_classify_emotions(image.copy())
# Avoid modifying the original image
cv2.imshow('Emotion Detection', result_image)
cv2.waitKey(0)
