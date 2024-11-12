import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Load model
model = load_model("E:/Python/Flask_Project/Face_Emotion_setection/facialemotionmodel.h5")

# Haar cascade file for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to preprocess input image
def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

# Access webcam
webcam = cv2.VideoCapture(0)

# Set desired output screen size (laptop size)
output_width = 1366
output_height = 768

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    # Resize frame to desired output dimensions
    frame = cv2.resize(frame, (output_width, output_height))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Prepare to display emotion probabilities
    emotion_probabilities = []

    for (x, y, w, h) in faces:
        face_image = gray[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (48, 48))
        face_image = extract_features(face_image)
        
        # Predict the emotion and get probabilities
        pred = model.predict(face_image)
        confidence = np.max(pred) * 100  # Confidence in percentage
        emotion_label = labels[pred.argmax()]

        # Collect emotion probabilities for display
        emotion_probabilities = pred.flatten() * 100  # Convert to percentage

        # Draw rectangle and display text with emotion and confidence percentage
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label_text = f"{emotion_label}: {confidence:.2f}%"
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Create a black area for displaying probabilities
    prob_display_area = np.zeros((100, frame.shape[1], 3), dtype=np.uint8)
    
    # Display each emotion percentage
    for i, (emotion, probability) in enumerate(zip(labels.values(), emotion_probabilities)):
        text = f"{emotion}: {probability:.2f}%"
        cv2.putText(prob_display_area, text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Overlay the probability display area on the frame
    frame[0:100, 0:frame.shape[1]] = prob_display_area

    # Show the output frame with probabilities
    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
