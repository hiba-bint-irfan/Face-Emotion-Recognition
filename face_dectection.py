import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

model = load_model("model\emotion_model.h5")
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Happy", 3: "Normal", 4: "Surprised", 5: "Yawning"}

def predict_emotion(face_image):
    if len(face_image.shape) == 2:
        pass
    else:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (48, 48))
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    face_image = face_image / 255.0
    emotion = np.argmax(model.predict(face_image))
    return emotion_dict[emotion]

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces in the frame
        results = face_detection.process(rgb_frame)
        # Draw the face detection annotations on the frame
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
                ih, iw, _ = frame.shape
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face = frame[y:y+h, x:x+w]
                if face.size > 0:
                    emotion = predict_emotion(face)
                    cv2.putText(frame, emotion, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    emotion = "No Face"
                    cv2.putText(frame, emotion, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Display the frame
        cv2.imshow('MediaPipe Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
