import cv2
import mediapipe as mp
import math
import joblib
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2.5
font_thickness = 3
text_color = (255, 255, 255)  # White

model_filename = 'hand_gesture_model.joblib'
model = joblib.load(model_filename)

while True:
    ret, frame = cap.read()

    image_height, image_width, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    hResults = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if hResults.multi_hand_landmarks:
        for num, hand in enumerate(hResults.multi_hand_landmarks):
            wrist_landmark = hand.landmark[mp_hands.HandLandmark.WRIST]
            distances = []

            for idx, landmark in enumerate(hand.landmark):
                dx = landmark.x - wrist_landmark.x
                dy = landmark.y - wrist_landmark.y
                dz = landmark.z - wrist_landmark.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                distances.append(distance)

            distances_array = np.array(distances).reshape(1, -1)

            predicted_alphabet = model.predict(distances_array)
            predicted_text = f'Predicted Alphabet: {predicted_alphabet[0]}'

            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                      )
            cv2.putText(image, predicted_text, (10, 70), font, font_scale, text_color, font_thickness)

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
