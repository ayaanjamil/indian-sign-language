import cv2
import mediapipe as mp
import math
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

csv_file_path = 'hand_data_alphabets.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['Alphabet', 'Frame', 'Hand', 'Landmark', 'X', 'Y', 'Z', 'Distance']
    writer.writerow(header)

    frame_number = 0

    for alphabet in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        print(f"Show '{alphabet}' on the screen.")
        input("Press Enter when ready...")

        for _ in range(20):
            ret, frame = cap.read()
            frame_number += 1

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

                        hand_label = "Right" if hResults.multi_handedness[num].classification[0].label == "Right" else "Left"
                        row = [alphabet, frame_number, hand_label, f'Landmark {idx}', landmark.x, landmark.y, landmark.z, distance]
                        writer.writerow(row)

                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                  )

                    print(f"Distances for Hand {hand_label}: {distances}")

            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
