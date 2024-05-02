import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Setup the paint window and color settings
paintWindow = np.zeros((471, 636, 3)) + 255
draw_color = (0, 0, 255)  # Blue color for drawing
drawing_points = deque(maxlen=512)  # Deque for storing drawing points

# Initialize the webcam
cap = cv2.VideoCapture(0)

def check_fingers_touching(finger1_tip, finger2_tip):
    """ Check if two finger tips are close enough to consider as touching. """
    distance = np.linalg.norm(np.array(finger1_tip) - np.array(finger2_tip))
    return distance < 20

def classify_shape(pts):
    """ Classify the shape of the contour formed by points. """
    contour = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        return "Rectangle"
    elif len(approx) > 4:
        return "Circle"
    return "Unknown"

drawing_active = False  # Flag to control whether drawing is active
previous_touching = False  # To check the previous state

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw buttons
    frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in handLms.landmark]
            index_tip = landmarks[8]  # Index finger tip
            thumb_tip = landmarks[4]  # Thumb tip

            currently_touching = check_fingers_touching(index_tip, thumb_tip)

            if currently_touching and not previous_touching and len(drawing_points) > 10:
                # When fingers just stop touching, classify shape
                shape_name = classify_shape(list(drawing_points))
                print(f"Detected shape: {shape_name}")
                drawing_points.clear()  # Optionally clear points after shape is classified

            if not currently_touching and not (40 <= index_tip[0] <= 140 and index_tip[1] <= 65):
                drawing_active = True
            elif 40 <= index_tip[0] <= 140 and index_tip[1] <= 65:
                drawing_points.clear()
                paintWindow[:, :, :] = 255

            if drawing_active and not currently_touching:
                drawing_points.append(index_tip)

            previous_touching = currently_touching

    # Draw points on the canvas
    for i in range(1, len(drawing_points)):
        if drawing_points[i - 1] is None or drawing_points[i] is None:
            continue
        cv2.line(frame, drawing_points[i - 1], drawing_points[i], draw_color, 2)
        cv2.line(paintWindow, drawing_points[i - 1], drawing_points[i], draw_color, 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
