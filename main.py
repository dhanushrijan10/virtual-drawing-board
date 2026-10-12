import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

canvas = None
previous_x, previous_y = None, None

brush_thickness = 6
eraser_thickness = 80

def fingers_up(hand_landmarks):
    landmarks = hand_landmarks.landmark

    thumb = landmarks[4].x < landmarks[3].x
    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y

    return [thumb, index, middle, ring, pinky]

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    mode = "Show your hand"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = hand_landmarks.landmark
        fingers = fingers_up(hand_landmarks)

        x = int(landmarks[8].x * width)
        y = int(landmarks[8].y * height)

        palm_x = int(landmarks[9].x * width)
        palm_y = int(landmarks[9].y * height)

        open_palm = all(fingers)

        if open_palm:
            mode = "ERASER"
            cv2.circle(
                frame,
                (palm_x, palm_y),
                eraser_thickness // 2,
                (0, 0, 255),
                2
            )

            cv2.circle(
                canvas,
                (palm_x, palm_y),
                eraser_thickness // 2,
                (0, 0, 0),
                cv2.FILLED
            )

            previous_x, previous_y = None, None

        elif fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            mode = "DRAW"
            cv2.circle(frame, (x, y), 8, (0, 255, 0), cv2.FILLED)

            if previous_x is not None and previous_y is not None:
                cv2.line(
                    canvas,
                    (previous_x, previous_y),
                    (x, y),
                    (0, 255, 0),
                    brush_thickness
                )

            previous_x, previous_y = x, y

        else:
            mode = "PAUSED"
            previous_x, previous_y = None, None

    else:
        previous_x, previous_y = None, None

    combined = cv2.addWeighted(frame, 0.7, canvas, 1, 0)

    cv2.putText(
        combined,
        f"Mode: {mode}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        combined,
        "Index finger: Draw | Open palm: Erase | C: Clear | Q: Quit",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow("Virtual Drawing Board", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas = np.zeros_like(frame)
        previous_x, previous_y = None, None

    elif key == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
