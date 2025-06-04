import cv2
import mediapipe as mp
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Helper function to detect if a finger is extended (based on joint y-positions)
def finger_is_extended(lms, tip, pip, mcp):
    return lms[tip].y < lms[pip].y < lms[mcp].y

# Gesture classification
def classify_gesture(lms):
    # Detect extended fingers
    index_up = finger_is_extended(lms, 8, 6, 5)
    middle_up = finger_is_extended(lms, 12, 10, 9)
    ring_up = finger_is_extended(lms, 16, 14, 13)
    pinky_up = finger_is_extended(lms, 20, 18, 17)

    thumb_tip_x = lms[4].x
    index_tip_x = lms[8].x

    # All fingers extended
    if all([index_up, middle_up, ring_up, pinky_up]):
        return "Stop"
    # Forward = only index up
    elif index_up and not middle_up:
        return "Forward"
    # Backward = only middle up
    elif middle_up and not index_up:
        return "Backward"
    # Direction by thumb (based on screen x-position)
    elif thumb_tip_x < index_tip_x:
        return "Right"
    elif thumb_tip_x > index_tip_x:
        return "Left"
    else:
        return "Unknown"

# Video capture
cap = cv2.VideoCapture(0)
last_gesture = None
last_print_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror for natural UX
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "None"
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            gesture = classify_gesture(handLms.landmark)

        # Print to terminal if changed
        current_time = time.time()
        if gesture != "Unknown" and gesture != last_gesture and current_time - last_print_time > 1:
            print(f"Detected gesture: {gesture}")
            last_gesture = gesture
            last_print_time = current_time

    # Display on-screen label
    cv2.putText(frame, f"Gesture:  {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
