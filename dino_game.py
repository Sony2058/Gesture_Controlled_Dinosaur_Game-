import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_fingers(hand_landmarks):
    fingers = [8, 12, 16, 20]  # Indices of fingertips
    count = 0

    for fingertip in fingers:
        # Check if the fingertip is above the corresponding lower joint
        if hand_landmarks.landmark[fingertip].y < hand_landmarks.landmark[fingertip - 2].y:
            count += 1

    # Check thumb separately
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:  # Left hand thumb
        count += 1

    return count

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video. Exiting...")
        break

    # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count raised fingers
            finger_count = count_fingers(hand_landmarks)

            # Map gestures to game controls
            if finger_count == 1:  # One finger raised
                pyautogui.press('space')  # Jump
                print("Jump!")
            elif finger_count == 2:  # Two fingers raised
                pyautogui.press('down')  # Duck (or stay still)
                print("Duck!")

    # Display the webcam feed
    cv2.imshow('Gesture Control - Dinosaur Game', frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
