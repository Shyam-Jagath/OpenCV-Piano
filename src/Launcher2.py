import cv2
import mediapipe as mp
import numpy as np
import os
import time
from play_notes import SoundPlayer

# Notes assigned to fingertips for left and right hands
LEFT_HAND_NOTES = ['C4', 'D4', 'E4', 'F4', 'G4']
RIGHT_HAND_NOTES = ['A4', 'B4', 'C5', 'D5', 'E5']

# Webcam settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Mediapipe face mesh for eye detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Mediapipe hand tracking with support for two hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.9)

# Sound player
player = SoundPlayer(LEFT_HAND_NOTES + RIGHT_HAND_NOTES)

# Tracking state
eye_closed_start_time = None
eye_closed_duration = 0
EYE_CLOSED_THRESHOLD = 3  # in seconds
finger_pressed_left = [False] * 5  # Track pressed state for left hand fingers
finger_pressed_right = [False] * 5  # Track pressed state for right hand fingers
previous_positions_left = [None] * 5  # Previous fingertip positions for left hand
previous_positions_right = [None] * 5  # Previous fingertip positions for right hand
desk_edge_y = None  # Desk edge position

# Function to check if eyes are closed
def are_eyes_closed(landmarks, frame_width, frame_height):
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    def get_eye_aspect_ratio(eye_landmarks):
        # Calculate the distances between the vertical eye landmarks
        vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        # Calculate the distance between the horizontal eye landmarks
        horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        # Calculate the eye aspect ratio
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    left_eye_landmarks = [(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in left_eye_indices]
    right_eye_landmarks = [(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in right_eye_indices]

    left_ear = get_eye_aspect_ratio(left_eye_landmarks)
    right_ear = get_eye_aspect_ratio(right_eye_landmarks)

    EYE_AR_THRESHOLD = 0.2
    return left_ear < EYE_AR_THRESHOLD and right_ear < EYE_AR_THRESHOLD

# Exponential smoothing function
def smooth_position(current, previous, alpha=0.3):
    if previous is None:
        return current
    return tuple(alpha * c + (1 - alpha) * p for c, p in zip(current, previous))

# Detect desk edge using Hough Line Transform
def detect_desk_edge(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 5:
                return (y1 + y2) // 2  
    return None

# Calculate velocity for a fingertip
def calculate_velocity(current, previous):
    if current is None or previous is None:
        return 0
    return np.linalg.norm(np.array(current) - np.array(previous))

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the video feed
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the face mesh
    face_results = face_mesh.process(rgb_frame)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            if are_eyes_closed(face_landmarks.landmark, frame.shape[1], frame.shape[0]):
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                else:
                    eye_closed_duration = time.time() - eye_closed_start_time
                    if eye_closed_duration >= EYE_CLOSED_THRESHOLD:
                        new_folder = os.path.join(os.path.dirname(__file__), "..", "resources", "New Sounds")
                        player.change_sounds_folder(new_folder)
                        eye_closed_start_time = None  # Reset the timer
            else:
                eye_closed_start_time = None
                eye_closed_duration = 0

    # Process the frame with the hand tracking
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Determine which hand is being tracked
            hand_label = hand_results.multi_handedness[hand_idx].classification[0].label
            hand_type = "LEFT" if hand_label == "Left" else "RIGHT"
            hand_notes = LEFT_HAND_NOTES if hand_type == "LEFT" else RIGHT_HAND_NOTES
            finger_pressed = finger_pressed_left if hand_type == "LEFT" else finger_pressed_right
            previous_positions = previous_positions_left if hand_type == "LEFT" else previous_positions_right

            # Extract fingertip positions for all fingers
            h, w, _ = frame.shape
            fingertip_indices = [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
            ]
            fingertips = []
            for idx, landmark_idx in enumerate(fingertip_indices):
                landmark = hand_landmarks.landmark[landmark_idx]
                fingertips.append((int(landmark.x * w), int(landmark.y * h)))

            # Smooth and calculate velocity
            velocities = []
            for i, pos in enumerate(fingertips):
                smoothed_pos = smooth_position(pos, previous_positions[i])
                velocity = calculate_velocity(smoothed_pos, previous_positions[i])
                previous_positions[i] = smoothed_pos
                velocities.append(velocity)

            # Update finger_pressed based on velocity
            for i, velocity in enumerate(velocities):
                if velocity > 5:  # Adjust the threshold as needed
                    finger_pressed[i] = True
                else:
                    finger_pressed[i] = False

            # Play the sounds if multiple fingers are detected and moving
            for i, pos in enumerate(fingertips):
                if i < len(hand_notes) and finger_pressed[i]:
                    player.play_note_by_index(i)

            # Draw fingertips and note names
            for i, pos in enumerate(fingertips):
                color = (0, 255, 0) if finger_pressed[i] else (0, 0, 255)
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 10, color, -1)
                cv2.putText(frame, hand_notes[i], (int(pos[0]) + 10, int(pos[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw desk edge
    if desk_edge_y is not None:
        cv2.line(frame, (0, desk_edge_y), (frame.shape[1], desk_edge_y), (255, 255, 0), 2)

    cv2.putText(frame, "Advanced Virtual Piano", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Virtual Piano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()