import cv2
import mediapipe as mp

# Load the watch image with a transparent background
watch_image = cv2.imread('Watch-PNG-HD.png', cv2.IMREAD_UNCHANGED)

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open the camera

while True:
    ret, frame = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for hand tracking
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract wrist landmark (adjust as needed based on your image)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Calculate wrist position in pixel coordinates
            h, w, _ = frame.shape
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            # Resize the watch image to fit the wrist
            watch_width = int(0.4 * w)  # Adjust the scale as needed
            watch_height = int(0.2 * h)  # Adjust the scale as needed
            watch_resized = cv2.resize(watch_image, (watch_width, watch_height))

            # Calculate coordinates for overlay
            x1 = wrist_x - watch_width // 2
            x2 = x1 + watch_width
            y1 = wrist_y - watch_height // 2
            y2 = y1 + watch_height

            # Check boundaries to avoid out-of-bounds errors
            if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                # Create a mask from the watch image's alpha channel
                watch_mask = watch_resized[:, :, 3]

                # Extract the RGB channels from the watch image
                watch_rgb = watch_resized[:, :, :3]

                # Calculate the region to overlay the watch
                frame_region = frame[y1:y2, x1:x2]

                # Use the mask to blend the watch image with the frame
                for c in range(3):
                    frame_region[:, :, c] = frame_region[:, :, c] * (1 - (watch_mask / 255.0)) + watch_rgb[:, :, c] * (watch_mask / 255.0)

    # Display the frame
    cv2.imshow('Virtual Watch Try-On', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


