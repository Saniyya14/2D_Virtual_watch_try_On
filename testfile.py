import cv2
import mediapipe as mp
import time
import WristtrackingModule as wtm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = wtm.handDetector()

while True:
    # Capture a frame from the video stream.
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)
    if len(lmList) != 0:
        print(lmList[0])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the frame.
    cv2.imshow('Video Stream', frame)

    # Break the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream.
cap.release()

# Close the display window.
cv2.destroyAllWindows()