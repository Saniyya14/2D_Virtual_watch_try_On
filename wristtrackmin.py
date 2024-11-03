import cv2
import mediapipe as mp
import time

# Open the video stream.
cap = cv2.VideoCapture(0)  # Use 0 for the default camera.


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    # Capture a frame from the video stream.
    ret, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
           for id, lm in enumerate(handLms.landmark):
               #print(id, lm)
               h, w, c = frame.shape
               cx, cy = int(lm.x*w), int(lm.y*h)
               #print(id, cx, cy)
               if id == 0:
                   cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                   print(id, cx, cy)


        mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)




    cTime = time.time()
    fps = 1/(cTime-pTime)
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

