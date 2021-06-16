# cam.py; written by Javed Roshan
import numpy as np
import cv2

# the index depends on your camera setup and which one is your USB camera.
# you may need to change to 1 depending on your local config

camSet='udpsrc port=5000 ! application/x-rtp,media=video,width=640,height=480,payload=96,clock-rate=90000,encoding-name=H265 ! rtph265depay ! h265parse ! omxh265dec ! videoconvert ! appsink'
cap= cv2.VideoCapture(camSet)

#cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
