import numpy as np
import cv2

# This example does not work on Ubuntu
# For video capture on ubuntu cheeck cam_ubuntu.py

cap = cv2.VideoCapture(0)

printed = False

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[:,::-1]
    cv2.imshow('frame',gray)
    if not printed:
        printed = True
        print(frame.shape)
        print(type(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
