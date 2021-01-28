import numpy as np
import cv2

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

while(True):
    ret0, frame0 = cap0.read()
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame0',gray0)
    cv2.imshow('frame0',frame0)
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()