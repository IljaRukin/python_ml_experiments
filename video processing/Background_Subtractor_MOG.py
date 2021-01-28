import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
	ret, image_rgb = cap.read()
	
	#scale_percent = 0.5
	#width = int(image_rgb.shape[1] * scale_percent)
	#height = int(image_rgb.shape[0] * scale_percent)
	#dim = (width, height)
	#image_rgb = cv2.resize(image_rgb,dim)
	
	img_bw = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
	
	cv2.imshow('original',image_rgb)

	fgmask = fgbg.apply(image_rgb)

	cv2.imshow('modified',fgmask)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()