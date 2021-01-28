import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fast = cv2.FastFeatureDetector_create(threshold = 20,nonmaxSuppression = True)

while(True):
	ret, image_rgb = cap.read()
	
	#scale_percent = 0.5
	#width = int(image_rgb.shape[1] * scale_percent)
	#height = int(image_rgb.shape[0] * scale_percent)
	#dim = (width, height)
	#image_rgb = cv2.resize(image_rgb,dim)
	
	img_bw = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
	
	cv2.imshow('original',image_rgb)

	#feature detection
	kp = fast.detect(image_rgb,None)
	img_features = cv2.drawKeypoints(image_rgb, kp, outImage = None, color=(255,0,0))

	cv2.imshow('modified',img_features)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()