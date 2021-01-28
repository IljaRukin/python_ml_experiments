import numpy as np
import cv2
import scipy.fftpack
import time

#nehme Bilder auf
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('picturesofsu.gif')

#konvertiere in schwarz-weiß
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
gray_old=gray

while(True):
	#nehme Bilder auf
	#frame = cv2.imread('spatter_orig.jpg',0)
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray,(5,5),0)

	#zeige originalbild
	cv2.imshow('original',gray)

	#detektiere veränderte Pixel und überspringe bei kleiner Änderung (Frame-Fehler)
	mask = np.abs(gray_old.astype(np.int32) - gray.astype(np.int32)) > 5
	masked = np.multiply(gray,mask)
	
	if np.sum(masked)<5000 :
		continue

	#Berechne Gradienten
	#masked = cv2.Laplacian(gray,cv2.CV_64F)
	#masked = cv2.Sobel(gray,cv2.CV_64F,1,1,ksize=5)
	masked = np.sqrt( np.square(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)) + np.square(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)) )
	masked = 255*(masked-np.min(masked))/(np.max(masked)-np.min(masked))
	masked = masked.astype(np.uint8)

	#Zeige Gradientenbild
	cv2.imshow('sobel gradient',masked)

	#Detektion mit Hough-Circle-Transform (Gradientenbild)
	circles = cv2.HoughCircles(masked,cv2.HOUGH_GRADIENT,dp=1,minDist=1,circles=10000,
								param1=20,param2=12,minRadius=3,maxRadius=10)

	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		try:
			# draw the outer circle
			cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1)
			# draw the center of the circle
			cv2.circle(frame,(i[0],i[1]),2,(0,0,255),1)
		except:
			pass
	#zeige Bild mit detektierten Partikeln
	cv2.imshow('detected sobel gradient',frame)

	gray_old=gray

	#time.sleep(1)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()