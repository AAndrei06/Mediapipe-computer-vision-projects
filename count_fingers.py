import mediapipe as mp
import cv2 
import numpy as np
import math
from subprocess import call

wCam, hCam = 640, 480

myHands = mp.solutions.hands
hands = myHands.Hands()
myDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

fingers = [(8,6),(12,10),(16,14),(20,18)]
while True:
	success, img = cap.read()
	
	result = hands.process(img)
	if result.multi_hand_landmarks:
		for handLm in result.multi_hand_landmarks:
			count = 0
			lmList = handLm.landmark
			h,w,c = img.shape
			x1,y1 = int(lmList[4].x*w), int(lmList[4].y*h)
			x2,y2 = int(lmList[17].x*w), int(lmList[17].y*h)
			if math.hypot(x2-x1,y2-y1) > 80:
				count+=1
			
			for finger in fingers:
				if lmList[finger[0]].y < lmList[finger[1]].y:
					count+=1
					
			myDraw.draw_landmarks(img, handLm,myHands.HAND_CONNECTIONS)
			cv2.putText(img,f'Count: {str(count)}',(40,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),3)
			
	cv2.imshow('Img',img)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
		
		
cap.release()
cv2.destroyAllWindows()
	
