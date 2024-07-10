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
while True:
	success, img = cap.read()
	
	result = hands.process(img)
	if result.multi_hand_landmarks:
		for handLm in result.multi_hand_landmarks:
			lmList = handLm.landmark

			h,w,c = img.shape
			x1, y1 = int(lmList[4].x*w), int(lmList[4].y*h)
			x2, y2 = int(lmList[8].x*w), int(lmList[8].y*h)
			cx,cy = (x1+x2)//2, (y1+y2)//2
			cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
			cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
			cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
			cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
			myDraw.draw_landmarks(img, handLm,myHands.HAND_CONNECTIONS)
			length = math.hypot(x2-x1, y2-y1)
			#print(length)
			
			vol = np.interp(length, [50,300], [0,100])
			print(vol)
			call(["amixer", "-D", "pulse", "sset", "Master", "{}%".format(vol)])
			cv2.putText(img,f"{int(vol)}%", (50,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
			cv2.rectangle(img, (50,100),(85,300),(0,255,0),3)
			cv2.rectangle(img, (50,300),(85,300-int(vol)*2),(255,0,255),cv2.FILLED)
			if (length < 50):
				cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
			
	cv2.imshow('Img',img)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
		
		
cap.release()
cv2.destroyAllWindows()
	
