import cv2
import mediapipe as mp
import numpy as np
import math


THICKNESS = 10
BRUSH = 30

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

MyHands = mp.solutions.hands
hands = MyHands.Hands()
myDraw = mp.solutions.drawing_utils


red = cv2.imread('pic4.png')
green = cv2.imread('pic3.png')
blue = cv2.imread('pic2.png')
eraser = cv2.imread('pic1.png')
curent = 3
pencil_list = [red,green,blue,eraser]
color_list = [(0,0,255),(0,255,0),(255,0,0),(0,0,0)]
distances = [(0,175),(175,350),(350,525),(525,700)]

blank = np.zeros((700,1200,3),np.uint8)

x_prev, y_prev = -1,-1


while True:
	isTrue, img = capture.read()
	img = cv2.resize(img,(1200,700))
	img = cv2.flip(img,1)
	height,width,channels = img.shape
	result = hands.process(img)
	if result.multi_hand_landmarks:
		for handLms in result.multi_hand_landmarks:
			lms = handLms.landmark
			cx, cy = int(lms[8].x*width),int(lms[8].y*height)
			
			for id, distance in enumerate(distances):
				if distance[0] < cy and distance[1] > cy and cx < 150:
					curent = id
					
			if curent == 3:
				cv2.circle(img,(cx,cy),BRUSH,(126,126,126),cv2.FILLED)
				
			
			x1,y1 = int(lms[12].x*width), int(lms[12].y*height)
			length =  math.hypot(abs(cx-x1),abs(cy-y1))
			if x_prev == -1 and y_prev == -1:
				x_prev = cx
				y_prev = cy
				continue
			
			
			if length > 100 and cx > 160:
				if curent == 3:
					cv2.line(blank,(x_prev,y_prev),(cx,cy),color_list[curent],thickness = BRUSH*2)
				else:
					cv2.line(blank,(x_prev,y_prev),(cx,cy),color_list[curent],thickness = THICKNESS)
				
			x_prev = cx
			y_prev = cy		
		
			myDraw.draw_landmarks(img,handLms,MyHands.HAND_CONNECTIONS)
	h,w,c = pencil_list[curent].shape
	img[0:h,0:w] = pencil_list[curent]
	
	
	imgGray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
	thresh, imgInv1 = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
	imgInv = cv2.cvtColor(imgInv1, cv2.COLOR_GRAY2BGR)
	img1 = cv2.bitwise_and(img,imgInv)
	img = cv2.bitwise_or(img,blank) 
	
	cv2.imshow('Img',img)
	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
		

capture.release()
cv2.destroyAllWindows()
