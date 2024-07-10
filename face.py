import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
	
	isTrue, img = cap.read()
	
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = faceDetection.process(imgRGB)
	print(results)
	
	if results.detections:
		for id, detection in enumerate(results.detections):
			mpDraw.draw_detection(img, detection)
			
	cv2.imshow('Image',img)
	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()
	 
	 
