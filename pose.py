import mediapipe as mp
import cv2

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)

while True:
	isTrue, img = cap.read()
	imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	results = pose.process(imgRGB)
	if results.pose_landmarks:
		mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
		for id, lm in enumerate(results.pose_landmarks.landmark):
			h,w,c = img.shape
			
			cx,cy = int(lm.x*w), int(lm.y*h)
			cv2.circle(img, (cx,cy), 10, (255,0,0),cv2.FILLED)
		
	cv2.imshow('Img',img)
	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()


