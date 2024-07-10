import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(color = (0,255,0),thickness = 1, circle_radius = 1)

while True:
	
	isTrue, img = cap.read()
	
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = faceMesh.process(imgRGB)
	
	if results.multi_face_landmarks:
		for faceLms in results.multi_face_landmarks:
			mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)
			
			for id,lm in enumerate(faceLms.landmark):
				ih, iw, ic = img.shape
				x,y = int(lm.x*iw), int(lm.y*ih)
				if (id == 9):
					cv2.circle(img,(x,y),10,(0,0,255),cv2.FILLED)
				print(id,x,y)
	
	cv2.imshow('Img',img)
	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
		
		
cap.release()
cv2.destroyAllWindows()
