import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
#cap=cv2.VideoCapture(0)
cap = cv2.VideoCapture('face.mp4')

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray,1.3,5)
	id=0

	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),	(255,0,0),2)
		roi_gray=gray[y:y+h,x:x+w]
		roi_color= img[y:y+h,x:x+w]
		id=id+1
		# eyes=eye_cascade.detectMultiScale(roi_gray)
		cv2.putText(roi_color, str(id), (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA)
		# for(ex,ey,ew,eh) in eyes:
		# 	cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		# 	roi_gray=gray[ey:ey+eh,ex:ex+ew]
		# 	roi_color= img[ey:ey+eh,ex:ex+ew]

	cv2.imshow('img',img)
	k=cv2.waitKey(30) & 0xff
	if k==27:
		break

cap.release()
cv2.destroyAllWindows()

